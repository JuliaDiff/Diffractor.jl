using Core.IR
using .CC:
    BasicBlock, CFG, IRCode, IncrementalCompact, Instruction, NewInstruction, NoCallInfo, StmtRange,
    bbidxiter, cfg_delete_edge!, cfg_insert_edge!, compute_basic_blocks, complete,
    construct_domtree, construct_ssa!, domsort_ssa!, finish, insert_node!,
    insert_node_here!, non_dce_finish!, quoted, retrieve_code_info,
    scan_slot_def_use, userefs, SimpleInferenceLattice

if VERSION < v"1.11.0-DEV.1351"
    using .CC: effect_free_and_nothrow as removable_if_unused
else
    using .CC: removable_if_unused
end

using Base.Meta

cname(nc, N, name) = Symbol(string("∂⃖", superscript(N), subscript(nc), name))

struct ∂ϕNode; end

struct BBEnv
    ctx_obj::Any
    bb_start_idx::Int
end

const debug = false

function check_back(nargs, covecs, msg)
    if length(covecs) != nargs
        error("Reverse for stmt `$msg` returned incorrect number of covectors (`$(typeof(covecs))`)")
    end
end

function new_to_regular(@nospecialize(stmt))
    urs = userefs(stmt)
    for op in urs
        val = op[]
        if isa(val, NewSSAValue)
            op[] = SSAValue(val.id)
        end
    end
    return urs[]
end

function expand_switch(code::Vector{Any}, bb_ranges::Vector{UnitRange{Int}}, slot_map)
    renumber = Vector{SSAValue}(undef, length(code))
    new_code = Vector{Any}()

    for val in values(slot_map)
        push!(new_code, Expr(:(=), val, ChainRulesCore.ZeroTangent()))
    end

    # TODO: This is a terrible data structure for this
    phi_rewrites = Dict{Pair{Int, Int}, Int}()

    # First expand switches into sequences of branches.
    # N.B: Code here isn't necessarily in domsort order, so we
    # must do the expansion and statement rewriting in two passes
    for (bb, range) in enumerate(bb_ranges)
        for i in range
            stmt = code[i]
            renumber[i] = SSAValue(length(new_code)+1)
            if isexpr(stmt, :switch)
                cond = stmt.args[1]
                labels = stmt.args[2]
                dests = stmt.args[3]
                for (label, dest) in zip(labels[1:end-1], dests[1:end-1])
                    push!(new_code, Expr(:call, !=, cond, label))
                    comp = NewSSAValue(length(new_code))
                    push!(new_code, GotoIfNot(comp, dest))
                    phi_rewrites[bb=>dest] = length(new_code)
                end
                push!(new_code, GotoNode(dests[end]))
                phi_rewrites[bb=>dests[end]] = length(new_code)
            else
                push!(new_code, stmt)
                if isa(stmt, GotoNode)
                    phi_rewrites[bb=>stmt.label] = length(new_code)
                elseif isa(stmt, GotoIfNot)
                    phi_rewrites[bb=>stmt.dest] = length(new_code)
                    phi_rewrites[bb=>bb+1] = length(new_code)
                else i == last(range)
                    phi_rewrites[bb=>bb+1] = length(new_code)
                end
                if isa(stmt, PhiNode)
                    # just temporarily remember the old bb here. This is a terrible hack, but oh well
                    push!(stmt.edges, bb)
                end
            end
        end
    end

    # Now rewrite branch targets back to statement indexing
    for i = 1:length(new_code)
        stmt = new_code[i]
        stmt = CC.renumber_ssa!(stmt, renumber)
        stmt = new_to_regular(stmt)
        if isa(stmt, GotoNode)
            stmt = GotoNode(renumber[first(bb_ranges[stmt.label])].id)
        elseif isa(stmt, GotoIfNot)
            stmt = GotoIfNot(stmt.cond, renumber[first(bb_ranges[stmt.dest])].id)
        elseif isa(stmt, PhiNode)
            old_phi_bb = pop!(stmt.edges)
            stmt = PhiNode(map(old_pred->Int32(phi_rewrites[old_pred=>old_phi_bb]), stmt.edges),
                stmt.values)
        end
        new_code[i] = stmt
    end

    new_code
end

include("compiler_utils.jl")
include("hacks.jl")

struct PendingCtx
    list::Vector{Int}
end
PendingCtx() = PendingCtx(Int[])

# Split critical edges
# This is absolutely terrible, we really need better tools for this in
# Base
function split_critical_edges!(ir)
    cfg = ir.cfg
    blocks_to_split = Int[]
    edges_to_split = Pair{Int, Int}[]
    for (bb, block) in enumerate(cfg.blocks)
        length(block.preds) <= 1 && continue
        for pred in block.preds
            if length(cfg.blocks[pred].succs) > 1
                if pred+1 == bb
                    # Splitting a fallthrough edge
                    push!(blocks_to_split, bb)
                else
                    push!(edges_to_split, pred=>bb)
                end
            end
        end
    end

    if length(edges_to_split) == 0 && length(blocks_to_split) == 0
        return ir
    end

    for (pred, bb) in edges_to_split
        push!(ir, NewInstruction(GotoNode(bb)))
        push!(ir.cfg, BasicBlock(StmtRange(length(ir.stmts), length(ir.stmts))))
        new_bb = length(ir.cfg.blocks)
        gin = ir.stmts[last(cfg.blocks[pred].stmts)][:inst]
        @assert isa(gin, GotoIfNot)
        ir.stmts[last(cfg.blocks[pred].stmts)][:inst] =
            GotoIfNot(gin.cond, new_bb)
        cfg_delete_edge!(cfg, pred, bb)
        cfg_insert_edge!(cfg, pred, new_bb)
        cfg_insert_edge!(cfg, new_bb, bb)

        for i in cfg.blocks[bb].stmts
            stmt = ir.stmts[i][:inst]
            if isa(stmt, PhiNode)
                map!(stmt.edges, stmt.edges) do edge
                    edge == pred ? new_bb : edge
                end
            elseif stmt !== nothing
                break
            end
        end
    end

    for bb in blocks_to_split
        insert_node!(ir, cfg.blocks[bb].stmts.start, NewInstruction(Expr(:new_bb_marker, bb)))
    end

    ir = compact!(ir)
    cfg = ir.cfg

    bb_rename_offset = Int[1 for _ in 1:length(cfg.blocks)]
    # Now expand basic blocks
    ninserted = 0
    i = 1
    while i < length(ir.stmts)
        if isexpr(ir.stmts[i][:inst], :new_bb_marker)
            bb = ir.stmts[i][:inst].args[1]
            ir.stmts[i][:inst] = nothing
            bbnew = bb + ninserted
            insert!(cfg.blocks, bbnew, BasicBlock(StmtRange(i:i)))
            bb_rename_offset[bb] += 1
            bblock = cfg.blocks[bbnew+1]
            cfg.blocks[bbnew+1] = BasicBlock(StmtRange((i+1):last(bblock.stmts)),
                bblock.preds, bblock.succs)
            i += 1
            while i <= last(bblock.stmts)
                stmt = ir.stmts[i][:inst]
                i += 1
                if isa(stmt, PhiNode)
                    map!(stmt.edges, stmt.edges) do edge
                        edge == bb-1 ? -bbnew : edge
                    end
                elseif stmt !== nothing
                    break
                end
            end
            continue
        end
        i += 1
    end
    bb_rename_mapping = cumsum(bb_rename_offset)
    # Repair CFG
    for i in 1:length(cfg.blocks)
        bb = cfg.blocks[i]
        cfg.blocks[i] = BasicBlock(bb.stmts, map(x->bb_rename_mapping[x], bb.preds),
            map(x->bb_rename_mapping[x], bb.succs))
    end
    for block in blocks_to_split
        cfg_delete_edge!(cfg, bb_rename_mapping[block-1], bb_rename_mapping[block])
        cfg_insert_edge!(cfg, bb_rename_mapping[block-1], bb_rename_mapping[block]-1)
        cfg_insert_edge!(cfg, bb_rename_mapping[block]-1, bb_rename_mapping[block])
    end

    # Repair IR
    for i in 1:length(ir.stmts)
        stmt = ir.stmts[i][:inst]
        isa(stmt, Union{GotoNode, PhiNode, GotoIfNot}) || continue
        if isa(stmt, GotoNode)
            ir.stmts[i][:inst] = GotoNode(bb_rename_mapping[stmt.label])
        elseif isa(stmt, PhiNode)
            map!(stmt.edges, stmt.edges) do edge
                edge < 0 && return -edge
                return bb_rename_mapping[edge]
            end
        else
            @assert isa(stmt, GotoIfNot)
            ir.stmts[i][:inst] = GotoIfNot(stmt.cond, bb_rename_mapping[stmt.dest])
        end
    end

    ir′ = compact!(domsort_ssa!(ir, construct_domtree(ir.cfg.blocks)))

    return ir′
end

import .CC: VarState
function sptypes(sparams)
    VarState[CC.VarState.(sparams, false)...]
end

function optic_transform(ci::CodeInfo, args...)
    newci = copy(ci)
    optic_transform!(newci, args...)
    return newci
end

const SSAFlagType = @static VERSION ≥ v"1.11.0-DEV.377" ? UInt32 : UInt8

function optic_transform!(ci::CodeInfo, mi::MethodInstance, nargs::Int, N::Int)
    code = ci.code
    sparams = mi.sparam_vals

    cfg = compute_basic_blocks(code)
    ci.slotnames = Symbol[Symbol("#self#"), :args, ci.slotnames...]
    ci.slotflags = UInt8[0x00, 0x00, ci.slotflags...]
    ci.slottypes = ci.slottypes === nothing ? Any[Any for _ in 1:length(ci.slotflags)] : Any[Any, Any, ci.slottypes...]

    type = Any[]
    info = CallInfo[NoCallInfo() for i = 1:length(code)]
    flag = SSAFlagType[zero(SSAFlagType) for i = 1:length(code)]
    argtypes = Any[Any for i = 1:2]
    meta = Expr[]
    @static if VERSION ≥ v"1.12.0-DEV.173"
        debuginfo = CC.DebugInfoStream(mi, ci.debuginfo, length(code))
        stmts = CC.InstructionStream(code, type, info, debuginfo.codelocs, flag)
        ir = IRCode(stmts, cfg, debuginfo, argtypes, meta, sptypes(sparams))
    else
        linetable = Core.LineInfoNode[ci.linetable...]
        stmts = CC.InstructionStream(code, type, info, ci.codelocs, flag)
        ir = IRCode(stmts, cfg, linetable, argtypes, meta, sptypes(sparams))
    end

    # SSA conversion
    meth = mi.def::Method
    domtree = construct_domtree(ir.cfg.blocks)
    stmts = @static VERSION < v"1.11.0-DEV.258" ? ir.stmts.inst : ir.stmts.stmt
    defuse_insts = scan_slot_def_use(Int(meth.nargs), ci, stmts)
    ci.ssavaluetypes = Any[Any for i = 1:ci.ssavaluetypes]
    @static if VERSION > v"1.11.0-DEV.337"
        interp = NativeInterpreter() # dummy interpreter (not used by `construct_ssa!`)
        opt_state=OptimizationState(mi, ci, interp)
        ir = construct_ssa!(ci, ir, opt_state, domtree, defuse_insts, SimpleInferenceLattice.instance)
    else
        ir = construct_ssa!(ci, ir, domtree, defuse_insts, ci.slottypes, SimpleInferenceLattice.instance)
    end
    ir = compact!(ir)

    nfixedargs = Int(meth.nargs)
    meth.isva && (nfixedargs -= 1)
    meth.isva || @assert nfixedargs == nargs+1

    ir = diffract_ir!(ir, ci, meth, sparams, nargs, N)

    CC.replace_code_newstyle!(ci, ir)

    ci.ssavaluetypes = length(ci.code)
    ci.ssaflags = SSAFlagType[zero(SSAFlagType) for i=1:length(ci.code)]
    ci.method_for_inference_limit_heuristics = meth
    ci.edges = MethodInstance[mi]

    if isdefined(Base, :__has_internal_change) && Base.__has_internal_change(v"1.12-alpha", :codeinfonargs)
        ci.nargs = 2
        ci.isva = true
    end

    return ci
end
