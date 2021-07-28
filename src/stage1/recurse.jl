using Core.Compiler: MethodInstance, IncrementalCompact, insert_node_here!,
    userefs, SlotNumber, IRCode, compute_basic_blocks, _methods_by_ftype,
    retrieve_code_info, CodeInfo, SSAValue, finish, complete, non_dce_finish!,
    GotoNode, GotoIfNot, block_for_inst, ReturnNode, Argument, compact!,
    OldSSAValue

using Base.Meta

cname(nc, N, name) = Symbol(string("∂⃖", superscript(N), subscript(nc), name))

using Core.Compiler: construct_domtree, scan_slot_def_use, construct_ssa!,
    NewInstruction, effect_free, CFG, BasicBlock, bbidxiter, PhiNode,
    Instruction, StmtRange, cfg_insert_edge!, insert_node!,
    non_effect_free, cfg_delete_edge!, domsort_ssa!

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
        stmt = Core.Compiler.renumber_ssa!(stmt, renumber)
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
        insert_node!(ir, cfg.blocks[bb].stmts.start,
            non_effect_free(NewInstruction(Expr(:new_bb_marker, bb))))
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
            insert!(cfg.blocks, bbnew, BasicBlock(i:i))
            bb_rename_offset[bb] += 1
            bblock = cfg.blocks[bbnew+1]
            cfg.blocks[bbnew+1] = BasicBlock((i+1):last(bblock.stmts),
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

Base.iterate(c::IncrementalCompact, args...) = Core.Compiler.iterate(c, args...)
Base.iterate(p::Core.Compiler.Pair, args...) = Core.Compiler.iterate(p, args...)
Base.iterate(urs::Core.Compiler.UseRefIterator, args...) = Core.Compiler.iterate(urs, args...)
Base.iterate(x::Core.Compiler.BBIdxIter, args...) = Core.Compiler.iterate(x, args...)
Base.getindex(urs::Core.Compiler.UseRefIterator, args...) = Core.Compiler.getindex(urs, args...)
Base.getindex(urs::Core.Compiler.UseRef, args...) = Core.Compiler.getindex(urs, args...)
Base.getindex(c::Core.Compiler.IncrementalCompact, args...) = Core.Compiler.getindex(c, args...)
Base.setindex!(c::Core.Compiler.IncrementalCompact, args...) = Core.Compiler.setindex!(c, args...)
Base.setindex!(urs::Core.Compiler.UseRef, args...) = Core.Compiler.setindex!(urs, args...)
function transform!(ci, meth, nargs, sparams, N)
    n_closures = 2^N - 1

    code = ci.code
    cfg = compute_basic_blocks(code)
    slotnames = Symbol[Symbol("#self#"), :args, ci.slotnames...]
    slotflags = UInt8[(0x00 for i = 1:2)..., ci.slotflags...]
    slottypes = UInt8[(0x00 for i = 1:2)..., ci.slotflags...]

    ir = IRCode(Core.Compiler.InstructionStream(code, Any[],
        Any[nothing for i = 1:length(code)],
        ci.codelocs, UInt8[0 for i = 1:length(code)]), cfg, Core.LineInfoNode[ci.linetable...],
        Any[Any for i = 1:2], Any[], Any[sparams...])

    # SSA conversion
    domtree = construct_domtree(ir.cfg.blocks)
    defuse_insts = scan_slot_def_use(VERSION >= v"1.8.0-DEV.267" ? Int(meth.nargs) : meth.nargs-1, ci, ir.stmts.inst)
    ci.ssavaluetypes = Any[Any for i = 1:ci.ssavaluetypes]
    if VERSION >= v"1.8.0-DEV.267"
        ir = construct_ssa!(ci, ir, domtree, defuse_insts, Any[Any for i = 1:length(slotnames)])
    else
        ir = construct_ssa!(ci, ir, domtree, defuse_insts, nargs, Any[Any for i = 1:length(slotnames)])
    end
    ir = compact!(ir)
    cfg = ir.cfg

    # If we have more than one basic block, canonicalize by creating a single
    # return node in the last basic block.

    if length(cfg.blocks) != 1
        ϕ = PhiNode()
        bb_start = length(ir.stmts)+1
        push!(ir, NewInstruction(ϕ))
        push!(ir, NewInstruction(ReturnNode(SSAValue(length(ir.stmts)))))
        push!(ir.cfg, BasicBlock(StmtRange(bb_start, length(ir.stmts))))
        new_bb_idx = length(cfg.blocks)

        for (bb, i) in bbidxiter(ir)
            bb == new_bb_idx && break
            stmt = ir.stmts[i].inst
            if isa(stmt, ReturnNode)
                push!(ϕ.edges, bb)
                if !isa(stmt.val, SSAValue)
                    push!(ϕ.values, insert_node!(ir, i,
                        non_effect_free(NewInstruction(stmt.val))))
                else
                    push!(ϕ.values, stmt.val)
                end
                ir[i] = NewInstruction(GotoNode(new_bb_idx))
                cfg_insert_edge!(cfg, bb, new_bb_idx)
            end
        end

        ir = compact!(ir)
        ir = split_critical_edges!(ir)
        cfg = ir.cfg

        # Now add a special control flow marker to every basic block
        # TODO: This lowering of control flow is extremely simplistic.
        # It needs to be improved in the very near future, but let's get
        # something working for now.
        for block in cfg.blocks
            if length(block.preds) != 0
                insert_node!(ir, block.stmts.start,
                    non_effect_free(NewInstruction(Expr(:phi_placeholder, copy(block.preds)))))
            end
        end

        ir = compact!(ir)
        cfg = ir.cfg
    end

    orig_bb_ranges = [first(bb.stmts):last(bb.stmts) for bb in cfg.blocks]

    # Special case: If there's only 1 `nothing` statement in the first BB,
    # we run the risk that compact skips it below. In that case, replace that
    # statement by a GotoNode to prevent that.
    if length(orig_bb_ranges[1]) == 1 && ir.stmts[1][:inst] === nothing
        ir.stmts[1][:inst] = GotoNode(2)
    end

    revs = Any[Any[nothing for i = 1:length(ir.stmts)] for i = 1:n_closures]
    opaque_cis = map(1:n_closures) do nc
        code = Any[] # Will be filled in later
        opaque_ci = ccall(:jl_new_code_info_uninit, Ref{CodeInfo}, ())
        opaque_ci.code = code
        if nc % 2 == 1
            opaque_ci.slotnames = Symbol[Symbol("#self#"), :Δ]
            opaque_ci.slotflags = UInt8[0, 0]
        else
            opaque_ci.slotnames = [Symbol("#oc#"), ci.slotnames...]
            opaque_ci.slotflags = UInt8[0, ci.slotflags...]
        end
        opaque_ci.linetable = Core.LineInfoNode[ci.linetable[1]]
        opaque_ci.inferred = false
        opaque_ci
    end

    nfixedargs = meth.isva ? meth.nargs - 1 : meth.nargs
    meth.isva || @assert nfixedargs == nargs+1

    extra_slotnames = Symbol[]
    extra_slotflags = UInt8[]

    # First go through and assign an accumulation slot to every used SSAValue/Argument
    has_cfg = length(cfg.blocks) != 1
    if has_cfg
        slot_map = Dict{Union{SSAValue, Argument}, SlotNumber}()
        phi_tmp_slot_map = Dict{Int, SlotNumber}()
        phi_uses = Dict{Int, Vector{Pair{SlotNumber, SlotNumber}}}()
        for (bb, i) in Iterators.reverse(bbidxiter(ir))
            first_bb_idx = cfg.blocks[bb].stmts.start
            stmt = ir.stmts[i][:inst]
            for urs in userefs(stmt)
                val = urs[]
                (isa(val, SSAValue) || isa(val, Argument)) || continue
                if !haskey(slot_map, val)
                    sn′ = SlotNumber(length(extra_slotnames) + 3)
                    push!(extra_slotnames, Symbol(string("for_", val)))
                    push!(extra_slotflags, UInt8(0))
                    slot_map[val] = sn′
                end
            end
            if isa(stmt, PhiNode)
                sn′ = SlotNumber(length(extra_slotnames) + 3)
                push!(extra_slotnames, Symbol(string("phi_temp_", i)))
                push!(extra_slotflags, UInt8(0))
                phi_tmp_slot_map[i] = sn′

                for use_n in 1:length(stmt.edges)
                    edge = stmt.edges[use_n]
                    # This assume the absence of critical edges
                    @assert length(cfg.blocks[edge].succs) == 1
                    val = stmt.values[use_n]
                    (isa(val, SSAValue) || isa(val, Argument)) || continue
                    push!(get!(phi_uses, edge, Vector{Pair{SlotNumber, SlotNumber}}()),
                        sn′=>slot_map[val])
                end
            end
        end
    end

    # TODO: Can we use the same method for each 2nd order of the transform
    # (except the last and the first one)
    for nc = 1:2:n_closures
        arg_accums = Union{Nothing, Vector{Any}}[nothing for i = 1:(meth.nargs)]
        accums = Union{Nothing, Vector{Any}}[nothing for i = 1:length(ir.stmts)]

        opaque_ci = opaque_cis[nc]
        code = opaque_ci.code

        function insert_node_rev!(node)
            push!(code, node)
            SSAValue(length(code))
        end

        if has_cfg
            append!(opaque_ci.slotnames, extra_slotnames)
            append!(opaque_ci.slotflags, extra_slotflags)
        end

        is_accumable(val) = isa(val, SSAValue) || isa(val, Argument)
        function accum!(val, accumulant)
            is_accumable(val) || return
            if !has_cfg
                if isa(val, SSAValue)
                    id = val.id
                    if isa(accums[id], Nothing)
                        accums[id] = Any[]
                    end
                    push!(accums[id], accumulant)
                elseif isa(val, Argument)
                    id = val.n
                    if isa(arg_accums[id], Nothing)
                        arg_accums[id] = Any[]
                    end
                    push!(arg_accums[id], accumulant)
                end
            else
                sn = slot_map[val]
                accumed = insert_node_rev!(Expr(:call, accum, sn, accumulant))
                insert_node_rev!(Expr(:(=), sn, accumed))
            end
        end

        function do_accum(for_val)
            if !has_cfg
                this_accums = isa(for_val, SSAValue) ? accums[for_val.id] :
                    arg_accums[for_val.n]
                if this_accums === nothing || isempty(this_accums)
                    return ChainRulesCore.ZeroTangent()
                elseif length(this_accums) == 1
                    return this_accums[]
                else
                    return insert_node_rev!(Expr(:call, accum, this_accums...))
                end
            else
                return get(slot_map, for_val, ChainRulesCore.ZeroTangent())
            end
        end

        to_back_bb(i) = length(cfg.blocks) - i + 1

        current_env = nothing
        ctx_map = Dict{Int, Any}()
        function retrieve_ctx_obj(current_env, i)
            if current_env === nothing
                return insert_node_rev!(Expr(:call, getfield, Argument(1),
                    i - first(orig_bb_ranges[end]) + 1))
            elseif isa(current_env, BBEnv)
                bbidx = i - current_env.bb_start_idx + 1
                @assert bbidx > 0
                return insert_node_rev!(Expr(:call, getfield, current_env.ctx_obj,
                    bbidx))
            end
            error()
        end

        function access_ctx_map(dest)
            if !haskey(ctx_map, dest)
                ctx_map[dest] = PendingCtx()
            end
            val = ctx_map[dest]
            if isa(val, PendingCtx)
                rval = insert_node_rev!(error)
                push!(val.list, rval.id)
                return rval
            end
            return val
        end

        bb_ranges = Vector{UnitRange{Int}}(undef, length(cfg.blocks))

        for (bb, i) in Iterators.reverse(bbidxiter(ir))
            first_bb_idx = cfg.blocks[bb].stmts.start
            last_bb_idx = cfg.blocks[bb].stmts.stop
            stmt = ir.stmts[i][:inst]

            if has_cfg && i == last_bb_idx
                if haskey(phi_uses, bb)
                    for (accumulant, accumulator) in phi_uses[bb]
                        accumed = insert_node_rev!(Expr(:call, accum, accumulator, accumulant))
                        insert_node_rev!(Expr(:(=), accumulator, accumed))
                    end
                end
                if !isa(stmt, Union{GotoNode, GotoIfNot, ReturnNode})
                    current_env = BBEnv(access_ctx_map(bb+1),
                        first(ir.cfg.blocks[bb].stmts))
                end
            end

            if isa(stmt, Core.ReturnNode)
                accum!(stmt.val, Argument(2))
            elseif isexpr(stmt, :call)
                Δ = do_accum(SSAValue(i))
                callee = retrieve_ctx_obj(current_env, i)
                vecs = call = insert_node_rev!(Expr(:call, callee, Δ))
                if nc != n_closures
                    vecs = insert_node_rev!(Expr(:call, getfield, call, 1))
                    revs[nc+1][i] = insert_node_rev!(Expr(:call, getfield, call, 2))
                end
                if debug
                    insert_node_rev!(Expr(:call, check_back, length(stmt.args), vecs, string(stmt)))
                end
                for (j, arg) in enumerate(stmt.args)
                    if is_accumable(arg)
                        accum!(arg, insert_node_rev!(Expr(:call, getfield, vecs, j)))
                    end
                end
            elseif isexpr(stmt, :new)
                Δ = do_accum(SSAValue(i))
                newT = retrieve_ctx_obj(current_env, i)
                if nc != n_closures
                    revs[nc+1][i] = newT
                end
                # No gradient accumulation for zero-argument structs
                if length(stmt.args) != 1
                    # TODO: Use newT here?
                    #canon = insert_node_rev!(Expr(:call, ChainRulesCore.canonicalize, Δ))
                    #nt = insert_node_rev!(Expr(:call, ChainRulesCore.backing, canon))
                    nt = Δ
                    for (j, arg) in enumerate(stmt.args)
                        j == 1 && continue
                        if is_accumable(arg)
                            accum!(arg, insert_node_rev!(Expr(:call, lifted_getfield, nt, j-1)))
                        end
                    end
                end
            elseif isexpr(stmt, :splatnew)
                @assert length(stmt.args) == 2
                Δ = do_accum(SSAValue(i))
                newT = retrieve_ctx_obj(current_env, i)
                if nc != n_closures
                    revs[nc+1][i] = newT
                end
                #canon = insert_node_rev!(Expr(:call, ChainRulesCore.canonicalize, Δ))
                #nt = insert_node_rev!(Expr(:call, ChainRulesCore.backing, canon))
                nt = Δ
                arg = stmt.args[2]
                if is_accumable(stmt.args[2])
                    accum!(stmt.args[2], nt)
                end
            elseif isa(stmt, GlobalRef) || isexpr(stmt, :static_parameter) || isexpr(stmt, :throw_undef_if_not)
                # We drop gradients for globals and static parameters
            elseif isexpr(stmt, :inbounds)
                # Nothing to do
            elseif isa(stmt, PhiNode)
                Δ = do_accum(SSAValue(i))
                @assert length(ir.cfg.blocks[bb].preds) >= 1
                insert_node_rev!(Expr(:(=), phi_tmp_slot_map[i], Δ))
            elseif isa(stmt, GotoNode)
                current_env = BBEnv(access_ctx_map(stmt.label),
                    first(ir.cfg.blocks[bb].stmts))
            elseif isa(stmt, GotoIfNot)
                current_env = BBEnv(insert_node_rev!(PhiNode(Int32.(map(to_back_bb, Int32[bb+1, stmt.dest])),
                    Any[access_ctx_map(bb+1),
                        access_ctx_map(stmt.dest)])), first(ir.cfg.blocks[bb].stmts))
            elseif isexpr(stmt, :phi_placeholder)
                @assert i == first_bb_idx
                tup = retrieve_ctx_obj(current_env, i)
                if length(stmt.args[1]) > 1
                    branch = insert_node_rev!(Expr(:call, getfield, tup, 1))
                    ctx = insert_node_rev!(Expr(:call, getfield, tup, 2))
                    insert_node_rev!(Expr(:switch, branch, collect(1:length(stmt.args[1])),
                        map(to_back_bb, stmt.args[1])))
                else
                    ctx = tup
                    insert_node_rev!(GotoNode(to_back_bb(stmt.args[1][1])))
                end
                if haskey(ctx_map, bb)
                    for item in (ctx_map[bb]::PendingCtx).list
                        code[item] = ctx
                    end
                end
                ctx_map[bb] = ctx
            elseif isa(stmt, Nothing)
                # Nothing to do
            else
                error((N, meth, stmt))
            end
            if has_cfg && haskey(slot_map, SSAValue(i))
                # Reset accumulator slots
                insert_node_rev!(Expr(:(=), slot_map[SSAValue(i)], ChainRulesCore.ZeroTangent()))
            end
            if i == first_bb_idx && bb != 1
                @assert isexpr(stmt, :phi_placeholder)
            end
            if i == first_bb_idx
                back_bb = to_back_bb(bb)
                bb_ranges[back_bb] = (back_bb == 1 ? 1 : (last(bb_ranges[back_bb - 1])+1)):length(code)
            end
        end

        ret_tuple = arg_tuple = insert_node_rev!(Expr(:call, tuple, [do_accum(Argument(i)) for i = 1:(nfixedargs)]...))
        if meth.isva
            ret_tuple = arg_tuple = insert_node_rev!(Expr(:call, Core._apply_iterate, iterate, Core.tuple, arg_tuple,
                do_accum(Argument(nfixedargs+1))))
        end
        if nc != n_closures
            lno = LineNumberNode(1, :none)
            next_oc = insert_node_rev!(Expr(:new_opaque_closure, Tuple{(Any for i = 1:nargs+1)...}, meth.isva, Union{}, Any,
                Expr(:opaque_closure_method, cname(nc+1, N, meth.name), Int(meth.nargs), lno, opaque_cis[nc+1]), revs[nc+1]...))
            ret_tuple = insert_node_rev!(Expr(:call, tuple, arg_tuple, next_oc))
        end
        insert_node_rev!(Core.ReturnNode(ret_tuple))
        bb_ranges[end] = first(bb_ranges[end]):length(code)

        if has_cfg
            code = opaque_ci.code = expand_switch(code, bb_ranges, slot_map)
        end

        opaque_ci.codelocs = Int32[0 for i=1:length(code)]
        opaque_ci.ssavaluetypes = length(code)
        opaque_ci.ssaflags = UInt8[0 for i=1:length(code)]
    end


    for nc = 2:2:n_closures
        fwds = Any[nothing for i = 1:length(ir.stmts)]

        opaque_ci = opaque_cis[nc]
        code = opaque_ci.code
        function insert_node_here!(node)
            push!(code, node)
            SSAValue(length(code))
        end

        for i in 1:length(ir.stmts)
            stmt = ir.stmts[i][:inst]
            if isa(stmt, Expr)
                stmt = copy(stmt)
            end
            urs = userefs(stmt)
            for op in urs
                val = op[]
                if isa(val, Argument)
                    op[] = Argument(val.n + 1)
                elseif isa(val, SSAValue)
                    op[] = fwds[val.id]
                else
                    op[] = ZeroTangent()
                end
            end
            stmt = urs[]

            if isexpr(stmt, :call)
                callee = insert_node_here!(Expr(:call, getfield, Argument(1), i))
                pushfirst!(stmt.args, callee)
                call = insert_node_here!(stmt)
                prim = insert_node_here!(Expr(:call, getfield, call, 1))
                dual = insert_node_here!(Expr(:call, getfield, call, 2))
                fwds[i] = prim
                revs[nc+1][i] = dual
            elseif isa(stmt, ReturnNode)
                lno = LineNumberNode(1, :none)
                next_oc = insert_node_here!(Expr(:new_opaque_closure, Tuple{Any}, false, Union{}, Any,
                    Expr(:opaque_closure_method, cname(nc+1, N, meth.name), 1, lno, opaque_cis[nc + 1]), revs[nc+1]...))
                ret_tup = insert_node_here!(Expr(:call, tuple, stmt.val, next_oc))
                insert_node_here!(ReturnNode(ret_tup))
            elseif isexpr(stmt, :new)
                newT = insert_node_here!(Expr(:call, getfield, Argument(1), i))
                revs[nc+1][i] = newT
                fnames = insert_node_here!(Expr(:call, fieldnames, newT))
                nt = insert_node_here!(Expr(:call, Core.apply_type, NamedTuple, fnames))
                stmt.head = :call
                stmt.args[1] = Core.tuple
                thetup = insert_node_here!(stmt)
                thent = insert_node_here!(Expr(:call, nt, thetup))
                ntT = insert_node_here!(Expr(:call, typeof, thent))
                compT = insert_node_here!(Expr(:call, Core.apply_type, Tangent, newT, ntT))
                fwds[i] = insert_node_here!(Expr(:new, compT, thent))
            elseif isexpr(stmt, :splatnew)
                error()
            elseif isa(stmt, GlobalRef)
                fwds[i] = ZeroTangent()
            elseif !isa(stmt, Expr)
                @show stmt
                error()
            else
                fwds[i] = insert_node_here!(stmt)
            end
        end

        opaque_ci.codelocs = Int32[0 for i=1:length(code)]
        opaque_ci.ssavaluetypes = length(code)
        opaque_ci.ssaflags = UInt8[0 for i=1:length(code)]
    end

    # TODO: This is absolutely aweful, but the best we can do given the data structures we have
    has_terminator = [isa(ir.stmts[last(range)].inst, Union{GotoNode, GotoIfNot}) for range in orig_bb_ranges]

    compact = IncrementalCompact(ir)

    arg_mapping = Any[]
    for argno in 1:nfixedargs
        push!(arg_mapping, insert_node_here!(compact,
            NewInstruction(Expr(:call, getfield, Argument(2), argno), Any, Int32(0))))
    end

    if meth.isva
        # Extract the rest of the arguments and make a tuple out of them
        ssas = map((nfixedargs+1):(nargs+1)) do i
            insert_node_here!(compact,
                NewInstruction(Expr(:call, getfield, Argument(2), i), Any, Int32(0)))
        end
        push!(arg_mapping,
            insert_node_here!(compact,
                NewInstruction(Expr(:call, tuple, ssas...), Any, Int32(0))))
    end

    rev = revs[1]
    active_bb = 1

    phi_nodes = Any[PhiNode() for _ = 1:length(orig_bb_ranges)]

    for ((old_idx, idx), stmt) in compact
        # remap arguments
        urs = userefs(stmt)
        compact[idx] = nothing
        for op in urs
            val = op[]
            if isa(val, Argument)
                op[] = arg_mapping[val.n]
            end
            if isexpr(val, :static_parameter)
                op[] = sparams[val.args[1]]
            end
        end
        compact[idx] = stmt = urs[]
        # f(args...) -> ∂⃖{N}(args...)
        orig_stmt = stmt
        if isexpr(stmt, :(=))
            stmt = stmt.args[2]
        end
        if isexpr(stmt, :call)
            compact[idx] = Expr(:call, ∂⃖{N}(), stmt.args...)
            if isexpr(orig_stmt, :(=))
                orig_stmt.args[2] = stmt
                stmt = orig_stmt
            end
            compact.ssa_rename[compact.idx-1] = insert_node_here!(compact,
                NewInstruction(Expr(:call, getfield, SSAValue(idx), 1), Any, compact.result[idx][:line]),
                true)
            rev[old_idx] = insert_node_here!(compact,
                NewInstruction(Expr(:call, getfield, SSAValue(idx), 2), Any, compact.result[idx][:line]),
                true)
        elseif isexpr(stmt, :static_parameter)
            stmt = sparams[stmt.args[1]]
            if isexpr(orig_stmt, :(=))
                orig_stmt.args[2] = stmt
                stmt = orig_stmt
            end
            compact[idx] = stmt
        elseif isexpr(stmt, :new) || isexpr(stmt, :splatnew)
            rev[old_idx] = stmt.args[1]
        elseif isexpr(stmt, :phi_placeholder)
            compact[idx] = phi_nodes[active_bb]
            # TODO: This is a base julia bug
            push!(compact.late_fixup, idx)
            rev[old_idx] = SSAValue(idx)
        elseif isa(stmt, Core.ReturnNode)
            lno = LineNumberNode(1, :none)
            compact[idx] = Expr(:new_opaque_closure, Tuple{Any}, false, Union{}, Any,
                Expr(:opaque_closure_method, cname(1, N, meth.name), 1, lno, opaque_cis[1]), rev[orig_bb_ranges[end]]...)
            argty = insert_node_here!(compact,
                NewInstruction(Expr(:call, typeof, stmt.val), Any, compact.result[idx][:line]), true)
            applyty = insert_node_here!(compact,
                NewInstruction(Expr(:call, Core.apply_type, OpticBundle, argty), Any, compact.result[idx][:line]),
                true)
            retval = insert_node_here!(compact,
                NewInstruction(Expr(:new, applyty, stmt.val, SSAValue(idx)), Any, compact.result[idx][:line]),
                true)
            compact.ssa_rename[compact.idx-1] = insert_node_here!(compact,
                NewInstruction(Core.ReturnNode(retval), Any, compact.result[idx][:line]),
                true)
        end

        succs = cfg.blocks[active_bb].succs
        if old_idx == last(orig_bb_ranges[active_bb]) && length(succs) != 0
            override = false
            if has_terminator[active_bb]
                terminator = compact[idx]
                compact[idx] = nothing
                override = true
            end
            function terminator_insert_node!(node)
                if override
                    compact[idx] = node.stmt
                    override = false
                    return SSAValue(idx)
                else
                    return insert_node_here!(compact, node, true)
                end
            end
            tup = terminator_insert_node!(
                effect_free(NewInstruction(Expr(:call, tuple, rev[orig_bb_ranges[active_bb]]...), Any, Int32(0))))
            for succ in succs
                preds = cfg.blocks[succ].preds
                if length(preds) == 1
                    val = tup
                else
                    selector = findfirst(==(active_bb), preds)
                    val = insert_node_here!(compact, effect_free(NewInstruction(Expr(:call, tuple, selector, tup), Any, Int32(0))), true)
                end
                pn = phi_nodes[succ]
                push!(pn.edges, active_bb)
                push!(pn.values, val)
            end
            if has_terminator[active_bb]
                insert_node_here!(compact, NewInstruction(terminator, Any, Int32(0)), true)
            end
            active_bb += 1
        end
    end

    non_dce_finish!(compact)
    ir = complete(compact)
    ir = compact!(ir)
    Core.Compiler.verify_ir(ir)

    Core.Compiler.replace_code_newstyle!(ci, ir, nargs+1)
    ci.ssavaluetypes = length(ci.code)
    ci.slotnames = slotnames
    ci.slotflags = slotflags
    ci.slottypes = slottypes


    ci
end
