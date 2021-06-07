using Core.Compiler: MethodInstance, IncrementalCompact, insert_node_here!,
    userefs, SlotNumber, IRCode, compute_basic_blocks, _methods_by_ftype,
    retrieve_code_info, CodeInfo, SSAValue, finish, complete, non_dce_finish!,
    GotoNode, GotoIfNot, block_for_inst, ReturnNode, Argument, compact!,
    OldSSAValue

using Base.Meta

function renumber_cfg!(cfg, code)
    for idx in 1:length(code)
        stmt = code[idx]
        # Convert GotoNode/GotoIfNot/PhiNode to BB addressing
        if isa(stmt, GotoNode)
            code[idx] = GotoNode(block_for_inst(cfg, stmt.label))
        elseif isexpr(stmt, :gotoifnot)
            new_dest = block_for_inst(cfg, stmt.args[2])
            if new_dest == block_for_inst(cfg, idx)+1
                # Drop this node - it's a noop
                code[idx] = stmt.args[1]
            else
                code[idx] = GotoIfNot(stmt.args[1], new_dest)
            end
        elseif isexpr(stmt, :enter)
            code[idx] = Expr(:enter, block_for_inst(cfg, stmt.args[1]))
            ssavalmap[idx] = SSAValue(idx) # Slot to store token for pop_exception
        end
    end
end

cname(nc, N, name) = Symbol(string("∂⃖", superscript(N), subscript(nc), name))

using Core.Compiler: construct_domtree, scan_slot_def_use, construct_ssa!,
    NewInstruction, effect_free, CFG, BasicBlock, bbidxiter, PhiNode,
    Instruction, StmtRange, cfg_insert_edge!, insert_node!,
    non_effect_free

struct ∂ϕNode; end

struct BBEnv
    ctx_obj::Any
    bb_start_idx::Int
end

function expand_switch(code::Vector{Any}, bb_ranges::Vector{UnitRange{Int}}, slot_map)
    renumber = Vector{SSAValue}(undef, length(code))
    new_code = Vector{Any}()

    for val in values(slot_map)
        push!(new_code, Expr(:(=), val, ChainRulesCore.Zero()))
    end

    # First expand switches into sequences of branches
    for i in 1:length(code)
        stmt = code[i]
        renumber[i] = SSAValue(length(new_code)+1)
        if isexpr(stmt, :switch)
            cond = stmt.args[1]
            labels = stmt.args[2]
            dests = stmt.args[3]
            for (label, dest) in zip(labels[1:end-1], dests[1:end-1])
                push!(new_code,
                    Core.Compiler.renumber_ssa!(Expr(:call, !=, cond, label), renumber))
                comp = SSAValue(length(new_code))
                push!(new_code, GotoIfNot(comp, dest))
            end
            push!(new_code, GotoNode(dests[end]))
        else
            push!(new_code, Core.Compiler.renumber_ssa!(stmt, renumber))
        end
    end

    # Now rewrite branch targets back to statement indexing
    for i = 1:length(new_code)
        stmt = new_code[i]
        if isa(stmt, GotoNode)
            stmt = GotoNode(renumber[first(bb_ranges[stmt.label])].id)
        elseif isa(stmt, GotoIfNot)
            stmt = GotoIfNot(stmt.cond, renumber[first(bb_ranges[stmt.dest])].id)
        end
        new_code[i] = stmt
    end

    new_code
end

include("compiler_utils.jl")
include("hacks.jl")

Base.iterate(c::IncrementalCompact, args...) = Core.Compiler.iterate(c, args...)
Base.iterate(p::Core.Compiler.Pair, args...) = Core.Compiler.iterate(p, args...)
Base.iterate(urs::Core.Compiler.UseRefIterator, args...) = Core.Compiler.iterate(urs, args...)
Base.iterate(x::Core.Compiler.BBIdxIter, args...) = Core.Compiler.iterate(x, args...)
Base.getindex(urs::Core.Compiler.UseRefIterator, args...) = Core.Compiler.getindex(urs, args...)
Base.getindex(urs::Core.Compiler.UseRef, args...) = Core.Compiler.getindex(urs, args...)
Base.setindex!(c::Core.Compiler.IncrementalCompact, args...) = Core.Compiler.setindex!(c, args...)
Base.setindex!(urs::Core.Compiler.UseRef, args...) = Core.Compiler.setindex!(urs, args...)
function transform!(ci, meth, nargs, sparams, N)
    n_closures = 2^N - 1

    code = ci.code
    cfg = compute_basic_blocks(code)
    renumber_cfg!(cfg, code)
    slotnames = Symbol[Symbol("#self#"), :args, ci.slotnames...]
    slotflags = UInt8[(0x00 for i = 1:2)..., ci.slotflags...]
    slottypes = UInt8[(0x00 for i = 1:2)..., ci.slotflags...]

    ir = IRCode(Core.Compiler.InstructionStream(code, Any[],
        Any[nothing for i = 1:length(code)],
        ci.codelocs, UInt8[0 for i = 1:length(code)]), cfg, Core.LineInfoNode[ci.linetable...],
        Any[Any for i = 1:2], Any[], Any[sparams...])

    # SSA conversion
    domtree = construct_domtree(ir.cfg.blocks)
    defuse_insts = scan_slot_def_use(meth.nargs-1, ci, ir.stmts.inst)
    ci.ssavaluetypes = Any[Any for i = 1:ci.ssavaluetypes]
    ir = construct_ssa!(ci, ir, domtree, defuse_insts, nargs, Any[Any for i = 1:length(slotnames)])
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

        # TODO: Split critical edges

        for (bb, i) in bbidxiter(ir)
            bb == new_bb_idx && break
            stmt = ir.stmts[i].inst
            if isa(stmt, ReturnNode)
                push!(ϕ.edges, bb)
                push!(ϕ.values, stmt.val)
                ir[i] = NewInstruction(GotoNode(new_bb_idx))
                cfg_insert_edge!(cfg, bb, new_bb_idx)
            end
        end

        # Now add a special control flow marker to every basic block with more
        # than one predecessor.
        for block in cfg.blocks
            if length(block.preds) > 1
                insert_node!(ir, block.stmts.start,
                    non_effect_free(NewInstruction(Expr(:phi_placeholder, copy(block.preds)))))
            end
        end

        ir = compact!(ir)
        cfg = ir.cfg
    end

    orig_bb_ranges = [first(bb.stmts):last(bb.stmts) for bb in cfg.blocks]

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

    slot_map = Dict{Union{SSAValue, Argument}, SlotNumber}()

    # TODO: Can we use the same method for each 2nd order of the transform
    # (except the last and the first one)
    for nc = 1:2:n_closures
        arg_accums = Union{Nothing, Vector{Any}}[nothing for i = 1:(meth.nargs)]
        accums = Union{Nothing, Vector{Any}}[nothing for i = 1:length(ir.stmts)]

        opaque_ci = opaque_cis[nc]
        code = opaque_ci.code
        has_cfg = length(cfg.blocks) != 1

        function insert_node_rev!(node)
            push!(code, node)
            SSAValue(length(code))
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
                sn = get!(slot_map, val) do
                    sn′ = SlotNumber(length(slot_map) + 3)
                    push!(opaque_ci.slotnames, Symbol(string("for_", val)))
                    push!(opaque_ci.slotflags, UInt8(0))
                    sn′
                end
                accumed = insert_node_rev!(Expr(:call, accum, sn, accumulant))
                insert_node_rev!(Expr(:(=), sn, accumed))
            end
        end

        function do_accum(for_val)
            if !has_cfg
                this_accums = isa(for_val, SSAValue) ? accums[for_val.id] :
                    arg_accums[for_val.n]
                if this_accums === nothing || isempty(this_accums)
                    return ChainRulesCore.Zero()
                elseif length(this_accums) == 1
                    return this_accums[]
                else
                    return insert_node_rev!(Expr(:call, accum, this_accums...))
                end
            else
                return get(slot_map, for_val, ChainRulesCore.Zero())
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
                return insert_node_rev!(Expr(:call, getfield, current_env.ctx_obj,
                    i - current_env.bb_start_idx + 1))
            end
            error()
        end

        bb_ranges = Vector{UnitRange{Int}}(undef, length(cfg.blocks))
        phi_reserve = Vector{Vector{Any}}(undef, length(cfg.blocks))

        for (bb, i) in Iterators.reverse(bbidxiter(ir))
            first_bb_idx = cfg.blocks[bb].stmts.start
            stmt = ir.stmts[i][:inst]
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
                    canon = insert_node_rev!(Expr(:call, ChainRulesCore.canonicalize, Δ))
                    nt = insert_node_rev!(Expr(:call, ChainRulesCore.backing, canon))
                    for (j, arg) in enumerate(stmt.args)
                        j == 1 && continue
                        if is_accumable(arg)
                            accum!(arg, insert_node_rev!(Expr(:call, lifted_getfield, nt, j-1)))
                        end
                    end
                end
            elseif isexpr(stmt, :splatnew)
                Δ = do_accum(SSAValue(i))
                newT = retrieve_ctx_obj(current_env, i)
                if nc != n_closures
                    revs[nc+1][i] = newT
                end
                canon = insert_node_rev!(Expr(:call, ChainRulesCore.canonicalize, Δ))
                nt = insert_node_rev!(Expr(:call, ChainRulesCore.backing, canon))
                arg = stmt.args[2]
                if is_accumable(stmt.args[2])
                    accum!(stmt.args[2], nt)
                end
            elseif isa(stmt, GlobalRef) || isexpr(stmt, :static_parameter) || isexpr(stmt, :throw_undef_if_not)
                # We drop gradients for globals and static parameters
            elseif isa(stmt, PhiNode)
                Δ = do_accum(SSAValue(i))
                @assert length(ir.cfg.blocks[bb].preds) >= 1
                for (edge, val) in zip(stmt.edges, stmt.values)
                    if !isassigned(phi_reserve, edge)
                        phi_reserve[edge] = Vector{Any}()
                    end
                    push!(phi_reserve[edge], val=>Δ)
                end
                # PhiNodes are ignored
            elseif isa(stmt, GotoNode)
                current_env = BBEnv(ctx_map[stmt.label],
                    first(ir.cfg.blocks[bb].stmts))
                if isassigned(phi_reserve, bb)
                    for (val, Δ) in phi_reserve[bb]
                        accum!(val, Δ)
                    end
                end
            elseif isa(stmt, GotoIfNot)
                if bb == 1
                    current_env = nothing
                else
                    error()
                end
            elseif isexpr(stmt, :phi_placeholder)
                @assert i == first_bb_idx
                tup = retrieve_ctx_obj(current_env, i)
                branch = insert_node_rev!(Expr(:call, getfield, tup, 1))
                ctx = insert_node_rev!(Expr(:call, getfield, tup, 2))
                insert_node_rev!(Expr(:switch, branch, collect(1:length(stmt.args[1])),
                    map(to_back_bb, stmt.args[1])))
                ctx_map[bb] = ctx
            else
                error((N, meth, stmt))
            end
            if i == first_bb_idx && bb != 1
                if !isexpr(stmt, :phi_placeholder)
                    preds = ir.cfg.blocks[bb].preds
                    @assert length(preds) == 1
                    insert_node_rev!(GotoNode(to_back_bb(preds[1])))
                end
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
                    op[] = Zero()
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
                compT = insert_node_here!(Expr(:call, Core.apply_type, Composite, newT, ntT))
                fwds[i] = insert_node_here!(Expr(:new, compT, thent))
            elseif isexpr(stmt, :splatnew)
                error()
            elseif isa(stmt, GlobalRef)
                fwds[i] = Zero()
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
            preds = stmt.args[1]
            values = Any[]
            for (selector, pred) in enumerate(preds)
                tup = my_insert_node!(compact, OldSSAValue(last(orig_bb_ranges[pred])),
                    effect_free(NewInstruction(Expr(:call, tuple, rev[orig_bb_ranges[pred]]...))))
                ctx = my_insert_node!(compact, OldSSAValue(last(orig_bb_ranges[pred])),
                    effect_free(NewInstruction(Expr(:call, tuple, selector, tup))))
                push!(values, ctx)
            end
            compact[idx] = PhiNode(map(Int32, preds), values)
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
    end

    non_dce_finish!(compact)
    ir = complete(compact)
    ir = compact!(ir)

    Core.Compiler.replace_code_newstyle!(ci, ir, nargs+1)
    ci.ssavaluetypes = length(ci.code)
    ci.slotnames = slotnames
    ci.slotflags = slotflags
    ci.slottypes = slottypes

    ci
end
