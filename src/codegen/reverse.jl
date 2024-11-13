# Codegen shared by both stage1 and stage2

function make_opaque_closure(interp, typ, name, meth_nargs::Int, isva, lno, ci, revs...)
    if interp !== nothing
        @static if VERSION ≥ v"1.12.0-DEV.15"
            rettype = Any # ci.rettype # TODO revisit
        else
            ci.inferred = true
            rettype = ci.rettype
        end
        @static if VERSION ≥ v"1.12.0-DEV.15"
            ocm = Core.OpaqueClosure(ci; rettype, nargs=meth_nargs, isva, sig=typ).source
        else
            ocm = ccall(:jl_new_opaque_closure_from_code_info, Any, (Any, Any, Any, Any, Any, Cint, Any, Cint, Cint, Any),
                typ, Union{}, rettype, @__MODULE__, ci, lno.line, lno.file, meth_nargs, isva, ()).source
        end
        return Expr(:new_opaque_closure, typ, Union{}, Any, ocm, revs...)
    else
        oc_nargs = Int64(meth_nargs)
        Expr(:new_opaque_closure, typ, Union{}, Any,
            Expr(:opaque_closure_method, name, oc_nargs, isva, lno, ci), revs...)
    end
end

function diffract_ir!(ir, ci, meth, sparams::Core.SimpleVector, nargs::Int, N::Int, interp=nothing, curs=nothing)
    n_closures = 2^N - 1
    cfg = ir.cfg

    # If we have more than one basic block, canonicalize by creating a single
    # return node in the last basic block.

    if length(cfg.blocks) != 1
        ϕ = PhiNode()

        bb_start = length(ir.stmts)+1
        push!(ir, NewInstruction(ϕ))
        push!(ir, NewInstruction(ReturnNode(SSAValue(length(ir.stmts)))))
        push!(ir.cfg, BasicBlock(StmtRange(bb_start, length(ir.stmts))))
        new_bb_idx = length(ir.cfg.blocks)

        for (bb, i) in bbidxiter(ir)
            bb == new_bb_idx && break
            stmt = ir.stmts[i].inst
            if isa(stmt, ReturnNode)
                push!(ϕ.edges, bb)
                if !isa(stmt.val, SSAValue)
                    push!(ϕ.values, insert_node!(ir, i,
                        NewInstruction(stmt.val)))
                else
                    push!(ϕ.values, stmt.val)
                end
                ir[i] = NewInstruction(GotoNode(new_bb_idx))
                cfg_insert_edge!(cfg, bb, new_bb_idx)
            end
        end

        ir = compact!(ir)
        ir = split_critical_edges!(ir)

        # If that resulted in the return not being the last block, fix that now.
        # We keep things simple this way, such that the basic blocks in the
        # forward and reverse are simply inverses of each other (i.e. the
        # exist block needs to be last, since the entry block needs to be first
        # in the reverse pass).

        if !isa(ir.stmts[end][:inst], ReturnNode)
            new_bb_idx = length(ir.cfg.blocks)+1
            for (bb, i) in bbidxiter(ir)
                stmt = ir.stmts[i][:inst]
                if isa(stmt, ReturnNode)
                    ir[i] = NewInstruction(GotoNode(new_bb_idx))
                    push!(ir, NewInstruction(stmt))
                    push!(ir.cfg, BasicBlock(StmtRange(length(ir.stmts), length(ir.stmts))))
                    cfg_insert_edge!(ir.cfg, bb, new_bb_idx)
                    break
                end
            end
        end

        cfg = ir.cfg

        # Now add a special control flow marker to every basic block
        # TODO: This lowering of control flow is extremely simplistic.
        # It needs to be improved in the very near future, but let's get
        # something working for now.
        for block in cfg.blocks
            if length(block.preds) != 0
                insert_node!(ir, block.stmts.start,
                    NewInstruction(Expr(:phi_placeholder, copy(block.preds))))
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
            if isdefined(Base, :__has_internal_change) && Base.__has_internal_change(v"1.12-alpha", :codeinfonargs)
                opaque_ci.nargs = 2
                opaque_ci.isva = false
            end
        else
            opaque_ci.slotnames = [Symbol("#oc#"), ci.slotnames...]
            opaque_ci.slotflags = UInt8[0, ci.slotflags...]
            if isdefined(Base, :__has_internal_change) && Base.__has_internal_change(v"1.12-alpha", :codeinfonargs)
                opaque_ci.nargs = 1 + ci.nargs
                opaque_ci.isva = ci.isva
            end
        end
        @static if VERSION ≥ v"1.12.0-DEV.173"
            opaque_ci.debuginfo = ci.debuginfo
        else
            opaque_ci.linetable = Core.LineInfoNode[ci.linetable[1]]
            opaque_ci.inferred = false
        end
        opaque_ci
    end

    nfixedargs = Int(meth.nargs)
    meth.isva && (nfixedargs -= 1)

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
        arg_accums = Union{Nothing, Vector{Any}}[nothing for i = 1:Int(meth.nargs)]
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
                current_env = nothing
            elseif isexpr(stmt, :call) || isexpr(stmt, :invoke)
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
                for (j, arg) in enumerate(isexpr(stmt, :invoke) ? stmt.args[2:end] : stmt.args)
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
            elseif isa(stmt, GlobalRef) || isexpr(stmt, :static_parameter) || isexpr(stmt, :throw_undef_if_not) || isexpr(stmt, :loopinfo)
                # We drop gradients for globals and static parameters
            elseif isexpr(stmt, :inbounds)
                # Nothing to do
            elseif isexpr(stmt, :boundscheck)
                # TODO: do something here
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
            next_oc = insert_node_rev!(make_opaque_closure(interp, Tuple{(Any for i = 1:nargs+1)...},
                                                           cname(nc+1, N, meth.name),
                                                           Int(meth.nargs),
                                                           meth.isva,
                                                           lno,
                                                           opaque_cis[nc+1],
                                                           revs[nc+1]...))
            ret_tuple = insert_node_rev!(Expr(:call, tuple, arg_tuple, next_oc))
        end
        insert_node_rev!(Core.ReturnNode(ret_tuple))
        bb_ranges[end] = first(bb_ranges[end]):length(code)

        if has_cfg
            code = opaque_ci.code = expand_switch(code, bb_ranges, slot_map)
        end

        @static if VERSION ≥ v"1.12.0-DEV.173"
            debuginfo = CC.DebugInfoStream(nothing, opaque_ci.debuginfo, length(code))
            debuginfo.def = :var"N/A"
            opaque_ci.debuginfo = Core.DebugInfo(debuginfo, length(code))
        else
            opaque_ci.codelocs = Int32[0 for i=1:length(code)]
        end
        opaque_ci.ssavaluetypes = length(code)
        opaque_ci.ssaflags = SSAFlagType[zero(SSAFlagType) for i=1:length(code)]
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
                next_oc = insert_node_here!(make_opaque_closure(interp, Tuple{Any},
                                                                cname(nc+1, N, meth.name),
                                                                1,
                                                                false,
                                                                lno,
                                                                opaque_cis[nc + 1],
                                                                revs[nc+1]...))
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
            elseif isexpr(stmt, :static_parameter)
                fwds[i] = ZeroTangent()
            elseif isa(stmt, Union{GotoNode, GotoIfNot})
                #@show stmt
                error("Control flow support not fully implemented yet for higher-order reverse mode (TODO)")
            elseif !isa(stmt, Expr)
                @show stmt
                error()
            else
                fwds[i] = insert_node_here!(stmt)
            end
        end

        @static if VERSION ≥ v"1.12.0-DEV.173"
            debuginfo = CC.DebugInfoStream(nothing, opaque_ci.debuginfo, length(code))
            debuginfo.def = :var"N/A"
            opaque_ci.debuginfo = Core.DebugInfo(debuginfo, length(code))
        else
            opaque_ci.codelocs = Int32[0 for i=1:length(code)]
        end
        opaque_ci.ssavaluetypes = length(code)
        opaque_ci.ssaflags = SSAFlagType[zero(SSAFlagType) for i=1:length(code)]
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

    if interp !== nothing
        new_argtypes = Any[Const(∂⃖recurse), tuple_tfunc(CC.optimizer_lattice(interp), ir.argtypes[1:nfixedargs])]
        empty!(ir.argtypes)
        append!(ir.argtypes, new_argtypes)
    end

    rev = revs[1]
    active_bb = 1

    phi_nodes = Any[PhiNode() for _ = 1:length(orig_bb_ranges)]

    for ((old_idx, idx), stmt) in compact
        # remap arguments
        urs = userefs(stmt)
        compact[SSAValue(idx)] = nothing
        for op in urs
            val = op[]
            if isa(val, Argument)
                op[] = arg_mapping[val.n]
            end
            if isexpr(val, :static_parameter)
                op[] = quoted(sparams[val.args[1]])
            end
        end
        compact[SSAValue(idx)] = stmt = urs[]
        # f(args...) -> ∂⃖{N}(args...)
        orig_stmt = stmt

        compact[SSAValue(idx)][:type] = Any # For now
        if isexpr(stmt, :(=))
            stmt = stmt.args[2]
        end
        if isexpr(stmt, :call)
            compact[SSAValue(idx)] = Expr(:call, ∂⃖{N}(), stmt.args...)
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
        elseif isexpr(stmt, :invoke)
            @assert interp !== nothing
            info = compact[SSAValue(idx)][:info]
            if isa(info, RecurseInfo)
                @show info
                new_curs = ADCursor(curs.level + 1, stmt.args[1])
                error()
            else
                new_curs = ADCursor(curs.level, stmt.args[1])
            end
            oc = codegen(interp, new_curs)
            stmt.args[1] = oc.source.specializations[1]
            # TODO: Bad. This is for the closure environment. Figure out what
            # this actually means.
            insert!(stmt.args, 2, ())
        elseif isexpr(stmt, :static_parameter)
            stmt = quoted(sparams[stmt.args[1]])
            if isexpr(orig_stmt, :(=))
                orig_stmt.args[2] = stmt
                stmt = orig_stmt
            end
            compact[SSAValue(idx)] = stmt
        elseif isexpr(stmt, :new) || isexpr(stmt, :splatnew)
            rev[old_idx] = stmt.args[1]
        elseif isexpr(stmt, :phi_placeholder)
            compact[SSAValue(idx)] = phi_nodes[active_bb]
            # TODO: This is a base julia bug
            push!(compact.late_fixup, idx)
            rev[old_idx] = SSAValue(idx)
        elseif isa(stmt, Core.ReturnNode)
            lno = LineNumberNode(1, :none)
            compact[SSAValue(idx)] = make_opaque_closure(interp, Tuple{Any},
                                                         cname(1, N, meth.name),
                                                         1,
                                                         false,
                                                         lno,
                                                         opaque_cis[1],
                                                         rev[orig_bb_ranges[end]]...)
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

        if old_idx == last(orig_bb_ranges[active_bb])
            if length(succs) != 0
                override = false
                if has_terminator[active_bb]
                    terminator = compact[SSAValue(idx)].inst
                    compact[SSAValue(idx)] = nothing
                    override = true
                end
                function terminator_insert_node!(node)
                    if override
                        compact[SSAValue(idx)] = node.stmt
                        override = false
                        return SSAValue(idx)
                    else
                        return insert_node_here!(compact, node, true)
                    end
                end
                tup = terminator_insert_node!(
                    removable_if_unused(NewInstruction(Expr(:call, tuple, rev[orig_bb_ranges[active_bb]]...), Any, Int32(0))))
                for succ in succs
                    preds = cfg.blocks[succ].preds
                    if length(preds) == 1
                        val = tup
                    else
                        selector = findfirst(==(active_bb), preds)
                        val = insert_node_here!(compact, removable_if_unused(NewInstruction(Expr(:call, tuple, selector, tup), Any, Int32(0))), true)
                    end
                    pn = phi_nodes[succ]
                    push!(pn.edges, active_bb)
                    push!(pn.values, val)
                end
                if has_terminator[active_bb]
                    insert_node_here!(compact, NewInstruction(terminator, Any, Int32(0)), true)
                end
            end
            active_bb += 1
        end
    end

    non_dce_finish!(compact)
    ir = complete(compact)
    #@show ir
    ir = compact!(ir)
    CC.verify_ir(ir, true, true)

    return ir
end
