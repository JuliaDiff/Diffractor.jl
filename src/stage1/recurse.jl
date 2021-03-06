using Core.Compiler: MethodInstance, IncrementalCompact, insert_node_here!,
    userefs, SlotNumber, IRCode, compute_basic_blocks, _methods_by_ftype,
    retrieve_code_info, CodeInfo, SSAValue, finish, complete, non_dce_finish!,
    GotoNode, GotoIfNot, block_for_inst, ReturnNode, Argument, compact!

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

using Core.Compiler: construct_domtree, scan_slot_def_use, construct_ssa!

Base.iterate(c::IncrementalCompact, args...) = Core.Compiler.iterate(c, args...)
Base.iterate(p::Core.Compiler.Pair, args...) = Core.Compiler.iterate(p, args...)
Base.iterate(urs::Core.Compiler.UseRefIterator, args...) = Core.Compiler.iterate(urs, args...)
Base.getindex(urs::Core.Compiler.UseRefIterator, args...) = Core.Compiler.getindex(urs, args...)
Base.getindex(urs::Core.Compiler.UseRef, args...) = Core.Compiler.getindex(urs, args...)
Base.setindex!(c::Core.Compiler.IncrementalCompact, args...) = Core.Compiler.setindex!(c, args...)
Base.setindex!(urs::Core.Compiler.UseRef, args...) = Core.Compiler.setindex!(urs, args...)
function transform!(ci, meth, nargs, sparams, N)
    @assert N == 1

    code = ci.code
    cfg = compute_basic_blocks(code)
    renumber_cfg!(cfg, code)
    slotnames = Symbol[Symbol("#self#"), :args, ci.slotnames...]
    slotflags = UInt8[(0x00 for i = 1:2)..., ci.slotflags...]
    slottypes = UInt8[(0x00 for i = 1:2)..., ci.slotflags...]
    ir = IRCode(Core.Compiler.InstructionStream(code, Any[],
        Any[nothing for i = 1:length(code)],
        ci.codelocs, UInt8[0 for i = 1:length(code)]), cfg, Core.LineInfoNode[ci.linetable...],
        Any[Any for i = 1:length(nargs)], Any[], Any[sparams...])


    # SSA conversion
    domtree = construct_domtree(ir.cfg.blocks)
    defuse_insts = scan_slot_def_use(nargs, ci, ir.stmts.inst)
    ci.ssavaluetypes = Any[Any for i = 1:ci.ssavaluetypes]
    ir = construct_ssa!(ci, ir, domtree, defuse_insts, nargs, Any[Any for i = 1:length(slotnames)])
    ir = compact!(ir)

    rev = Any[nothing for i = 1:length(ir.stmts)]
    code = Any[] # Will be filled in later
    opaque_ci = ccall(:jl_new_code_info_uninit, Ref{CodeInfo}, ())
    opaque_ci.code = code
    opaque_ci.slotnames = Symbol[Symbol("#self#"), :Δ]
    opaque_ci.slotflags = UInt8[0, 0]
    opaque_ci.linetable = Core.LineInfoNode[ci.linetable[1]]
    opaque_ci.inferred = false

    arg_accums = Union{Nothing, Vector{Any}}[nothing for i = 1:(nargs+1)]
    accums = Union{Nothing, Vector{Any}}[nothing for i = 1:length(ir.stmts)]
    is_accumable(val) = isa(val, SSAValue) || isa(val, Argument)
    function accum!(val, accum)
        is_accumable(val) || return
        if isa(val, SSAValue)
            id = val.id
            if isa(accums[id], Nothing)
                accums[id] = Any[]
            end
            push!(accums[id], accum)
        elseif isa(val, Argument)
            id = val.n
            if isa(arg_accums[id], Nothing)
                arg_accums[id] = Any[]
            end
            push!(arg_accums[id], accum)
        end
    end

    function insert_node_rev!(node)
        push!(code, node)
        SSAValue(length(code))
    end

    function do_accum(this_accums)
        if this_accums === nothing || isempty(this_accums)
            return ChainRulesCore.Zero()
        elseif length(this_accums) == 1
            return this_accums[]
        else
            return insert_node_rev!(Expr(:call, accum, this_accums...))
        end
    end

    for i in reverse(1:length(ir.stmts))
        stmt = ir.stmts[i][:inst]
        if isa(stmt, Core.ReturnNode)
            accum!(stmt.val, Argument(2))
        elseif isexpr(stmt, :call)
            Δ = do_accum(accums[i])
            callee = insert_node_rev!(Expr(:call, getfield, Argument(1), i))
            call = insert_node_rev!(Expr(:call, callee, Δ))
            for (j, arg) in enumerate(stmt.args)
                if is_accumable(arg)
                    accum!(arg, insert_node_rev!(Expr(:call, getfield, call, j)))
                end
            end
        elseif isexpr(stmt, :new)
            Δ = do_accum(accums[i])
            canon = insert_node_rev!(Expr(:call, ChainRulesCore.canonicalize, Δ))
            nt = insert_node_rev!(Expr(:call, ChainRulesCore.backing, canon))
            for (j, arg) in enumerate(stmt.args)
                j == 1 && continue
                if is_accumable(arg)
                    accum!(arg, insert_node_rev!(Expr(:call, getfield, nt, j-1)))
                end
            end
        elseif isexpr(stmt, :splatnew)
            Δ = do_accum(accums[i])
            canon = insert_node_rev!(Expr(:call, ChainRulesCore.canonicalize, Δ))
            nt = insert_node_rev!(Expr(:call, ChainRulesCore.backing, canon))
            arg = stmt.args[2]
            if is_accumable(stmt.args[2])
                accum!(stmt.args[2], nt)
            end
        elseif isa(stmt, GlobalRef) || isexpr(stmt, :static_parameter)
            # We drop gradients for globals and static parameters
        else
            @show stmt
            error()
        end
    end

    arg_tuple = insert_node_rev!(Expr(:call, tuple, [do_accum(arg_accums[i]) for i = 1:(nargs+1)]...))
    insert_node_rev!(Core.ReturnNode(arg_tuple))

    opaque_ci.codelocs = Int32[0 for i=1:length(code)]
    opaque_ci.ssavaluetypes = length(code)
    opaque_ci.ssaflags = UInt8[0 for i=1:length(code)]

    compact = IncrementalCompact(ir)

    nfixedargs = meth.isva ? meth.nargs - 1 : meth.nargs
    meth.isva || @assert nfixedargs == nargs+1

    arg_mapping = Any[]
    for argno in 1:nfixedargs
        push!(arg_mapping, insert_node_here!(compact,
            Expr(:call, getfield, Argument(2), argno),
            Any, Int32(0)))
    end


    if meth.isva
        # Extract the rest of the arguments and make a tuple out of them
        ssas = map((nfixedargs+1):(nargs+1)) do i
            insert_node_here!(compact,
                Expr(:call, getfield, Argument(2), i-1),
                Any, Int32(0))
        end
        push!(arg_mapping,
            insert_node_here!(compact,
                Expr(:call, tuple, ssas...), Any, Int32(0)))
    end

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
        # f(args...) -> ∂⃖{1}(args...)
        orig_stmt = stmt
        if isexpr(stmt, :(=))
            stmt = stmt.args[2]
        end
        if isexpr(stmt, :call)
            compact[idx] = Expr(:call, ∂⃖{1}(), stmt.args...)
            if isexpr(orig_stmt, :(=))
                orig_stmt.args[2] = stmt
                stmt = orig_stmt
            end
            compact.ssa_rename[compact.idx-1] = insert_node_here!(compact, Expr(:call, getfield, SSAValue(idx), 1), Any, compact.result[idx][:line], true)
            rev[old_idx] = insert_node_here!(compact, Expr(:call, getfield, SSAValue(idx), 2), Any, compact.result[idx][:line], true)
        elseif isexpr(stmt, :static_parameter)
            stmt = sparams[stmt.args[1]]
            if isexpr(orig_stmt, :(=))
                orig_stmt.args[2] = stmt
                stmt = orig_stmt
            end
            compact[idx] = stmt
        elseif isa(stmt, Core.ReturnNode)
            lno = LineNumberNode(1, :none)
            compact[idx] = Expr(:new_opaque_closure, Tuple{Any}, false, Union{}, Any,
                Expr(:opaque_closure_method, 1, lno, opaque_ci), rev...)
            argty = insert_node_here!(compact, Expr(:call, typeof, stmt.val), Any, compact.result[idx][:line], true)
            applyty = insert_node_here!(compact, Expr(:call, Core.apply_type, OpticBundle, argty), Any, compact.result[idx][:line], true)
            retval = insert_node_here!(compact, Expr(:new, applyty, stmt.val, SSAValue(idx)), Any, compact.result[idx][:line], true)
            compact.ssa_rename[compact.idx-1] = insert_node_here!(compact, Core.ReturnNode(retval), Any, compact.result[idx][:line], true)
        end
    end

    non_dce_finish!(compact)
    ir = complete(compact)

    Core.Compiler.replace_code_newstyle!(ci, ir, nargs+1)
    ci.ssavaluetypes = length(ci.code)
    ci.slotnames = slotnames
    ci.slotflags = slotflags
    ci.slottypes = slottypes
    ci
end
