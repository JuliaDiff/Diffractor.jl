using .CC: IRInterpretationState, construct_postdomtree, PiNode,
    is_known_call, argextype, postdominates, userefs, PhiCNode, UpsilonNode

#=
function forward_diff!(ir::IRCode, interp, irsv::IRInterpretationState, to_diff::Vector{Pair{SSAValue, Int}}; custom_diff! = (args...)->nothing, diff_cache=Dict{SSAValue, SSAValue}())
    Δs = SSAValue[]
    rets = findall(@nospecialize(x)->isa(x, ReturnNode) && isdefined(x, :val), ir.stmts.inst)
    postdomtree = construct_postdomtree(ir.cfg.blocks)
    for (ssa, order) in to_diff
        Δssa = forward_diff!(ir, interp, irsv, ssa, order; custom_diff!, diff_cache)
        Δblock = block_for_inst(ir, Δssa.id)
        for idx in rets
            retblock = block_for_inst(ir, idx)
            if !postdominates(postdomtree, retblock, Δblock)
                error("Stmt %$ssa does not dominate all return blocks $(rets)")
            end
        end
        push!(Δs, Δssa)
    end
    return (ir, Δs)
end
=#

# TODO interp::AbstractADInterpreter instead interp::AbstractInterpreter?

function forward_diff!(ir::IRCode, interp::AbstractInterpreter, irsv::IRInterpretationState,
                       ssa::SSAValue, order::Int;
                       custom_diff!, diff_cache)
    if haskey(diff_cache, ssa)
        return diff_cache[ssa]
    end
    inst = ir[ssa]
    stmt = inst[:inst]
    Δssa = forward_diff_uncached!(ir, interp, irsv, ssa, inst, order::Int; custom_diff!, diff_cache)
    @assert Δssa !== nothing
    if isa(Δssa, SSAValue)
        diff_cache[ssa] = Δssa
    end
    return Δssa
end
function forward_diff!(ir::IRCode, interp::AbstractInterpreter, irsv::IRInterpretationState,
                       val, order::Int;
                       custom_diff!, diff_cache, eras_mode)
    return ChainRulesCore.zero_tangent(val)
end
function forward_diff!(ir::IRCode, interp::AbstractInterpreter, irsv::IRInterpretationState,
                       arg::Argument, order::Int;
                       custom_diff!, diff_cache, eras_mode)
    recurse(x) = forward_diff!(ir, interp, irsv, x; custom_diff!, diff_cache, eras_mode)
    val = custom_diff!(ir, SSAValue(0), arg, recurse)
    if val !== nothing
        return val
    end
    return ChainRulesCore.NoTangent()
end

function forward_diff_uncached!(ir::IRCode, interp::AbstractInterpreter, irsv::IRInterpretationState,
                                ssa::SSAValue, inst::CC.Instruction, order::Int;
                                custom_diff!, diff_cache, eras_mode)
    stmt = inst[:inst]
    recurse(x) = forward_diff!(ir, interp, irsv, x, order; custom_diff!, diff_cache, eras_mode)
    if (val = custom_diff!(ir, ssa, stmt, recurse)) !== nothing
        return val
    elseif isa(stmt, PiNode)
        return recurse(stmt.val)
    elseif isa(stmt, SSAValue)
        return recurse(stmt)
    elseif isa(stmt, PhiNode)
        Δphi = PhiNode(copy(stmt.edges), similar(stmt.values))
        T = Union{}
        for i in 1:length(stmt.values)
            isassigned(stmt.values, i) || continue
            Δphi.values[i] = recurse(stmt.values[i])
            T = CC.tmerge(CC.optimizer_lattice(interp), T, argextype(Δphi.values[i], ir))
        end
        return insert_node!(ir, ssa, NewInstruction(Δphi, T), true)
    elseif is_known_call(stmt, tuple, ir)
        Δtpl = Expr(:call, GlobalRef(Core, :tuple))
        for arg in stmt.args[2:end]
            arg = recurse(arg)
            push!(Δtpl.args, arg)
        end
        argtypes = Any[argextype(arg, ir) for arg in Δtpl.args[2:end]]
        tup_typ = CC.tuple_tfunc(CC.typeinf_lattice(interp), argtypes)
        Δssa = insert_node!(ir, ssa, NewInstruction(Δtpl, tup_typ), true)
        return Δssa
    elseif isexpr(stmt, :new)
        Δtpl = Expr(:call, GlobalRef(Core, :tuple))
        for arg in stmt.args[2:end]
            push!(Δtpl.args, recurse(arg))
        end
        argtypes = Any[argextype(arg, ir) for arg in Δtpl.args[2:end]]
        tup_typ = CC.tuple_tfunc(CC.typeinf_lattice(interp), argtypes)
        Δbacking = insert_node!(ir, ssa, NewInstruction(Δtpl, tup_typ))
        newT = argextype(stmt.args[1], ir)
        @assert isa(newT, Const)
        tup_typ_typ = CC.typeof_tfunc(tup_typ)
        if !(newT.val <: Tuple)
            tup_typ_typ = CC.apply_type_tfunc(Const(NamedTuple{fieldnames(newT.val)}), tup_typ_typ)
            Δbacking = insert_node!(ir, ssa, NewInstruction(Expr(:splatnew, widenconst(tup_typ), Δbacking), tup_typ_typ.val))
        end
        tangentT = CC.apply_type_tfunc(Const(ChainRulesCore.Tangent), newT, tup_typ_typ).val
        Δtangent = insert_node!(ir, ssa, NewInstruction(Expr(:new, tangentT, Δbacking), tangentT))
        return Δtangent
    else # general frule handling
        info = inst[:info]
        if isexpr(stmt, :invoke)
            args = stmt.args[2:end]
        else
            args = copy(stmt.args)
        end
        Δtpl = Expr(:call, GlobalRef(Core, :tuple), nothing)
        for arg in args[2:end]
            push!(Δtpl.args, recurse(arg))
        end
        argtypes = Any[argextype(arg, ir) for arg in Δtpl.args[2:end]]
        tup_T = CC.tuple_tfunc(CC.typeinf_lattice(interp), argtypes)

        Δ = insert_node!(ir, ssa, NewInstruction(Δtpl, tup_T))

        # Now that we know the arguments, do a proper typeinf for this particular callsite
        new_spec_types = Tuple{typeof(ChainRulesCore.frule), widenconst(tup_T), (widenconst(argextype(arg, ir)) for arg in args)...}
        new_match = Base._which(new_spec_types)

        # Now do proper type inference with the known arguments
        interp′ = disable_forward(interp)
        new_frame = CC.typeinf_frame(interp′, new_match.method, new_match.spec_types, new_match.sparams, #=run_optimizer=#true)

        # Create :invoke expression for the newly inferred frule
        frule_mi = CC.EscapeAnalysis.analyze_match(new_match, length(args)+2)
        frule_call = Expr(:invoke, frule_mi, GlobalRef(ChainRulesCore, :frule), Δ, args...)
        frule_flag = CC.flags_for_effects(new_frame.ipo_effects)

        result = new_frame.result.result
        if isa(result, Const) && result.val === nothing
            error("Diffractor thought we had an frule at inference time, but no frule found")
        end

        # Incidence analysis through the rt call
        # TODO: frule_mi is wrong here, should be the mi of the caller
        frule_rt = info.frule_call.rt
        (improve_frule_rt, nothrow) = CC.concrete_eval_invoke(interp, frule_call, frule_mi, irsv)
        if improve_frule_rt !== nothing
            frule_rt = improve_frule_rt
        end
        frule_result = insert_node!(ir, ssa, NewInstruction(
            frule_call, frule_rt, info.frule_call.info, inst[:line],
            frule_flag))
        replace_call!(ir, ssa, Expr(:call, GlobalRef(Core, :getfield), frule_result, 1))
        Δssa = insert_node!(ir, ssa, NewInstruction(
            Expr(:call, GlobalRef(Core, :getfield), frule_result, 2), CC.getfield_tfunc(CC.typeinf_lattice(interp), frule_rt, Const(2))), #=attach_after=#true)
        return Δssa
    end
end

function forward_visit!(ir::IRCode, ssa::SSAValue, order::Int, ssa_orders::Vector{Pair{Int, Bool}}, visit_custom!)
    if ssa_orders[ssa.id][1] >= order
        return
    end
    ssa_orders[ssa.id] = order => ssa_orders[ssa.id][2]
    inst = ir[ssa]
    stmt = inst[:inst]
    recurse(@nospecialize(val), new_order=order) = forward_visit!(ir, val, new_order, ssa_orders, visit_custom!)
    if visit_custom!(ir, ssa, order, recurse)
        ssa_orders[ssa.id] = order => true
        return
    elseif isa(stmt, PiNode)
        return recurse(stmt.val)
    elseif isa(stmt, PhiNode)
        for i = 1:length(stmt.values)
            isassigned(stmt.values, i) || continue
            recurse(stmt.values[i])
        end
        return
    elseif isexpr(stmt, :new) || isexpr(stmt, :invoke)
        foreach(recurse, stmt.args[2:end])
    elseif isexpr(stmt, :call)
        foreach(recurse, stmt.args)
    elseif isa(stmt, SSAValue)
        recurse(stmt)
    elseif isexpr(stmt, :code_coverage_effect)
        return
    elseif isexpr(stmt, :loopinfo) #TODO preserve this properly
        return
    elseif isexpr(stmt, :boundscheck)
        return
    elseif !isa(stmt, Expr)
        return
    else
        @show stmt
        error()
    end
end
forward_visit!(::IRCode, @nospecialize(x), ::Int, ::Vector{Pair{Int, Bool}}, _) = nothing
function forward_visit!(ir::IRCode, a::Argument, order::Int, ssa_orders::Vector{Pair{Int, Bool}}, visit_custom!)
    recurse(@nospecialize(val)) = forward_visit!(ir, val, order, ssa_orders, visit_custom!)
    return visit_custom!(ir, a, order, recurse)
end


"""
	forward_diff_no_inf!(ir::IRCode, to_diff::Vector{Pair{SSAValue,Int}}; visit_custom!, transform!)

Internal method which generates the code for forward mode diffentiation


 - `ir` the IR being differnetation
 - `to_diff`: collection of all SSA values for which the derivative is to be taken,
              paired with the order (first deriviative, second derivative etc)

 - `visit_custom!(ir::IRCode, ssa, order::Int, recurse::Bool) -> Bool`:
		decides if the custom `transform!` should be applied to a `stmt` or not
		Default: `false` for all statements
 - `transform!(ir::IRCode, ssa::SSAValue, order::Int)` mutates `ir` to do a custom tranformation.
 - `eras_mode`: determines if to error if not all derivatives are taylor
"""
function forward_diff_no_inf!(ir::IRCode, to_diff::Vector{Pair{SSAValue,Int}};
                              eras_mode = false,
                              visit_custom! = (@nospecialize args...)->false,
                              transform! = (@nospecialize args...)->error())
    # Step 1: For each SSAValue in the IR, keep track of the differentiation order needed
    ssa_orders = [-1=>false for i = 1:length(ir.stmts)]
    for (ssa, order) in to_diff
        forward_visit!(ir, ssa, order, ssa_orders, visit_custom!)
    end
    for (ssa, (order, custom)) in enumerate(ssa_orders)
        if order == -1
            ssa_orders[ssa] = 0 => custom
        end
    end

    truncation_map = Dict{Pair{SSAValue, Int}, SSAValue}()

    # Step 2: Transform
    function maparg(@nospecialize(arg), ssa::SSAValue, order::Int)
        if isa(arg, SSAValue)
            if arg.id > length(ssa_orders)
                # This is possible if the custom transform touched another statement.
                # In that case just pass this through and assume the `transform!` did
                # it correctly.
                return arg
            end
            (argorder, _) = ssa_orders[arg.id]
            if argorder != order
                @assert order < argorder
                return get!(truncation_map, arg=>order) do
                    # identify where to insert. Must be after phi blocks
                    pos = SSAValue(find_end_of_phi_block(ir, arg.id))
                    if order == 0
                        insert_node!(ir, pos, NewInstruction(Expr(:call, primal, arg)), #=attach_after=#true)
                    else
                        insert_node!(ir, pos, NewInstruction(Expr(:call, truncate, arg, Val{order}())), #=attach_after=#true)
                    end
                end
            end
            return arg
        elseif order == 0
            return arg
        elseif isa(arg, Argument)
            # TODO: Should we remember whether the callbacks wanted the arg?
            return transform!(ir, arg, order, maparg)
        elseif isa(arg, GlobalRef)
            @assert isconst(arg.mod, arg.name)
            return zero_bundle{order}()(getfield(arg.mod, arg.name))
        elseif isa(arg, QuoteNode)
            return zero_bundle{order}()(arg.value)
        end
        @assert !isa(arg, Expr)
        return zero_bundle{order}()(arg)
    end

    for (ssa, (order, custom)) in enumerate(ssa_orders)
        if custom
            transform!(ir, SSAValue(ssa), order, maparg)
        elseif order == 0
            inst = ir[SSAValue(ssa)]
            stmt = inst[:inst]
            urs = userefs(stmt)
            for ur in urs
                ur[] = maparg(ur[], SSAValue(ssa), order)
            end
            inst[:inst] = urs[]
            continue
        else
            inst = ir[SSAValue(ssa)]
            stmt = inst[:inst]
            if isexpr(stmt, :invoke)
                newargs = map(stmt.args[2:end]) do @nospecialize arg
                    maparg(arg, SSAValue(ssa), order)
                end
                replace_call!(ir, SSAValue(ssa), Expr(:call, ∂☆{order, eras_mode}(), newargs...))
            elseif isexpr(stmt, :call) || isexpr(stmt, :new)
                newargs = map(stmt.args) do @nospecialize arg
                    maparg(arg, SSAValue(ssa), order)
                end
                f = isexpr(stmt, :call) ? ∂☆{order, eras_mode}() : ∂☆new{order}()
                replace_call!(ir, SSAValue(ssa), Expr(:call, f, newargs...))
            elseif isa(stmt, PiNode)
                # TODO: New PiNode that discriminates based on primal?
                inst[:inst] = maparg(stmt.val, SSAValue(ssa), order)
                inst[:type] = Any
                inst[:flag] |= CC.IR_FLAG_REFINED
            elseif isa(stmt, GlobalRef)
                if !isconst(stmt)
                    # Non-const GlobalRefs need to need to be accessed as seperate statements
                    stmt = insert_node!(ir, ssa, NewInstruction(inst))
                end

                replace_call!(ir, SSAValue(ssa), Expr(:call, zero_bundle{order}(), stmt))
            elseif isa(stmt, SSAValue) || isa(stmt, QuoteNode)
                inst[:inst] = maparg(stmt, SSAValue(ssa), order)
                inst[:type] = Any
                inst[:flag] |= CC.IR_FLAG_REFINED
            elseif isexpr(stmt, :boundscheck)
                val = ZeroBundle{order}(stmt.args[1])
                inst[:inst] = val
                inst[:type] = Const(val)
                inst[:flag] |= CC.IR_FLAG_REFINED
            elseif isa(stmt, Expr) || isa(stmt, PhiNode) || isa(stmt, PhiCNode) ||
                   isa(stmt, UpsilonNode) || isa(stmt, GotoIfNot) || isa(stmt, Argument)
                urs = userefs(stmt)
                for ur in urs
                    ur[] = maparg(ur[], SSAValue(ssa), order)
                end
                inst[:inst] = urs[]
                inst[:type] = Any
                inst[:flag] |= CC.IR_FLAG_REFINED
            else
                val = zero_bundle{order}()(inst[:inst])
                inst[:inst] = val
                inst[:type] = Const(val)
            end
        end
    end
end

function forward_diff!(interp::ADInterpreter, ir::IRCode, src::CodeInfo, mi::MethodInstance,
                       to_diff::Vector{Pair{SSAValue, Int}}; kwargs...)
    forward_diff_no_inf!(ir, to_diff; kwargs...)

    # Step 3: Re-inference

    ir = compact!(ir)

    for i = 1:length(ir.stmts)
        inst = ir[SSAValue(i)][:inst]
        if !isa(inst, ReturnNode) && ir[SSAValue(i)][:type] === Any
            if iszero(ir[SSAValue(i)][:flag] & CC.IR_FLAG_REFINED)
                @warn "IR_FLAG_REFINED Flag missed on statement" i inst
                ir[SSAValue(i)][:flag] |= CC.IR_FLAG_REFINED
            end
        end
    end

    method_info = CC.MethodInfo(src)
    argtypes = ir.argtypes[1:mi.def.nargs]
    world = get_inference_world(interp)
    irsv = IRInterpretationState(interp, method_info, ir, mi, argtypes, world, src.min_world, src.max_world)
    rt = CC._ir_abstract_constant_propagation(interp, irsv)

    ir = compact!(ir)

    return ir
end
