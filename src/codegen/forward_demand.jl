using Core.Compiler: IRInterpretationState, construct_postdomtree, PiNode,
    is_known_call, argextype, postdominates

function forward_diff!(ir::IRCode, interp, irsv::IRInterpretationState, pantelides::Vector{SSAValue}; custom_diff! = (args...)->nothing, diff_cache=Dict{SSAValue, SSAValue}())
    Δs = SSAValue[]
    rets = findall(@nospecialize(x)->isa(x, ReturnNode) && isdefined(x, :val), ir.stmts.inst)
    postdomtree = construct_postdomtree(ir.cfg.blocks)
    for ssa in pantelides
        Δssa = forward_diff!(ir, interp, irsv, ssa; custom_diff!, diff_cache)
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

function diff_unassigned_variable!(ir, ssa)
    return insert_node!(ir, ssa, NewInstruction(
        Expr(:call, GlobalRef(Intrinsics, :state_ddt), ssa), Float64), #=attach_after=#true)
end

function forward_diff!(ir::IRCode, interp, irsv::IRInterpretationState, ssa::SSAValue; custom_diff!, diff_cache)
    if haskey(diff_cache, ssa)
        return diff_cache[ssa]
    end
    inst = ir[ssa]
    stmt = inst[:inst]
    if isa(stmt, SSAValue)
        return forward_diff!(ir, interp, irsv, stmt; custom_diff!, diff_cache)
    end
    Δssa = forward_diff_uncached!(ir, interp, irsv, ssa, inst; custom_diff!, diff_cache)
    @assert Δssa !== nothing
    if isa(Δssa, SSAValue)
        diff_cache[ssa] = Δssa
    end
    return Δssa
end
forward_diff!(ir::IRCode, interp, irsv::IRInterpretationState, val::Union{Integer, AbstractFloat}; custom_diff!, diff_cache) = zero(val)
forward_diff!(ir::IRCode, interp, irsv::IRInterpretationState, @nospecialize(arg); custom_diff!, diff_cache) = ChainRulesCore.NoTangent()
function forward_diff!(ir::IRCode, interp, irsv::IRInterpretationState, arg::Argument; custom_diff!, diff_cache)
    recurse(x) = forward_diff!(ir, interp, irsv, x; custom_diff!, diff_cache)
    val = custom_diff!(ir, SSAValue(0), arg, recurse)
    if val !== nothing
        return val
    end
    return ChainRulesCore.NoTangent()
end

function forward_diff_uncached!(ir::IRCode, interp, irsv::IRInterpretationState, ssa::SSAValue, inst::Core.Compiler.Instruction; custom_diff!, diff_cache)
    stmt = inst[:inst]
    recurse(x) = forward_diff!(ir, interp, irsv, x; custom_diff!, diff_cache)
    if (val = custom_diff!(ir, ssa, stmt, recurse)) !== nothing
        return val
    elseif isa(stmt, PiNode)
        return recurse(stmt.val)
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
        tup_typ_typ = Core.Compiler.typeof_tfunc(tup_typ)
        if !(newT.val <: Tuple)
            tup_typ_typ = Core.Compiler.apply_type_tfunc(Const(NamedTuple{fieldnames(newT.val)}), tup_typ_typ)
            Δbacking = insert_node!(ir, ssa, NewInstruction(Expr(:splatnew, widenconst(tup_typ), Δbacking), tup_typ_typ.val))
        end
        tangentT = Core.Compiler.apply_type_tfunc(Const(ChainRulesCore.Tangent), newT, tup_typ_typ).val
        Δtangent = insert_node!(ir, ssa, NewInstruction(Expr(:new, tangentT, Δbacking), tangentT))
        return Δtangent
    else # general frule handling
        info = inst[:info]
        if !isa(info, FRuleCallInfo)
            @show info
            @show inst[:inst]
            display(ir)
            error()
        end
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

        Δ = insert_node!(ir, ssa, NewInstruction(
            Δtpl, tup_T))

        # Now that we know the arguments, do a proper typeinf for this particular callsite
        new_spec_types = Tuple{typeof(ChainRulesCore.frule), widenconst(tup_T), (widenconst(argextype(arg, ir)) for arg in args)...}
        new_match = Base._which(new_spec_types)

        # Now do proper type inference with the known arguments
        interp′ = disable_forward(interp)
        new_frame = Core.Compiler.typeinf_frame(interp′, new_match.method, new_match.spec_types, new_match.sparams, #=run_optimizer=#true)

        # Create :invoke expression for the newly inferred frule
        frule_mi = CC.EscapeAnalysis.analyze_match(new_match, length(args)+2)
        frule_call = Expr(:invoke, frule_mi, GlobalRef(ChainRulesCore, :frule), Δ, args...)
        frule_flag = CC.flags_for_effects(new_frame.ipo_effects)

        result = new_frame.result.result
        if isa(result, Const) && result.val === nothing
            error("DAECompiler thought we had an frule at inference time, but no frule found")
        end

        # Incidence analysis through the rt call
        # TODO: frule_mi is wrong here, should be the mi of the caller
        frule_rt = info.frule_call.rt
        improve_frule_rt = CC.concrete_eval_invoke(interp, frule_call, frule_mi, irsv)
        if improve_frule_rt !== nothing
            frule_rt = improve_frule_rt
        end
        frule_result = insert_node!(ir, ssa, NewInstruction(
            frule_call, frule_rt, info.frule_call.info, inst[:line],
            frule_flag))
        ir[ssa][:inst] = Expr(:call, GlobalRef(Core, :getfield), frule_result, 1)
        Δssa = insert_node!(ir, ssa, NewInstruction(
            Expr(:call, GlobalRef(Core, :getfield), frule_result, 2), CC.getfield_tfunc(CC.typeinf_lattice(interp), frule_rt, Const(2))), #=attach_after=#true)
        return Δssa
    end
end
