import .CC: abstract_call_gf_by_type, abstract_call_opaque_closure
using .CC: Const, isconstType, argtypes_to_type, tuple_tfunc, Const,
    getfield_tfunc, _methods_by_ftype, VarTable, nfields_tfunc,
    ArgInfo, singleton_type, CallMeta, MethodMatchInfo, specialize_method,
    PartialOpaque, UnionSplitApplyCallInfo, typeof_tfunc, apply_type_tfunc, instanceof_tfunc,
    StmtInfo, NoCallInfo
using Core: PartialStruct
using Base.Meta

get_fname(@nospecialize(fT::DataType)) = @static VERSION ≥ v"1.13.0-DEV.647" ? fT.name.singletonname : fT.name.mt.name

function CC.abstract_call_gf_by_type(interp::ADInterpreter, @nospecialize(f),
        arginfo::ArgInfo, si::StmtInfo, @nospecialize(atype), sv::InferenceState, max_methods::Int)
    (;argtypes) = arginfo
    if interp.backward
        if f isa ∂⃖recurse
            inner_argtypes = argtypes[2:end]
            ft = inner_argtypes[1]
            f = singleton_type(ft)
            rinterp = raise_level(interp)
            call = abstract_call_gf_by_type(rinterp, f, ArgInfo(nothing, inner_argtypes), argtypes_to_type(inner_argtypes), sv, max_methods)
            if isa(call.info, MethodMatchInfo)
                if length(call.info.results.matches) == 0
                    @show inner_argtypes
                    error()
                end
                mi = specialize_method(call.info.results.matches[1], preexisting=true)
                ci = get(rinterp.unopt[rinterp.current_level], mi, nothing)
                clos = AbstractCompClosure(rinterp.current_level, 1, call.info, ci.stmt_info)
                clos = PartialOpaque(Core.OpaqueClosure{<:Tuple, <:Any}, nothing, sv.linfo, clos)
            elseif isa(call.info, RRuleInfo)
                if rinterp.current_level == 1
                    clos = getfield_tfunc(call.info.rrule_rt, Const(2))
                else
                    name = get_fname(call.info.info.results.matches[1].method.sig.parameters[2])
                    clos = PrimClosure(name, rinterp.current_level - 1, 1, getfield_tfunc(call.info.rrule_rt, Const(2)), call.info, nothing)
                end
            end
            # TODO: use abstract_new instead, when it exists
            obtype = instanceof_tfunc(apply_type_tfunc(Const(OpticBundle), typeof_tfunc(call.rt)))[1]
            if obtype isa DataType
                rt2 = PartialStruct(obtype, Any[call.rt, clos])
            else
                rt2 = obtype
            end
            @static if VERSION ≥ v"1.11.0-DEV.945"
            return CallMeta(rt2, call.exct, call.effects, RecurseInfo(call.info))
            else
            return CallMeta(rt2, call.effects, RecurseInfo(call.info))
            end
        end

        # Check if there is a rrule for this function
        if interp.current_level != 0 && f !== ChainRules.rrule
            rrule_argtypes = Any[Const(ChainRules.rrule); argtypes]
            rrule_atype = argtypes_to_type(rrule_argtypes)
            # In general we want the forward type of an rrule'd function to match
            # what the function itself would have returned, but let's support this
            # not being the case.
            if f == accum
                error()
            end
            call = abstract_call_gf_by_type(lower_level(interp), ChainRules.rrule, ArgInfo(nothing, rrule_argtypes), rrule_atype, sv, -1)
            if call.rt != Const(nothing)
                newrt = getfield_tfunc(call.rt, Const(1))
                @static if VERSION ≥ v"1.11.0-DEV.945"
                return CallMeta(newrt, call.exct, call.effects, RRuleInfo(call.rt, call.info))
                else
                return CallMeta(newrt, call.exct, call.effects, RRuleInfo(call.rt, call.info))
                end
            end
        end
    end

    ret = @invoke CC.abstract_call_gf_by_type(interp::AbstractInterpreter, f::Any,
        arginfo::ArgInfo, si::StmtInfo, atype::Any, sv::InferenceState, max_methods::Int)

    if interp.forward
        return fwd_abstract_call_gf_by_type(interp, f, arginfo, si, sv, ret)
    end

    return ret
end

function abstract_accum(interp::AbstractInterpreter, argtypes::Vector{Any}, sv::InferenceState)
    argtypes = filter(@nospecialize(x)->!(widenconst(x) <: Union{ZeroTangent, NoTangent}), argtypes)

    if length(argtypes) == 0
        @static if VERSION ≥ v"1.11.0-DEV.945"
        return CallMeta(ZeroTangent, Any, Effects(), NoCallInfo())
        else
        return CallMeta(ZeroTangent, Effects(), NoCallInfo())
        end
    end

    if length(argtypes) == 1
        @static if VERSION ≥ v"1.11.0-DEV.945"
        return CallMeta(argtypes[1], Any, Effects(), NoCallInfo())
        else
        return CallMeta(argtypes[1], Effects(), NoCallInfo())
        end
    end

    rtype = reduce(tmerge, argtypes)
    if widenconst(rtype) <: Tuple
        targs = Any[]
        for i = 1:nfields_tfunc(rtype).val
            push!(targs, abstract_accum(interp, Any[getfield_tfunc(arg, Const(i)) for arg in argtypes], sv).rt)
        end
        rt = tuple_tfunc(targs)
        @static if VERSION ≥ v"1.11.0-DEV.945"
        return CallMeta(rt, Any, Effects(), NoCallInfo())
        else
        return CallMeta(rt, Effects(), NoCallInfo())
        end
    end
    call = abstract_call(change_level(interp, 0), nothing, Any[typeof(accum), argtypes...],
        sv::InferenceState)
    return call
end

function repackage_apply_rt(info, Δ, argtypes)
    argwise = Any[NoTangent, NoTangent, getfield_prop_zero_tfunc(Δ, Const(1))]
    curarg = 2
    # Repackage this according to the arguments
    for argt in argtypes
        arg_rt = widenconst(argt)
        #@show arg_rt
        if arg_rt <: Tuple
            n = length(arg_rt.parameters)
            tts = Any[]
            for j = curarg:(curarg+n-1)
                push!(tts, getfield_prop_zero_tfunc(Δ, Const(j)))
            end
            push!(argwise, tuple_tfunc(tts))
            curarg += n
        else
            error()
        end
    end
    final_rt = tuple_tfunc(argwise)
    return final_rt
end

function infer_cc_backward(interp::ADInterpreter, cc::AbstractCompClosure, @nospecialize(cc_Δ), sv::InferenceState)
    mi = specialize_method(cc.primal_info.results.matches[1], preexisting=true)
    ni = change_level(interp, cc.order)
    ci = get(code_cache(ni), mi, nothing)
    primal = ci.inferred

    function derive_closure_type(info)
        isa(info, CallMeta) && (info = info.info)
        if isa(info, RRuleInfo)
            if cc.order == 1
                return (0, getfield_tfunc(info.rrule_rt, Const(2)))
            else
                tn = info.info.results.matches[1].method.sig.parameters[2].name
                name = isdefined(tn, :mt) ? tn.mt.name : tn.name
                return (cc.order - 1, PrimClosure(name, cc.order - 1, 1, getfield_tfunc(info.rrule_rt, Const(2)), info, nothing))
            end
        elseif isa(info, MethodMatchInfo)
            mi′ = specialize_method(info.results.matches[1], preexisting=true)
            ci′ = get(code_cache(ni), mi′, nothing)
            clos = AbstractCompClosure(cc.order, cc.seq, info, ci′.inferred.ir.stmts.info)
            clos = PartialOpaque(Core.OpaqueClosure{<:Tuple, <:Any}, nothing, mi, clos)
            return (cc.order, clos)
        elseif isa(info, CompClosInfo)
            return (info.clos.order, AbstractCompClosure(info.clos.order, info.clos.seq + 1, info.clos.primal_info, info.infos))
        elseif isa(info, PrimClosInfo)
            return (info.next.order, info.next)
        elseif isa(info, ReifyInfo)
            return derive_closure_type(info.info)
        else
            @show info
            error()
        end
    end

    primal = primal.ir

    ssa_accums = Vector{Union{Nothing, CallMeta}}(undef, length(primal.stmts))
    ssa_infos = Union{Nothing, CallMeta}[nothing for i = 1:length(primal.stmts)]
    arg_accums = Vector{Union{Nothing, CallMeta}}(undef, length(primal.argtypes))

    arg_typs = Any[Any[] for i = 1:length(primal.argtypes)]
    ssa_typs = Any[Any[] for i = 1:length(primal.stmts)]
    function accum!(use, val)
        if val == Union{}
            @show (use, val)
            @show cc.seq
            error()
        end
        if isa(use, SSAValue)
            push!(ssa_typs[use.id], val)
        elseif isa(use, Argument)
            push!(arg_typs[use.n], val)
        end
    end

    function bail!(inst)
        @show ("bail", inst)
        error()
        for i = 1:length(inst.args)
            accum!(inst.args[i], Any)
        end
    end

    function arg_accum!(inst, rt)
        isinvoke = isexpr(inst, :invoke)
        if rt === Union{}
            @show primal
            @show inst
            @show rt
            error()
        end
        for i = (isinvoke ? 2 : 1):length(inst.args)
            accum!(inst.args[i], getfield_tfunc(rt, Const(i - (isinvoke ? 1 : 0))))
        end
    end

    for i = length(primal.stmts):-1:1
        inst = primal.stmts[i][:inst]

        if isa(inst, ReturnNode)
            accum!(inst.val, cc_Δ)
            continue
        elseif isa(inst, GlobalRef) || isexpr(inst, :static_parameter)
            continue
        else
            if !(isexpr(inst, :call) || isexpr(inst, :new) || isexpr(inst, :invoke))
                @show mi
                @show inst
                @show primal
                error()
            end
        end

        accum_call = abstract_accum(change_level(interp, 0), ssa_typs[i], sv)
        ssa_accums[i] = accum_call
        Δ = accum_call.rt
        if Δ == Union{}
            error()
        end

        if isexpr(inst, :new)
            for i = 2:length(inst.args)
                # TODO: This is wrong
                rt = getfield_tfunc(Δ, Const(i-1))
                # Struct gradients are allowed to be sparse
                rt === Union{} && (rt = ZeroTangent)
                accum!(inst.args[i], rt)
            end
            accum!(inst.args[1], ChainRules.NoTangent)
            continue
        end

        info = cc.prev_seq_infos[i]
        isa(info, CallMeta) && (info = info.info)

        call_info = info
        while isa(call_info, UnionSplitApplyCallInfo)
            @assert length(info.infos) == 1
            call_info = call_info.infos[1].call
        end

        if call_info === nothing
            if isexpr(inst, :invoke)
                error()
            else
                ft = argextype(inst.args[1], primal, primal.sptypes)
                f = singleton_type(ft)
                if isa(f, Core.Builtin)
                    rt = backwards_tfunc(f, primal, inst, Δ)
                    @static if VERSION ≥ v"1.11.0-DEV.945"
                    call = CallMeta(rt, Any, Effects(), NoCallInfo())
                    else
                    call = CallMeta(rt, Effects(), NoCallInfo())
                    end
                else
                    bail!(inst)
                    continue
                end
            end
        else
            if cc.seq == 1 && isa(call_info, CompClosInfo)
                call_info = ReifyInfo(call_info)
            end

            if isa(call_info, RecurseInfo)
                clos = getfield_tfunc(Δ, Const(2))
                arg = getfield_tfunc(Δ, Const(1))
                call = abstract_call(interp, nothing, Any[clos, arg], sv)
                # No derivative wrt the functor
                rt = tuple_tfunc(Any[NoTangent; tuple_type_fields(call.rt)...])
                @static if VERSION ≥ v"1.11.0-DEV.945"
                call = CallMeta(rt, Any, Effects(), ReifyInfo(call.info))
                else
                call = CallMeta(rt, Effects(), ReifyInfo(call.info))
                end
            else
                (level, close) = derive_closure_type(call_info)
                call = abstract_call(change_level(interp, level), ArgInfo(nothing, Any[close, Δ]), sv)
            end
        end

        if isa(info, UnionSplitApplyCallInfo)
            argts = Any[argextype(inst.args[i], primal, primal.sptypes) for i = 4:length(inst.args)]
            rt = repackage_apply_rt(info, call.rt, argts)
            newinfo = UnionSplitApplyCallInfo([ApplyCallInfo(call.info)])
            @static if VERSION ≥ v"1.11.0-DEV.945"
            call = CallMeta(rt, Any, Effects(), newinfo)
            else
            call = CallMeta(rt, Effects(), newinfo)
            end
        end

        if isa(call_info, ReifyInfo)
            new_rt = tuple_tfunc(Any[derive_closure_type(call.info)[2]; call.rt])
            newinfo = RecurseInfo(call.info)
            @static if VERSION ≥ v"1.11.0-DEV.945"
            call = CallMeta(new_rt, Any, Effects(), newinfo)
            else
            call = CallMeta(new_rt, Effects(), newinfo)
            end
        end

        if call.rt === Union{}
            @show inst
            @show Δ
            @show cc_Δ
            @show mi
            @show primal
            @show cc.order
            @show close
            error()
        end

        arg_accum!(inst, call.rt)
        ssa_infos[i] = call
        continue
    end


    tup_elemns = Any[]
    for (i, this_arg_typs) in enumerate(arg_typs)
        let tup_push! = (mi.def.isva && i == length(arg_typs)) ? (a, t)->append!(a, tuple_type_fields(t)) : push!
            if length(this_arg_typs) <= 1
                push!(arg_accums, nothing)
                length(this_arg_typs) == 0 && tup_push!(tup_elemns, ChainRules.NoTangent)
                length(this_arg_typs) == 1 && tup_push!(tup_elemns, this_arg_typs[1])
                continue
            end
            accum_call = abstract_accum(interp, this_arg_typs, sv)
            if accum_call.rt == Union{}
                @show accum_call.rt
                @static if VERSION ≥ v"1.11.0-DEV.945"
                return CallMeta(Union{}, Any, Effects(), NoCallInfo())
                else
                return CallMeta(Union{}, Effects(), NoCallInfo())
                end
            end
            push!(arg_accums, accum_call)
            tup_push!(tup_elemns, accum_call.rt)
        end
    end

    rt = tuple_tfunc(Any[tup_elemns...])
    @static if VERSION ≥ v"1.11.0-DEV.945"
    return CallMeta(rt, Any, Effects(), CompClosInfo(cc, ssa_infos))
    else
    return CallMeta(rt, Effects(), CompClosInfo(cc, ssa_infos))
    end
end

function infer_cc_forward(interp::ADInterpreter, cc::AbstractCompClosure, @nospecialize(cc_Δ), sv::InferenceState)
    @show ("enter", cc_Δ, cc)

    primal_mi = specialize_method(cc.primal_info.results.matches[1], true)
    ni = change_level(interp, cc.order)
    ci = get(code_cache(ni), primal_mi, nothing)
    primal = ci.inferred

    function derive_closure_type(info)
        isa(info, CallMeta) && (info = info.info)
        if isa(info, RRuleInfo)
            return (cc.order - 1, getfield_tfunc(info.rrule_rt, Const(2)))
        elseif isa(info, MethodMatchInfo)
            mi′ = specialize_method(info.results.matches[1], true)
            ci′ = get(code_cache(lower_level(ni)), mi′, nothing)

            return (0, AbstractCompClosure(cc.order - 1, 1, info, ci′.inferred.stmts.info))
        elseif isa(info, CompClosInfo)
            return (info.clos.order, AbstractCompClosure(info.clos.order, info.clos.seq + 1, info.clos.primal_info, info.infos))
        elseif isa(info, PrimClosInfo)
            return (info.next.order, info.next)
        elseif isa(info, ReifyInfo)
            return derive_closure_type(info.info)
        else
            @show info
            error()
        end
    end

    function accum_arg(arg)
        if isa(arg, SSAValue)
            if !isassigned(accums, arg.id)
                @show primal
                @show arg
                error()
            end
            return accums[arg.id]
        elseif isa(arg, Argument)
            argn = arg.n
            if primal_mi.def.isva && argn == length(primal.argtypes)
                return tuple_tfunc(Any[tuple_type_fields(cc_Δ)[argn:end]...])
            else
                return getfield_tfunc(cc_Δ, Const(argn))
            end
        else
            return NoTangent
        end
    end

    ssa_accums = Vector{Union{Nothing, CallMeta}}(undef, length(primal.stmts))
    ssa_infos = Vector{Union{Nothing, CallMeta}}(undef, length(primal.stmts))

    accums = Vector{Any}(undef, length(primal.stmts))
    vals = Vector{Union{Nothing, CallMeta}}(undef, length(primal.stmts))

    for i = 1:length(primal.stmts)
        inst = primal.stmts[i][:inst]
        info = cc.prev_seq_infos[i]

        if isa(inst, GlobalRef)
            accums[i] = NoTangent
            continue
        end

        if isa(inst, ReturnNode)
            rt = accum_arg(inst.val)
            @static if VERSION ≥ v"1.11.0-DEV.945"
            return CallMeta(rt, Any, Effects(), CompClosInfo(cc, ssa_infos))
            else
            return CallMeta(rt, Effects(), CompClosInfo(cc, ssa_infos))
            end
        end

        args = Any[]
        for i = 1:length(inst.args)
            push!(args, accum_arg(inst.args[i]))
        end

        if isexpr(inst, :new)
            T, exact = instanceof_tfunc(argextype(inst.args[1], primal, primal.sptypes))
            # Special case for empty structs
            if isempty(fieldnames(T))
                accums[i] = ZeroTangent
                continue
            end
            Δ = NamedTuple{fieldnames(T), widenconst(tuple_tfunc(args[2:end]))}
            accums[i] = Δ
            continue
        end

        Δ = tuple_tfunc(args)

        if info === nothing
            ssa_infos[i] = nothing
            ft = argextype(inst.args[1], primal, primal.sptypes)
            f = singleton_type(ft)
            if isa(f, Core.Builtin)
                accums[i] = forward_tfunc(f, primal, inst, Δ)
                continue
            else
                bail!(inst)
                continue
            end
        end

        ssa_accums[i] = nothing
        isa(info, CallMeta) && (info = info.info)

        call_info = info
        while isa(call_info, UnionSplitApplyCallInfo)
            @assert length(info.infos) == 1
            call_info = call_info.infos[1].call

            # Tranforms Δ
            args = Any[]
            push!(args, tuple_type_fields(Δ)[3])
            for arg in tuple_type_fields(Δ)[4:end]
                append!(args, tuple_type_fields(arg))
            end
            Δ = tuple_tfunc(args)
        end

        if isa(call_info, ReifyInfo)
            # TODO: Is this always the case?
            Δ = tuple_tfunc(Any[tuple_type_fields(Δ)[2:end]...])
        end

        if isa(call_info, RecurseInfo)
            clos = getfield_tfunc(Δ, Const(1))
            arg = getfield_tfunc(Δ, Const(2))
            call = abstract_call(interp, nothing, Any[clos, arg], sv)
            # No derivative wrt the functor
            newrt = tuple_tfunc(Any[NoTangent; tuple_type_fields(call.rt)...])
            @static if VERSION ≥ v"1.11.0-DEV.945"
            call = CallMeta(newrt, Any, Effects(), ReifyInfo(call.info))
            else
            call = CallMeta(newrt, Effects(), ReifyInfo(call.info))
            end
            #error()
        else
            (level, clos) = derive_closure_type(call_info)

            call = abstract_call(change_level(interp, level), nothing, Any[clos, Δ], sv)
        end

        if isa(call_info, ReifyInfo)
            new_rt = tuple_tfunc(Any[call.rt; derive_closure_type(call.info)[2]])
            @static if VERSION ≥ v"1.11.0-DEV.945"
            call = CallMeta(new_rt, Any, Effects(), RecurseInfo())
            else
            call = CallMeta(new_rt, Effects(), RecurseInfo())
            end
        end

        if isa(info, UnionSplitApplyCallInfo)
            newinfo = UnionSplitApplyCallInfo([ApplyCallInfo(call.info)])
            @static if VERSION ≥ v"1.11.0-DEV.945"
            call = CallMeta(call.rt, call.exct, Effects(), newinfo)
            else
            call = CallMeta(call.rt, Effects(), newinfo)
            end
        end

        accums[i] = call.rt
        @show (i, accums[i])
        ssa_infos[i] = call
    end

    error()
end

function infer_comp_closure(interp::ADInterpreter, cc::AbstractCompClosure, @nospecialize(Δ), sv::InferenceState)
    if cc.seq & 1 != 0
        infer_cc_backward(interp, cc, Δ, sv)
    else
        infer_cc_forward(interp, cc, Δ, sv)
    end
end

function infer_prim_closure(interp::ADInterpreter, pc::PrimClosure, @nospecialize(Δ), sv::InferenceState)
    if pc.seq == 1
        call = abstract_call(change_level(interp, pc.order), nothing, Any[pc.dual, Δ], sv)
        rt = call.rt
        @show (pc, Δ, rt)
        newinfo = PrimClosInfo(PrimClosure(pc.name, pc.order, pc.seq + 1, nothing, call.info, pc.info_below))
        @static if VERSION ≥ v"1.11.0-DEV.945"
        return CallMeta(call.rt, call.exct, Effects(), newinfo)
        else
        return CallMeta(call.rt, Effects(), newinfo)
        end
    elseif pc.seq == 2
        ni = change_level(interp, pc.order)
        mi′ = specialize_method(pc.info_below.results.matches[1], true)
        ci′ = get(code_cache(ni), mi′, nothing)
        cc = AbstractCompClosure(pc.order, 1, pc.info_below, ci′.inferred.stmts.info)
        call = infer_comp_closure(ni, cc, Δ, sv)
        rt = getfield_tfunc(call.rt, Const(2))
        @show (pc, Δ, rt)
        newinfo = PrimClosInfo(PrimClosure(pc.name, pc.order, pc.seq + 1, getfield_tfunc(call.rt, Const(1)), call.info, pc.info_carried))
        @static if VERSION ≥ v"1.11.0-DEV.945"
        return CallMeta(rt, Any, Effects(), newinfo)
        else
        return CallMeta(rt, Effects(), newinfo)
        end
    elseif pc.seq == 3
        ni = change_level(interp, pc.order)
        mi′ = specialize_method(pc.info_carried.info.results.matches[1], true)
        ci′ = get(code_cache(ni), mi′, nothing)
        clos = AbstractCompClosure(pc.order, 1, pc.info_carried.info, ci′.inferred.stmts.info)
        call = abstract_call(change_level(interp, pc.order), nothing,
            Any[clos, tuple_tfunc(Any[Δ, pc.dual])], sv)
        rt = tuple_tfunc(Any[tuple_type_fields(call.rt)[2:end]...])
        @show (pc, Δ, rt)
        newinfo = PrimClosInfo(PrimClosure(pc.name, pc.order, pc.seq + 1, nothing, call.info, pc.info_below))
        @static if VERSION ≥ v"1.11.0-DEV.945"
        return CallMeta(rt, Any, Effects(), newinfo)
        else
        return CallMeta(rt, Effects(), newinfo)
        end
    elseif mod(pc.seq, 4) == 0
        info = pc.info_below
        clos = AbstractCompClosure(info.clos.order, info.clos.seq + 1, info.clos.primal_info, info.infos)
        # Add back gradient w.r.t. rrule
        Δ = tuple_tfunc(Any[NoTangent, tuple_type_fields(Δ)...])
        call = abstract_call(change_level(interp, pc.order), nothing, Any[clos, Δ], sv)
        rt = getfield_tfunc(call.rt, Const(1))
        @show (pc, Δ, rt)
        newinfo = PrimClosInfo(PrimClosure(pc.name, pc.order, pc.seq + 1, getfield_tfunc(call.rt, Const(2)), call.info, pc.info_carried))
        @static if VERSION ≥ v"1.11.0-DEV.945"
        return CallMeta(rt, Any, Effects(), newinfo)
        else
        return CallMeta(rt, Effects(), newinfo)
        end
    elseif mod(pc.seq, 4) == 1
        info = pc.info_carried
        clos = AbstractCompClosure(info.clos.order, info.clos.seq + 1, info.clos.primal_info, info.infos)
        call = abstract_call(change_level(interp, pc.order), nothing, Any[clos, tuple_tfunc(Any[pc.dual, Δ])], sv)
        rt = call.rt
        @show (pc, Δ, rt)
        newinfo = PrimClosInfo(PrimClosure(pc.name, pc.order, pc.seq + 1, nothing, call.info, pc.info_below))
        @static if VERSION ≥ v"1.11.0-DEV.945"
        return CallMeta(rt, Any, Effects(), newinfo)
        else
        return CallMeta(rt, Effects(), newinfo)
        end
    elseif mod(pc.seq, 4) == 2
        info = pc.info_below
        clos = AbstractCompClosure(info.clos.order, info.clos.seq + 1, info.clos.primal_info, info.infos)
        call = abstract_call(change_level(interp, pc.order), nothing, Any[clos, Δ], sv)
        rt = getfield_tfunc(call.rt, Const(2))
        @show (pc, Δ, rt)
        newinfo = PrimClosInfo(PrimClosure(pc.name, pc.order, pc.seq + 1, getfield_tfunc(call.rt, Const(1)), call.info, pc.info_carried))
        @static if VERSION ≥ v"1.11.0-DEV.945"
        return CallMeta(rt, Any, Effects(), newinfo)
        else
        return CallMeta(rt, Effects(), newinfo)
        end
    elseif mod(pc.seq, 4) == 3
        info = pc.info_carried
        clos = AbstractCompClosure(info.clos.order, info.clos.seq + 1, info.clos.primal_info, info.infos)
        call = abstract_call(change_level(interp, pc.order), nothing, Any[clos, tuple_tfunc(Any[Δ, pc.dual])], sv)
        rt = tuple_tfunc(Any[tuple_type_fields(call.rt)[2:end]...])
        @show (pc, Δ, rt)
        newinfo = PrimClosInfo(PrimClosure(pc.name, pc.order, pc.seq + 1, nothing, call.info, pc.info_below))
        @static if VERSION ≥ v"1.11.0-DEV.945"
        return CallMeta(rt, Any, Effects(), newinfo)
        else
        return CallMeta(rt, Effects(), newinfo)
        end
    end
    error()
end

function CC.abstract_call_opaque_closure(interp::ADInterpreter,
    closure::PartialOpaque, arginfo::ArgInfo, sv::InferenceState, check::Bool=true)

    if isa(closure.source, AbstractCompClosure)
        (;argtypes) = arginfo
        if length(argtypes) !== 2
            error("bad argtypes")
        end
        return infer_comp_closure(interp, closure.source, argtypes[2], sv)
    elseif isa(closure.source, PrimClosure)
        (;argtypes) = arginfo
        return infer_prim_closure(interp, closure.source, argtypes[2], sv)
    end

    return @invoke CC.abstract_call_opaque_closure(interp::AbstractInterpreter,
        closure::PartialOpaque, arginfo::ArgInfo, sv::InferenceState, check::Bool)
end
