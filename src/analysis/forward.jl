using Core.Compiler: StmtInfo, ArgInfo, CallMeta

function fwd_abstract_call_gf_by_type(interp::AbstractInterpreter, @nospecialize(f),
        arginfo::ArgInfo, si::StmtInfo, sv::InferenceState, primal_call::CallMeta)
    if f === ChainRulesCore.frule
        # TODO: Currently, we don't have any termination analysis for the non-stratified
        # forward analysis, so bail out here.
        return nothing
    end

    nargs = length(arginfo.argtypes)-1

    # Here we simply check for the frule existance - we don't want to do a full
    # inference with specialized argtypes and everything since the problem is
    # likely sparse and we only need to do a full inference on a few calls.
    # Thus, here we pick `Any` for the tangent types rather than trying to
    # discover what they are. frules should be written in such a way that
    # whether or not they return `nothing`, only depends on the non-tangent arguments
    frule_preargtypes = Any[Const(ChainRulesCore.frule), Tuple{Nothing,Vararg{Any,nargs}}]
    frule_argtypes = append!(frule_preargtypes, arginfo.argtypes)
    frule_arginfo = ArgInfo(nothing, frule_argtypes)
    frule_si = StmtInfo(true)
    frule_atype = CC.argtypes_to_type(frule_argtypes)
    # turn off frule analysis in the frule to avoid cycling
    interp′ = disable_forward(interp)
    frule_call = CC.abstract_call_gf_by_type(interp′,
        ChainRulesCore.frule, frule_arginfo, frule_si, frule_atype, sv, #=max_methods=#-1)
    if frule_call.rt !== Const(nothing)
        return CallMeta(primal_call.rt, primal_call.effects, FRuleCallInfo(primal_call.info, frule_call))
    else
        CC.add_mt_backedge!(sv, frule_mt, frule_atype)
    end

    return nothing
end

const frule_mt = methods(ChainRulesCore.frule).mt
