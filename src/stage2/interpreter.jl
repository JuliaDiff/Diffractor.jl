using OffsetArrays
using Cthulhu

struct ADCursor <: Cthulhu.AbstractCursor
    level::Int
    mi::MethodInstance
    transformed::Bool
end
# Cthulhu.get_mi(c::ADCursor) = c.mi
ADCursor(level::Int, mi::MethodInstance) = ADCursor(level, mi, false)

#=
Cthulhu.get_cursor(c::ADCursor, callinfo::Cthulhu.PullbackCallInfo) = ADCursor(c.level+1, Cthulhu.get_mi(callinfo.mi))
Cthulhu.get_cursor(c::ADCursor, cs::Cthulhu.Callsite) = Cthulhu.get_cursor(c, cs.info)
Cthulhu.get_cursor(c::ADCursor, callinfo) = ADCursor(c.level, Cthulhu.get_mi(callinfo))
=#

struct ADGraph
    code::OffsetVector{Dict{MethodInstance, Any}}
    msgs::Vector{Tuple{Int, MethodInstance, Int, String}}
    entry_mi::MethodInstance
end
entrypoint(graph::ADGraph) = ADCursor(0, graph.entry_mi)

using Core: MethodInstance, CodeInstance
using .CC: AbstractInterpreter, ArgInfo, Effects, InferenceResult, InferenceState,
    IRInterpretationState, NativeInterpreter, OptimizationState, StmtInfo, WorldRange

const OptCache = Dict{MethodInstance, CodeInstance}
const UnoptCache = Dict{Union{MethodInstance, InferenceResult}, Cthulhu.InferredSource}
const RemarksCache = Dict{Union{MethodInstance,InferenceResult}, Cthulhu.PC2Remarks}

struct ADInterpreter <: AbstractInterpreter
    # Modes settings
    forward::Bool
    backward::Bool

    # This cache is stratified by AD nesting level. Depending on the
    # nesting level of the derivative, The AD primitives may behave
    # differently.
    # Level 0 == Straightline Code, no AD
    # Level 1 == Gradients
    # Level 2 == Seconds Derivatives
    # and so on
    opt::OffsetVector{OptCache}
    unopt::Union{OffsetVector{UnoptCache},Nothing}
    transformed::OffsetVector{OptCache}

    native_interpreter::NativeInterpreter
    current_level::Int
    remarks::OffsetVector{RemarksCache}

    function _ADInterpreter()
        return new(
            #=forward::Bool=#false,
            #=backward::Bool=#true,
            #=opt::OffsetVector{OptCache}=#OffsetVector([OptCache(), OptCache()], 0:1),
            #=unopt::Union{OffsetVector{UnoptCache},Nothing}=#OffsetVector([UnoptCache(), UnoptCache()], 0:1),
            #=transformed::OffsetVector{OptCache}=#OffsetVector([OptCache(), OptCache()], 0:1),
            #=native_interpreter::NativeInterpreter=#NativeInterpreter(),
            #=current_level::Int=#0,
            #=remarks::OffsetVector{RemarksCache}=#OffsetVector([RemarksCache()], 0:0))
    end
    function ADInterpreter(interp::ADInterpreter = _ADInterpreter();
                           forward::Bool = interp.forward,
                           backward::Bool = interp.backward,
                           opt::OffsetVector{OptCache} = interp.opt,
                           unopt::Union{OffsetVector{UnoptCache},Nothing} = interp.unopt,
                           transformed::OffsetVector{OptCache} = interp.transformed,
                           native_interpreter::NativeInterpreter = interp.native_interpreter,
                           current_level::Int = interp.current_level,
                           remarks::OffsetVector{RemarksCache} = interp.remarks)
        return new(forward, backward, opt, unopt, transformed, native_interpreter, current_level, remarks)
    end
end

change_level(interp::ADInterpreter, new_level::Int) = ADInterpreter(interp; current_level=new_level)
raise_level(interp::ADInterpreter) = change_level(interp, interp.current_level + 1)
lower_level(interp::ADInterpreter) = change_level(interp, interp.current_level - 1)

disable_forward(interp::ADInterpreter) = ADInterpreter(interp; forward=false)

#=
function Cthulhu.get_optimized_codeinst(interp::ADInterpreter, curs::ADCursor)
    @show curs
    (curs.transformed ? interp.transformed : interp.opt)[curs.level][curs.mi]
end
Cthulhu.AbstractCursor(interp::ADInterpreter, mi::MethodInstance) = ADCursor(0, mi, false)
=#

# This is a lie, but let's clean this up later
Cthulhu.can_descend(interp::ADInterpreter, @nospecialize(key), optimize::Bool) = true

function Cthulhu.lookup(interp::ADInterpreter, curs::ADCursor, optimize::Bool; allow_no_src::Bool=false)
    if !optimize
        entry = interp.unopt[curs.level][curs.mi]
        codeinf = src = copy(entry.src)
        rt = entry.rt
        infos = entry.stmt_info
        effects = Cthulhu.get_effects(entry)
        slottypes = src.slottypes
        if isnothing(slottypes)
            slottypes = Any[ Any for i = 1:length(src.slotflags) ]
        end
    else
        codeinst = Cthulhu.get_optimized_codeinst(interp, curs)
        rt = Cthulhu.codeinst_rt(codeinst)
        opt = codeinst.inferred
        if opt !== nothing
            opt = opt::Cthulhu.OptimizedSource
            src = CC.copy(opt.ir)
            codeinf = opt.src
            infos = src.stmts.info
            slottypes = src.argtypes
        elseif allow_no_src
            # This doesn't showed up as covered, but it is (see the CI test with `coverage=false`).
            # But with coverage on, the empty function body isn't empty due to :code_coverage_effect expressions.
            codeinf = src = nothing
            infos = []
            slottypes = Any[Base.unwrap_unionall(mi.specTypes).parameters...]
        else
            Core.eval(Main, quote
                interp = $interp
                mi = $mi
                optimize = $optimize
            end)
            error("couldn't find the source; inspect `Main.interp` and `Main.mi`")
        end
        effects = Cthulhu.get_effects(codeinst)
    end
    (; src, rt, infos, slottypes, codeinf, effects)
end

function Cthulhu.custom_toggles(interp::ADInterpreter)
    return Cthulhu.CustomToggle[
        Cthulhu.CustomToggle(false, 'a', "utomatic differentiation",
            function (curs::Cthulhu.AbstractCursor)
                if curs isa ADCursor
                    return ADCursor(curs.level, curs.mi, true)
                end
                return curs
            end,
            function (curs::Cthulhu.AbstractCursor)
                if curs isa ADCursor
                    return ADCursor(curs.level, curs.mi, false)
                end
                return curs
            end)
    ]
end

# TODO: Something is going very wrong here
function Cthulhu.get_effects(interp::ADInterpreter, mi::MethodInstance, opt::Bool)
    if haskey(interp.unopt[0], mi)
        return interp.unopt[0][mi].effects
    else
        return Effects()
    end
end

function CC.is_same_frame(interp::ADInterpreter, linfo::MethodInstance, frame::InferenceState)
    linfo === frame.linfo || return false
    return interp.current_level === frame.interp.current_level
end

# Special handling for Recursion
#=
struct RecurseCallInfo <: Cthulhu.CallInfo
    vmi::Cthulhu.CallInfo # callinfo to be descended
end
Cthulhu.get_mi((; vmi)::RecurseCallInfo) = Cthulhu.get_mi(vmi)
Cthulhu.get_rt((; vmi)::RecurseCallInfo) = Cthulhu.get_rt(vmi)
Cthulhu.get_effects(::RecurseCallInfo) = Effects()

function Cthulhu.print_callsite_info(limiter::IO, info::RecurseCallInfo)
    print(limiter, "< Diffractor recurse > ")
    Cthulhu.show_callinfo(limiter, info.vmi)
end

# Special handling for rrule
struct RRuleCallInfo <: Cthulhu.CallInfo
    ci::Cthulhu.CallInfo # callinfo of the rrule
end
Cthulhu.get_mi((; ci)::RRuleCallInfo) = Cthulhu.get_mi(ci)
Cthulhu.get_rt((; ci)::RRuleCallInfo) = Cthulhu.get_rt(ci)
Cthulhu.get_effects(::RRuleCallInfo) = Effects()

function Cthulhu.print_callsite_info(limiter::IO, info::RRuleCallInfo)
    print(limiter, "< rrule > ")
    Cthulhu.show_callinfo(limiter, info.ci)
end

# Special handling for comp closure
struct CompClosCallInfo <: Cthulhu.CallInfo
    rt
end
Cthulhu.get_mi((; ci)::CompClosCallInfo) = nothing
Cthulhu.get_rt((; rt)::CompClosCallInfo) = rt
Cthulhu.get_effects(::CompClosCallInfo) = Effects()

function Cthulhu.print_callsite_info(limiter::IO, info::CompClosCallInfo)
    print(limiter, "< cc > ")
end

# Navigation
function Cthulhu.navigate(curs::ADCursor, callsite::Cthulhu.Callsite)
    if isa(callsite.info, RecurseCallInfo)
        return ADCursor(curs.level + 1, Cthulhu.get_mi(callsite))
    elseif isa(callsite.info, RRuleCallInfo)
        return ADCursor(curs.level - 1, Cthulhu.get_mi(callsite))
    end
    return ADCursor(curs.level, Cthulhu.get_mi(callsite))
end
=#

function Cthulhu.process_info(interp::ADInterpreter, @nospecialize(info::CC.CallInfo), argtypes::Cthulhu.ArgTypes, @nospecialize(rt), optimize::Bool, @nospecialize(exct))
    if isa(info, RecurseInfo)
        newargtypes = argtypes[2:end]
        callinfos = Cthulhu.process_info(interp, info.info, newargtypes, Cthulhu.unwrapType(widenconst(rt)), optimize, exct)
        if length(callinfos) == 1
            vmi = only(callinfos)
        else
            @assert isempty(callinfos)
            argt = Cthulhu.unwrapType(widenconst(newargtypes[2]))::DataType
            sig = Tuple{widenconst(newargtypes[1]), argt.parameters...}
            vmi = Cthulhu.FailedCallInfo(sig, Union{})
        end
        return Any[RecurseCallInfo(vmi)]
    elseif isa(info, RRuleInfo)
        newargtypes = [Const(rrule); argtypes[2:end]]
        callinfos = Cthulhu.process_info(interp, info.info, newargtypes, Cthulhu.unwrapType(widenconst(rt)), optimize, exct)
        if length(callinfos) == 1
            vmi = only(callinfos)
        else
            @assert isempty(callinfos)
            argt = Cthulhu.unwrapType(widenconst(newargtypes[2]))::DataType
            sig = Tuple{widenconst(newargtypes[1]), argt.parameters...}
            vmi = Cthulhu.FailedCallInfo(sig, Union{})
        end
        return Any[RRuleCallInfo(vmi)]
    elseif isa(info, CompClosInfo)
        return Any[CompClosCallInfo(rt)]
    end
    return invoke(Cthulhu.process_info, Tuple{AbstractInterpreter, CC.CallInfo, Cthulhu.ArgTypes, Any, Bool},
        interp, info, argtypes, rt, optimize)
end

CC.InferenceParams(ei::ADInterpreter) = InferenceParams(ei.native_interpreter)
CC.OptimizationParams(ei::ADInterpreter) = OptimizationParams(ei.native_interpreter)
#=CC.=#get_inference_world(ei::ADInterpreter) = get_inference_world(ei.native_interpreter)
CC.get_inference_cache(ei::ADInterpreter) = get_inference_cache(ei.native_interpreter)

# No need to do any locking since we're not putting our results into the runtime cache
CC.lock_mi_inference(ei::ADInterpreter, mi::MethodInstance) = nothing
CC.unlock_mi_inference(ei::ADInterpreter, mi::MethodInstance) = nothing

@static if VERSION ≥ v"1.11.0-DEV.1552"
CC.cache_owner(ei::ADInterpreter) = ei.opt
end

function CC.code_cache(ei::ADInterpreter)
    while ei.current_level > lastindex(ei.opt)
        push!(ei.opt, Dict{MethodInstance, Any}())
    end
    ei.opt[ei.current_level]
end
CC.may_optimize(ei::ADInterpreter) = true
CC.may_compress(ei::ADInterpreter) = false
CC.may_discard_trees(ei::ADInterpreter) = false

function CC.add_remark!(interp::ADInterpreter, sv::InferenceState, msg)
    key = (@static VERSION ≥ v"1.12.0-DEV.317" ? CC.is_constproped(sv) : CC.any(sv.result.overridden_by_const)) ? sv.result : sv.linfo
    push!(get!(Cthulhu.PC2Remarks, interp.remarks[interp.current_level], key), sv.currpc=>msg)
end

# TODO: `get_remarks` should get a cursor?
#Cthulhu.get_remarks(interp::ADInterpreter, key::Union{MethodInstance,InferenceResult}) = get(interp.remarks[interp.current_level], key, nothing)

@static if VERSION ≥ v"1.13.0-DEV.126"
function diffractor_finish(@specialize(finishfunc), state::InferenceState, interp::ADInterpreter, cycleid::Int)
    res = @invoke finishfunc(state::InferenceState, interp::AbstractInterpreter, cycleid::Int)
    key = CC.is_constproped(state) ? state.result : state.linfo
    interp.unopt[interp.current_level][key] = Cthulhu.InferredSource(state)
    return res
end
else
function diffractor_finish(@specialize(finishfunc), state::InferenceState, interp::ADInterpreter)
    res = @invoke finishfunc(state::InferenceState, interp::AbstractInterpreter)
    key = (@static VERSION ≥ v"1.12.0-DEV.317" ? CC.is_constproped(state) : CC.any(state.result.overridden_by_const)) ? state.result : state.linfo
    interp.unopt[interp.current_level][key] = Cthulhu.InferredSource(state)
    return res
end
end

@static if VERSION ≥ v"1.12-"
CC.finishinfer!(state::InferenceState, interp::ADInterpreter, cycleid::Int) = diffractor_finish(CC.finishinfer!, state, interp, cycleid)
function CC.finish!(interp::ADInterpreter, caller::InferenceState, validation_world::UInt, time_before::UInt64)
    Cthulhu.set_cthulhu_source!(caller.result)
    return @invoke CC.finish!(interp::AbstractInterpreter, caller::InferenceState, validation_world::UInt, time_before::UInt64)
end
elseif VERSION ≥ v"1.11-"
CC.finish(state::InferenceState, interp::ADInterpreter) = diffractor_finish(CC.finish, state, interp)
function CC.finish!(interp::ADInterpreter, caller::InferenceState)
    result = caller.result
    opt = result.src
    Cthulhu.set_cthulhu_source!(result)
    if opt isa CC.OptimizationState
        CC.ir_to_codeinf!(opt)
    end
    return nothing
end
function CC.transform_result_for_cache(::ADInterpreter, ::MethodInstance, ::WorldRange,
                                        result::InferenceResult)
    return result.src
end

else # VERSION < v"1.11-"
CC.finish(state::InferenceState, interp::ADInterpreter) = diffractor_finish(CC.finish, state, interp)
function CC.transform_result_for_cache(::ADInterpreter, ::MethodInstance, ::WorldRange,
                                        result::InferenceResult)
    return create_cthulhu_source(result.src, result.ipo_effects)
end
function CC.finish!(::ADInterpreter, caller::InferenceResult)
    Cthulhu.set_cthulhu_source(interp, caller)
end

end # @static if

const StmtFlag = @static VERSION ≥ v"1.11.0-DEV.377" ? UInt32 : UInt8
function diffractor_inlining_policy(@nospecialize(src), @nospecialize(info::CC.CallInfo),
                                    stmt_flag::StmtFlag)
    # Disallow inlining things away that have an frule
    if isa(info, FRuleCallInfo)
        return nothing
    end
    @static if VERSION < v"1.11.0-DEV.879"
        if isa(src, CC.SemiConcreteResult)
            return src
        end
    end
    @assert isa(src, Cthulhu.OptimizedSource) || isnothing(src)
    if isa(src, Cthulhu.OptimizedSource)
        if CC.is_stmt_inline(stmt_flag) || src.isinlineable
            return src.ir
        end
        return nothing
    end
    return missing
end

@static if VERSION ≥ v"1.12.0-DEV.45"
function CC.src_inlining_policy(interp::ADInterpreter,
    @nospecialize(src), @nospecialize(info::CC.CallInfo), stmt_flag::StmtFlag)
    ret = diffractor_inlining_policy(src, info, stmt_flag)
    ret === nothing && return false
    ret !== missing && return true
    return @invoke CC.src_inlining_policy(interp::AbstractInterpreter,
        src::Any, info::CC.CallInfo, stmt_flag::StmtFlag)
end
else
function CC.inlining_policy(interp::ADInterpreter,
    @nospecialize(src), @nospecialize(info::CC.CallInfo), stmt_flag::StmtFlag,
    mi::MethodInstance, argtypes::Vector{Any})
    ret = diffractor_inlining_policy(src, info, stmt_flag)
    ret === nothing && return nothing
    ret !== missing && return ret
    # the default inlining policy may try additional effor to find the source in a local cache
    return @invoke CC.inlining_policy(interp::AbstractInterpreter,
        nothing, info::CC.CallInfo,
        stmt_flag::StmtFlag,
        mi::MethodInstance, argtypes::Vector{Any})
end
end

#=
function CC.optimize(interp::ADInterpreter, opt::OptimizationState,
    params::OptimizationParams, caller::InferenceResult)

    # TODO: Enable some amount of inlining
    #@timeit "optimizer" ir = run_passes(opt.src, opt, caller)

    sv = opt
    ci = opt.src
    ir = CC.convert_to_ircode(ci, sv)
    ir = CC.slot2reg(ir, ci, sv)
    # TODO: Domsorting can produce an updated domtree - no need to recompute here
    ir = CC.compact!(ir)
    return CC.finish(interp, opt, params, ir, caller)
end
=#

@static if VERSION ≥ v"1.11.0-DEV.1278"
function CC.bail_out_const_call(interp::ADInterpreter, result::CC.MethodCallResult,
                                si::StmtInfo, sv::CC.AbsIntState)
    if result.rt isa CC.LimitedAccuracy
        return false
    end
    return @invoke CC.bail_out_const_call(interp::AbstractInterpreter, result::CC.MethodCallResult,
                                          si::StmtInfo, sv::CC.AbsIntState)
end
end

function ir2codeinst(ir::IRCode, inst::CodeInstance, ci::CodeInfo)
    CodeInstance(inst.def, inst.rettype, isdefined(inst, :rettype_const) ? inst.rettype_const : nothing,
                 Cthulhu.OptimizedSource(CC.copy(ir), ci, inst.inferred.isinlineable, CC.decode_effects(inst.purity_bits)),
                 Int32(0), inst.min_world, inst.max_world, inst.ipo_purity_bits, inst.purity_bits,
                 inst.argescapes, inst.relocatability)
end

using Core: OpaqueClosure
function codegen(interp::ADInterpreter, curs::ADCursor, cache=Dict{ADCursor, OpaqueClosure}())
    ir = CC.copy(Cthulhu.get_optimized_codeinst(interp, curs).inferred.ir)
    codeinst = interp.opt[curs.level][curs.mi]
    ci = codeinst.inferred.src
    if curs.level >= 1
        ir = diffract_ir!(ir, ci, curs.mi.def, curs.level, interp, curs)
        interp.transformed[curs.level][curs.mi] = ir2codeinst(ir, codeinst, ci)
        return OpaqueClosure(ir; isva=true)
    end
    duals = Vector{SSAValue}(undef, length(ir.stmts))
    for i = 1:length(ir.stmts)
        inst = ir.stmts[i][:inst]
        info = ir.stmts[i][:info]
        if isexpr(inst, :invoke)
            if isa(info, RecurseInfo)
                @show info
                new_curs = ADCursor(curs.level + 1, inst.args[1])
                error()
            else
                new_curs = ADCursor(curs.level, inst.args[1])
            end
            if haskey(cache, new_curs)
                oc = cache[new_curs]
            else
                oc = codegen(interp, new_curs, cache)
            end
            inst.args[1] = oc.source.specializations[1]
        elseif isexpr(inst, :call)
            if isa(info, RecurseInfo)
                mi′ = specialize_method(info.info.results.matches[1], preexisting=true)
                new_curs = ADCursor(curs.level + 1, mi′)
                if haskey(cache, new_curs)
                    oc = cache[new_curs]
                else
                    oc = codegen(interp, new_curs, cache)
                end
                rrule_mi = oc.source.specializations[1]
                rrule_rt = Any # TODO
                rrule_call = insert_node!(ir, i, NewInstruction(Expr(:invoke, rrule_mi, inst.args...), rrule_rt))
                ir.stmts[i][:inst] = rrule_call
            elseif isa(info, RRuleInfo)
                rrule_mi = specialize_method(info.info.results.matches[1], preexisting=true)
                (;rrule_rt) = info
                rrule_call = insert_node!(ir, i, NewInstruction(Expr(:invoke, rrule_mi, rrule, inst.args...), rrule_rt))
                arg1 = insert_node!(ir, i, NewInstruction(Expr(:call, getfield, rrule_call, 1), getfield_tfunc(rrule_rt, Const(1))))
                arg2 = insert_node!(ir, i, NewInstruction(Expr(:call, getfield, rrule_call, 2), getfield_tfunc(rrule_rt, Const(2))))
                ir.stmts[i][:inst] = arg1
                duals[i] = arg2
            elseif curs.level != 0
                @show inst
                @show info
                error()
            end
        end
    end
    ir = compact!(ir)
    resize!(ir.argtypes, length(curs.mi.specTypes.parameters))
    interp.transformed[curs.level][curs.mi] = ir2codeinst(ir, codeinst, ci)
    oc = OpaqueClosure(ir; isva=curs.mi.def.isva)
    return oc
end
