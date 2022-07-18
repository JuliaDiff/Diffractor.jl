using OffsetArrays
using Cthulhu

struct ADCursor <: Cthulhu.AbstractCursor
    level::Int
    mi::MethodInstance
end
Cthulhu.get_mi(c::ADCursor) = c.mi

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

#=
Compiler3.has_codeinfo(graph::ADGraph, cursor::ADCursor) =
    lastindex(graph.code) >= cursor.level && haskey(graph.code[cursor.level], cursor.mi)
function Compiler3.get_codeinstance(graph::ADGraph, cursor::ADCursor)
    return graph.code[cursor.level][cursor.mi]
end
=#

using Core.Compiler: AbstractInterpreter, NativeInterpreter, InferenceState,
    InferenceResult, CodeInstance, WorldRange

struct ADInterpreter <: AbstractInterpreter
    # This cache is stratified by AD nesting level. Depending on the
    # nesting level of the derivative, The AD primitives may behave
    # differently.
    # Level 0 == Straightline Code, no AD
    # Level 1 == Gradients
    # Level 2 == Seconds Derivatives
    # and so on
    opt::OffsetVector{Dict{MethodInstance, CodeInstance}}
    unopt::Union{OffsetVector{Dict{Union{MethodInstance, InferenceResult}, Cthulhu.InferredSource}}, Nothing}

    native_interpreter::NativeInterpreter
    current_level::Int
    msgs::Vector{Tuple{Int, MethodInstance, Int, String}}
end
change_level(interp::ADInterpreter, new_level::Int) = ADInterpreter(interp.opt, interp.unopt, interp.native_interpreter, new_level, interp.msgs)
raise_level(interp::ADInterpreter) = change_level(interp, interp.current_level + 1)
lower_level(interp::ADInterpreter) = change_level(interp, interp.current_level - 1)

Cthulhu.get_optimized_code(interp::ADInterpreter, curs::ADCursor) = interp.opt[curs.level][curs.mi]
Cthulhu.AbstractCursor(interp::ADInterpreter, mi::MethodInstance) = ADCursor(0, mi)

# This is a lie, but let's clean this up later
Cthulhu.can_descend(interp::ADInterpreter, mi::MethodInstance, optimize::Bool) = true

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
        codeinst = interp.opt[curs.level][curs.mi]
        rt = Cthulhu.codeinst_rt(codeinst)
        opt = codeinst.inferred
        if opt !== nothing
            opt = opt::Cthulhu.OptimizedSource
            src = Core.Compiler.copy(opt.ir)
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

# TODO: Something is going very wrong here
using Core.Compiler: Effects, OptimizationState
function Cthulhu.get_effects(interp::ADInterpreter, mi::MethodInstance, opt::Bool)
    if haskey(interp.unopt[0], mi)
        return interp.unopt[0][mi].effects
    else
        return Effects()
    end
end

function Core.Compiler.is_same_frame(interp::ADInterpreter, linfo::MethodInstance, frame::InferenceState)
    linfo === frame.linfo || return false
    return interp.current_level === frame.interp.current_level
end

# Special handling for Recursion
struct RecurseCallInfo <: Cthulhu.CallInfo
    vmi::Cthulhu.CallInfo # callinfo to be descended
end
Cthulhu.get_mi((; vmi)::RecurseCallInfo) = Cthulhu.get_mi(vmi)
Cthulhu.get_rt((; vmi)::RecurseCallInfo) = Cthulhu.get_rt(vmi)
Cthulhu.get_effects(::RecurseCallInfo) = Effects()

function Cthulhu.print_callsite_info(limiter::IO, info::RecurseCallInfo)
    print(limiter, " = < Diffractor recurse > ")
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
    print(limiter, " = < rrule > ")
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
    print(limiter, " = < cc > ")
end

# Navigation
function Cthulhu.navigate(curs::ADCursor, callsite)
    if isa(callsite.info, RecurseCallInfo)
        return ADCursor(curs.level + 1, Cthulhu.get_mi(callsite))
    elseif isa(callsite.info, RRuleCallInfo)
        return ADCursor(curs.level - 1, Cthulhu.get_mi(callsite))
    end
    return ADCursor(curs.level, Cthulhu.get_mi(callsite))
end

function Cthulhu.process_info(interp::ADInterpreter, @nospecialize(info), argtypes::Cthulhu.ArgTypes, @nospecialize(rt), optimize::Bool)
    if isa(info, RecurseInfo)
        newargtypes = argtypes[2:end]
        callinfos = Cthulhu.process_info(interp, info.info, newargtypes, Cthulhu.unwrapType(widenconst(rt)), optimize)
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
        callinfos = Cthulhu.process_info(interp, info.info, newargtypes, Cthulhu.unwrapType(widenconst(rt)), optimize)
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
    return invoke(Cthulhu.process_info, Tuple{Any, Any, Cthulhu.ArgTypes, Any, Bool},
        interp, info, argtypes, rt, optimize)
end

ADInterpreter() = ADInterpreter(
    OffsetVector([Dict{MethodInstance, CodeInstance}(), Dict{MethodInstance, CodeInstance}()], 0:1),
    OffsetVector([Dict{MethodInstance, Cthulhu.InferredSource}(), Dict{MethodInstance, Cthulhu.InferredSource}()], 0:1),
    NativeInterpreter(),
    0,
    Vector{Tuple{Int, MethodInstance, Int, String}}()
)

ADInterpreter(fg::ADGraph, level) =
    ADInterpreter(fg.code, NativeInterpreter(), level, fg.msgs)

Core.Compiler.InferenceParams(ei::ADInterpreter) = InferenceParams(ei.native_interpreter)
Core.Compiler.OptimizationParams(ei::ADInterpreter) = OptimizationParams(ei.native_interpreter)
Core.Compiler.get_world_counter(ei::ADInterpreter) = get_world_counter(ei.native_interpreter)
Core.Compiler.get_inference_cache(ei::ADInterpreter) = get_inference_cache(ei.native_interpreter)

# No need to do any locking since we're not putting our results into the runtime cache
lock_mi_inference(ei::ADInterpreter, mi::MethodInstance) = nothing
unlock_mi_inference(ei::ADInterpreter, mi::MethodInstance) = nothing

struct CodeInfoView
    d::Dict{MethodInstance, Any}
end

function Core.Compiler.code_cache(ei::ADInterpreter)
    while ei.current_level > lastindex(ei.opt)
        push!(ei.opt, Dict{MethodInstance, Any}())``
    end
    ei.opt[ei.current_level]
end
Core.Compiler.may_optimize(ei::ADInterpreter) = true
Core.Compiler.may_compress(ei::ADInterpreter) = false
Core.Compiler.may_discard_trees(ei::ADInterpreter) = false
function Core.Compiler.get(view::CodeInfoView, mi::MethodInstance, default)
    r = get(view.d, mi, nothing)
    if r === nothing
        return default
    end
    if isa(r, OptimizationState)
        r = r.src
    end
    return r::CodeInfo
end

function Core.Compiler.add_remark!(ei::ADInterpreter, sv::InferenceState, msg)
    push!(ei.msgs, (ei.current_level, sv.linfo, sv.currpc, msg))
end

#=
function Core.Compiler.const_prop_heuristic(interp::AbstractInterpreter, method::Method, mi::MethodInstance)
    return true
end
=#

function Core.Compiler.finish(state::InferenceState, interp::ADInterpreter)
    res = @invoke Core.Compiler.finish(state::InferenceState, interp::AbstractInterpreter)
    key = Core.Compiler.any(state.result.overridden_by_const) ? state.result : state.linfo
    interp.unopt[interp.current_level][key] = Cthulhu.InferredSource(
        copy(state.src),
        copy(state.stmt_info),
        isdefined(Core.Compiler, :Effects) ? state.ipo_effects : nothing,
        state.result.result)
    return res
end

function Core.Compiler.transform_result_for_cache(interp::ADInterpreter,
    linfo::MethodInstance, valid_worlds::WorldRange, @nospecialize(inferred_result),
    ipo_effects::Core.Compiler.Effects)
    return Cthulhu.create_cthulhu_source(inferred_result, ipo_effects)
end

#=
function Core.Compiler.optimize(interp::ADInterpreter, opt::OptimizationState,
    params::OptimizationParams, caller::InferenceResult)

    # TODO: Enable some amount of inlining
    #@timeit "optimizer" ir = run_passes(opt.src, opt, caller)
    
    sv = opt
    ci = opt.src
    ir = Core.Compiler.convert_to_ircode(ci, sv)
    ir = Core.Compiler.slot2reg(ir, ci, sv)
    # TODO: Domsorting can produce an updated domtree - no need to recompute here
    ir = Core.Compiler.compact!(ir)
    return Core.Compiler.finish(interp, opt, params, ir, caller)
end
=#

function Core.Compiler.finish!(interp::ADInterpreter, caller::InferenceResult)
    effects = caller.ipo_effects
    caller.src = Cthulhu.create_cthulhu_source(caller.src, effects)
end

using Core: OpaqueClosure
function codegen(interp::ADInterpreter, curs::ADCursor, cache=Dict{ADCursor, OpaqueClosure}())
    ir = Core.Compiler.copy(Cthulhu.get_optimized_code(interp, curs).inferred.ir)
    @show curs.mi
    display(ir)
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
                @show inst
                mi′ = specialize_method(info.info.results.matches[1], preexisting=true)
                new_curs = ADCursor(curs.level + 1, mi′)
                if haskey(cache, new_curs)
                    oc = cache[new_curs]
                else
                    oc = codegen(interp, new_curs, cache)
                end
                rrule_mi = oc.source.specializations[1]
                rrule_rt = Any # TODO
                rrule_call = insert_node!(ir, i, NewInstruction(Expr(:invoke, rrule_mi, inst.args[2:end]...), rrule_rt))
                arg1 = insert_node!(ir, i, NewInstruction(Expr(:call, getfield, rrule_call, 1), getfield_tfunc(rrule_rt, Const(1))))
                arg2 = insert_node!(ir, i, NewInstruction(Expr(:call, getfield, rrule_call, 2), getfield_tfunc(rrule_rt, Const(2))))
                ir.stmts[i][:inst] = arg1
                duals[i] = arg2
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
    display(ir)
    return OpaqueClosure(ir)
end