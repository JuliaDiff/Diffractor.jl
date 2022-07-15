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
function Cthulhu.get_effects(interp::ADInterpreter, mi::MethodInstance, opt::Bool)
    interp.unopt[0][mi].effects
end

function Core.Compiler.is_same_frame(interp::ADInterpreter, linfo::MethodInstance, frame::InferenceState)
    linfo === frame.linfo || return false
    return interp.current_level === frame.interp.current_level
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
Core.Compiler.may_optimize(ei::ADInterpreter) = false
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
    return Cthulhu.maybe_create_optsource(inferred_result, ipo_effects)
end

function Core.Compiler.finish!(interp::ADInterpreter, caller::InferenceResult)
    effects = caller.ipo_effects
    caller.src = Cthulhu.maybe_create_optsource(caller.src, effects)
    @show typeof(caller.src)
end
