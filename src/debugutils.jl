using .CC: AbstractInterpreter, CodeInstance, MethodInstance, WorldView, NativeInterpreter
using InteractiveUtils

function infer_function(interp, tt)
    world = get_world_counter()

    # Find all methods that are applicable to these types
    mthds = _methods_by_ftype(tt, -1, world)
    if mthds === false || length(mthds) != 1
        error("Unable to find single applicable method for $tt")
    end

    mtypes, msp, m = mthds[1]

    # Grab the appropriate method instance for these types
    mi = CC.specialize_method(m, mtypes, msp)

    # Construct InferenceResult to hold the result,
    result = CC.InferenceResult(mi)

    # Create an InferenceState to begin inference, give it a world that is always newest
    frame = CC.InferenceState(result, #=cached=# true, interp)

    # Run type inference on this frame.  Because the interpreter is embedded
    # within this InferenceResult, we don't need to pass the interpreter in.
    CC.typeinf(interp, frame)

    # Give the result back
    return (mi, result)
end

struct ExtractingInterpreter <: CC.AbstractInterpreter
    code::Dict{MethodInstance, CodeInstance}
    native_interpreter::NativeInterpreter
    msgs::Vector{Tuple{MethodInstance, Int, String}}
    optimize::Bool
end

ExtractingInterpreter(;optimize=false) = ExtractingInterpreter(
    Dict{MethodInstance, Any}(),
    NativeInterpreter(),
    Vector{Tuple{MethodInstance, Int, String}}(),
    optimize
)

import .CC: InferenceParams, OptimizationParams, #=get_inference_world,=#
    get_inference_cache, code_cache,
    WorldView, lock_mi_inference, unlock_mi_inference, InferenceState
InferenceParams(ei::ExtractingInterpreter) = InferenceParams(ei.native_interpreter)
OptimizationParams(ei::ExtractingInterpreter) = OptimizationParams(ei.native_interpreter)
get_inference_world(ei::ExtractingInterpreter) = get_inference_world(ei.native_interpreter)
get_inference_cache(ei::ExtractingInterpreter) = get_inference_cache(ei.native_interpreter)

# No need to do any locking since we're not putting our results into the runtime cache
lock_mi_inference(ei::ExtractingInterpreter, mi::MethodInstance) = nothing
unlock_mi_inference(ei::ExtractingInterpreter, mi::MethodInstance) = nothing

code_cache(ei::ExtractingInterpreter) = ei.code
CC.get(a::WorldView{<:Dict}, b, c) = Base.get(a.cache,b,c)
CC.haskey(a::WorldView{<:Dict}, b) =
    CC.haskey(a.cache, b)
CC.may_optimize(ei::ExtractingInterpreter) = ei.optimize
CC.may_compress(ei::ExtractingInterpreter) = false
CC.may_discard_trees(ei::ExtractingInterpreter) = false

function CC.add_remark!(ei::ExtractingInterpreter, sv::InferenceState, msg)
    push!(ei.msgs, (sv.linfo, sv.currpc, msg))
end

macro code_typed_nocache(ex0...)
    esc(:(@InteractiveUtils.code_typed interp=$(ExtractingInterpreter)() $(ex0...)))
end

export @code_typed_nocache
