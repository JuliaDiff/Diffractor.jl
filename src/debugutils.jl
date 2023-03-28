using Core.Compiler: AbstractInterpreter, CodeInstance, MethodInstance, WorldView, NativeInterpreter
using InteractiveUtils

function infer_function(interp, tt)
    world = Core.Compiler.get_world_counter()

    # Find all methods that are applicable to these types
    mthds = _methods_by_ftype(tt, -1, world)
    if mthds === false || length(mthds) != 1
        error("Unable to find single applicable method for $tt")
    end

    mtypes, msp, m = mthds[1]

    # Grab the appropriate method instance for these types
    mi = Core.Compiler.specialize_method(m, mtypes, msp)

    # Construct InferenceResult to hold the result,
    result = Core.Compiler.InferenceResult(mi)

    # Create an InferenceState to begin inference, give it a world that is always newest
    frame = Core.Compiler.InferenceState(result, #=cached=# true, interp)

    # Run type inference on this frame.  Because the interpreter is embedded
    # within this InferenceResult, we don't need to pass the interpreter in.
    Core.Compiler.typeinf(interp, frame)

    # Give the result back
    return (mi, result)
end

struct ExtractingInterpreter <: Core.Compiler.AbstractInterpreter
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

import Core.Compiler: InferenceParams, OptimizationParams, get_world_counter,
    get_inference_cache, code_cache,
    WorldView, lock_mi_inference, unlock_mi_inference, InferenceState
InferenceParams(ei::ExtractingInterpreter) = InferenceParams(ei.native_interpreter)
OptimizationParams(ei::ExtractingInterpreter) = OptimizationParams(ei.native_interpreter)
get_world_counter(ei::ExtractingInterpreter) = get_world_counter(ei.native_interpreter)
get_inference_cache(ei::ExtractingInterpreter) = get_inference_cache(ei.native_interpreter)

# No need to do any locking since we're not putting our results into the runtime cache
lock_mi_inference(ei::ExtractingInterpreter, mi::MethodInstance) = nothing
unlock_mi_inference(ei::ExtractingInterpreter, mi::MethodInstance) = nothing

code_cache(ei::ExtractingInterpreter) = ei.code
Core.Compiler.get(a::Dict, b, c) = Base.get(a,b,c)
Core.Compiler.get(a::WorldView{<:Dict}, b, c) = Base.get(a.cache,b,c)
Core.Compiler.haskey(a::Dict, b) = Base.haskey(a, b)
Core.Compiler.haskey(a::WorldView{<:Dict}, b) =
    Core.Compiler.haskey(a.cache, b)
Core.Compiler.setindex!(a::Dict, b, c) = setindex!(a, b, c)
Core.Compiler.may_optimize(ei::ExtractingInterpreter) = ei.optimize
Core.Compiler.may_compress(ei::ExtractingInterpreter) = false
Core.Compiler.may_discard_trees(ei::ExtractingInterpreter) = false

function Core.Compiler.add_remark!(ei::ExtractingInterpreter, sv::InferenceState, msg)
    @show msg
    push!(ei.msgs, (sv.linfo, sv.currpc, msg))
end

macro code_typed_nocache(ex0...)
    esc(:(@InteractiveUtils.code_typed interp=$(ExtractingInterpreter)() $(ex0...)))
end

export @code_typed_nocache
