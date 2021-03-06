using OffsetArrays

using Compiler3
import Compiler3: entrypoint, mi_at_cursor, FunctionGraph

struct ADCursor
    level::Int
    mi::MethodInstance
end
mi_at_cursor(c::ADCursor) = c.mi

#=
Cthulhu.get_cursor(c::ADCursor, callinfo::Cthulhu.PullbackCallInfo) = ADCursor(c.level+1, Cthulhu.get_mi(callinfo.mi))
Cthulhu.get_cursor(c::ADCursor, cs::Cthulhu.Callsite) = Cthulhu.get_cursor(c, cs.info)
Cthulhu.get_cursor(c::ADCursor, callinfo) = ADCursor(c.level, Cthulhu.get_mi(callinfo))
=#

struct ADGraph <: FunctionGraph
    code::OffsetVector{Dict{MethodInstance, Any}}
    msgs::Vector{Tuple{Int, MethodInstance, Int, String}}
    entry_mi::MethodInstance
end
entrypoint(graph::ADGraph) = ADCursor(0, graph.entry_mi)
Compiler3.has_codeinfo(graph::ADGraph, cursor::ADCursor) =
    lastindex(graph.code) >= cursor.level && haskey(graph.code[cursor.level], cursor.mi)
function Compiler3.get_codeinstance(graph::ADGraph, cursor::ADCursor)
    return graph.code[cursor.level][cursor.mi]
end

struct ADInterpreter <: AbstractInterpreter
    # This cache is stratified by AD nesting level. Depending on the
    # nesting level of the derivative, The AD primitives may behave
    # differently.
    # Level 0 == Straightline Code, no AD
    # Level 1 == Gradients
    # Level 2 == Seconds Derivatives
    # and so on
    code::OffsetVector{Dict{MethodInstance, Any}}
    native_interpreter::NativeInterpreter
    current_level::Int
    msgs::Vector{Tuple{Int, MethodInstance, Int, String}}
end
change_level(interp::ADInterpreter, new_level::Int) = ADInterpreter(interp.code, interp.native_interpreter, new_level, interp.msgs)
raise_level(interp::ADInterpreter) = change_level(interp, interp.current_level + 1)
lower_level(interp::ADInterpreter) = change_level(interp, interp.current_level - 1)

function Core.Compiler.is_same_frame(interp::ADInterpreter, linfo::MethodInstance, frame::InferenceState)
    linfo === frame.linfo || return false
    return interp.current_level === frame.interp.current_level
end

ADInterpreter() = ADInterpreter(
    OffsetVector([Dict{MethodInstance, Any}(), Dict{MethodInstance, Any}()], 0:1),
    NativeInterpreter(),
    0,
    Vector{Tuple{Int, MethodInstance, Int, String}}()
)

ADInterpreter(fg::ADGraph, level) =
    ADInterpreter(fg.code, NativeInterpreter(), level, fg.msgs)

InferenceParams(ei::ADInterpreter) = InferenceParams(ei.native_interpreter)
OptimizationParams(ei::ADInterpreter) = OptimizationParams(ei.native_interpreter)
get_world_counter(ei::ADInterpreter) = get_world_counter(ei.native_interpreter)
get_inference_cache(ei::ADInterpreter) = get_inference_cache(ei.native_interpreter)

# No need to do any locking since we're not putting our results into the runtime cache
lock_mi_inference(ei::ADInterpreter, mi::MethodInstance) = nothing
unlock_mi_inference(ei::ADInterpreter, mi::MethodInstance) = nothing

struct CodeInfoView
    d::Dict{MethodInstance, Any}
end

function code_cache(ei::ADInterpreter)
    while ei.current_level > lastindex(ei.code)
        push!(ei.code, Dict{MethodInstance, Any}())
    end
    ei.code[ei.current_level]
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

function Core.Compiler.const_prop_heuristic(interp::AbstractInterpreter, method::Method, mi::MethodInstance)
    return true
end

function Core.Compiler.transform_result_for_cache(ei::ADInterpreter, linfo::MethodInstance, @nospecialize(result::Any))
    if isa(result, OptimizationState)
        ci = result.src
        nargs = result.nargs - 1
        ir = Core.Compiler.convert_to_ircode(ci, Core.Compiler.copy_exprargs(ci.code), false, nargs, result)
        ir = Core.Compiler.slot2reg(ir, ci, nargs, result)
        ir = compact!(ir)
        resize!(ir.argtypes, result.nargs)
        result = ir
    end
    return result
end
