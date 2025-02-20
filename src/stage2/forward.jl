using .CC: compact!

function is_known_invoke_or_call(@nospecialize(x), @nospecialize(func), ir::Union{IRCode,IncrementalCompact})
    return is_known_invoke(x, func, ir) || CC.is_known_call(x, func, ir)
end

function is_known_invoke(@nospecialize(x), @nospecialize(func), ir::Union{IRCode,IncrementalCompact})
    isexpr(x, :invoke) || return false
    ft = argextype(x.args[2], ir)
    return singleton_type(ft) === func
end

@noinline function dont_use_ddt_intrinsic(x::Float64)
    if Base.inferencebarrier(true)
        error("Intrinsic not transformed")
    end
    return Base.inferencebarrier(0.0)::Float64
end

# Engineering entry point for the 2nd-order forward AD functionality. This is
# unlikely to be the actual interface. For now, it is used for testing.
function dontuse_nth_order_forward_stage2(tt::Type, order::Int=1; eras_mode = false)
    interp = ADInterpreter(; forward=true, backward=false)
    mi = @ccall jl_method_lookup_by_tt(tt::Any, Base.tls_world_age()::Csize_t, #= method table =# nothing::Any)::Ref{MethodInstance}
    ci = CC.typeinf_ext_toplevel(interp, mi, CC.SOURCE_MODE_ABI)

    src = CC.copy(interp.unopt[0][mi].src)
    ir = CC.copy((@atomic :monotonic ci.inferred).ir::IRCode)

    # Find all Return Nodes
    vals = Pair{SSAValue, Int}[]
    for i = 1:length(ir.stmts)
        if isa(ir[SSAValue(i)][:inst], ReturnNode)
            push!(vals, SSAValue(i)=>order)
        end
    end

    function visit_custom!(ir::IRCode, ssa::Union{SSAValue,Argument}, order, recurse)
        if isa(ssa, Argument)
            return true
        end

        stmt = ir[ssa][:inst]
        if isa(stmt, ReturnNode)
            recurse(stmt.val)
            return true
        elseif is_known_invoke_or_call(stmt, dont_use_ddt_intrinsic, ir)
            recurse(stmt.args[end], order+1)
            return true
        else
            return false
        end
    end

    function transform!(ir::IRCode, ssa::SSAValue, _, maparg)
        inst = ir[ssa]
        stmt = inst[:inst]
        if isa(stmt, ReturnNode)
            if order == 0
                return
            end
            nr = insert_node!(ir, ssa, NewInstruction(Expr(:call, getindex, stmt.val, TaylorTangentIndex(order))))
            inst[:inst] = ReturnNode(nr)
        elseif is_known_invoke_or_call(stmt, dont_use_ddt_intrinsic, ir)
            arg = maparg(stmt.args[end], ssa, order+1)
            if order > 0
                replace_call!(ir, ssa, Expr(:call, error, "Only order 0 implemented here"))
            else
                replace_call!(ir, ssa, Expr(:call, getindex, arg, TaylorTangentIndex(1)))
            end
        else
            error()
        end
    end

    function transform!(ir::IRCode, arg::Argument, order, _)
        if order == 0
            return arg
        else
            return insert_node!(ir, SSAValue(1), NewInstruction(Expr(:call, ∂xⁿ{order}(), arg), typeof(∂xⁿ{order}()(1.0))))
        end
    end

    ir = forward_diff!(interp, ir, src, mi, vals; visit_custom!, transform!, eras_mode)

    return OpaqueClosure(ir)
end
