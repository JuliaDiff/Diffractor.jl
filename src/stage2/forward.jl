using .CC: compact!

# Engineering entry point for the 2nd-order forward AD functionality. This is
# unlikely to be the actual interface. For now, it is used for testing.
function dontuse_nth_order_forward_stage2(tt::Type, order::Int=1)
    interp = ADInterpreter(; forward=true, backward=false)
    match = Base._which(tt)
    frame = Core.Compiler.typeinf_frame(interp, match.method, match.spec_types, match.sparams, #=run_optimizer=#true)

    ir = copy((interp.opt[0][frame.linfo].inferred).ir::IRCode)

    # Find all Return Nodes
    vals = Pair{SSAValue, Int}[]
    for i = 1:length(ir.stmts)
        if isa(ir[SSAValue(i)][:inst], ReturnNode)
            push!(vals, SSAValue(i)=>order)
        end
    end

    function visit_custom!(ir::IRCode, @nospecialize(stmt), order, recurse)
        if isa(stmt, ReturnNode)
            recurse(stmt.val)
            return true
        elseif isa(stmt, Argument)
            return true
        else
            return false
        end
    end

    function transform!(ir::IRCode, ssa::SSAValue, _)
        inst = ir[ssa]
        stmt = inst[:inst]
        if isa(stmt, ReturnNode)
            nr = insert_node!(ir, ssa, NewInstruction(Expr(:call, getindex, stmt.val, TaylorTangentIndex(order)), Any))
            inst[:inst] = ReturnNode(nr)
        else
            error()
        end
    end

    function transform!(ir::IRCode, arg::Argument, _)
        return insert_node!(ir, SSAValue(1), NewInstruction(Expr(:call, ∂xⁿ{order}(), arg), typeof(∂xⁿ{order}()(1.0))))
    end


    irsv = CC.IRInterpretationState(interp, ir, frame.linfo, CC.get_world_counter(interp), ir.argtypes[1:frame.linfo.def.nargs])
    ir = forward_diff!(ir, interp, frame.linfo, CC.get_world_counter(interp), vals; visit_custom!, transform!)

    ir = compact!(ir)
    return OpaqueClosure(ir)
end
