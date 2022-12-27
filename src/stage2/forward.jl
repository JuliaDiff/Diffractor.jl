using .CC: compact!

# Engineering entry point for the 2nd-order forward AD functionality. This is
# unlikely to be the actual interface. For now, it is used for testing.
function dontuse_nth_order_forward_stage2(tt::Type)
    interp = ADInterpreter(; forward=true, backward=false)
    match = Base._which(tt)
    frame = Core.Compiler.typeinf_frame(interp, match.method, match.spec_types, match.sparams, #=run_optimizer=#true)

    ir = copy((interp.opt[0][frame.linfo].inferred).ir::IRCode)

    # Find all Return Nodes
    vals = SSAValue[]
    for i = 1:length(ir.stmts)
        if isa(ir[SSAValue(i)][:inst], ReturnNode)
            push!(vals, SSAValue(i))
        end
    end

    function custom_diff!(ir, ssa, stmt, recurse)
        if isa(stmt, ReturnNode)
            r = recurse(stmt.val)
            ir[ssa][:inst] = ReturnNode(r)
            return ssa
        elseif isa(stmt, Argument)
            return 1.0
        end
        return nothing
    end

    irsv = CC.IRInterpretationState(interp, ir, frame.linfo, CC.get_world_counter(interp), ir.argtypes[1:frame.linfo.def.nargs])
    forward_diff!(ir, interp, irsv, vals; custom_diff!)

    ir = compact!(ir)
    return OpaqueClosure(ir)
end
