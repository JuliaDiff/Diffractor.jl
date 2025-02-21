
module forward_diff_no_inf
    using Core: SSAValue

    using Diffractor, Test
    const CC = Diffractor.CC

    ##################### Helpers:

    # this is needed as transform! is *always* called on Arguments regardless of what visit_custom says
    identity_transform!(ir, ssa::SSAValue, order, _) = ir[ssa]
    function identity_transform!(ir, arg::Core.Argument, order, _)
        return CC.insert_node!(ir, SSAValue(1), CC.NewInstruction(Expr(:call, Diffractor.zero_bundle{1}(), arg), Any))
    end


    function infer_ir!(ir)
        interp = CC.NativeInterpreter()
        mi = ccall(:jl_new_method_instance_uninit, Ref{Core.MethodInstance}, ());
        mi.specTypes = Tuple{map(CC.widenconst, ir.argtypes)...}
        mi.def = @__MODULE__

        for i in 1:length(ir.stmts)
            inst = ir[SSAValue(i)][:inst]
            if Meta.isexpr(inst, :code_coverage_effect)
                # delete these as CC._ir_abstract_constant_propagation doesn't work on them
                ir[SSAValue(i)][:inst] = nothing
                ir[SSAValue(i)][:type] = Nothing
            end
            # For testing purposes we are going to refine everything else
            ir[SSAValue(i)][:flag] |= CC.IR_FLAG_REFINED
        end

        info = @static if VERSION ≥ v"1.12.0-DEV.1293"
            CC.SpecInfo(#=nargs=#length(ir.argtypes), #=isva=#false, #=propagate_inbounds=#true, nothing)
        else
            CC.MethodInfo(#=propagate_inbounds=#true, nothing)
        end
        min_world = world = (interp).world
        max_world = Diffractor.get_world_counter()
        irsv = CC.IRInterpretationState(interp, info, ir, mi, ir.argtypes, world, min_world, max_world)
        (rt, nothrow) = CC.ir_abstract_constant_propagation(interp, irsv)
        return rt
    end

    function isfully_inferred(ir)
        for stmt in ir.stmts
            inst = stmt[:inst]
            if Meta.isexpr(inst, :call) || Meta.isexpr(inst, :invoke)
                typ = stmt[:type]
                !isa(typ, Type) && continue  # If not a Type then something even more informed like a Const
                if isabstracttype(typ) || typ <: Union || typ <: UnionAll
#                    @error "Not fully inferred" inst typ
                    return false
                end
            end
        end
        return true
    end

    function findfirst_ssa(predicate, ir)
        for ii in 1:length(ir.stmts)
            try
                inst = ir[SSAValue(ii)][:inst]
                if predicate(inst)
                    return SSAValue(ii)
                end
            catch
                # ignore errors so predicate can be simple
            end
        end
        return nothing
    end

    ############################### Actual tests:

    @testset "Constructors in forward_diff_no_inf!" begin
        struct Bar148
            v
        end
        foo_148(x) = Bar148(x)

        ir = first(only(Base.code_ircode(foo_148, Tuple{Float64})))
        Diffractor.forward_diff_no_inf!(ir, [SSAValue(1) => 1]; transform! = identity_transform!)
        ir2 = CC.compact!(ir)
        f = Core.OpaqueClosure(ir2; do_compile=false)
        @test f(1.0) == Bar148(1.0)  # This would error if we were not handling constructors (%new) right
    end

    @testset "nonconst globals in forward_diff_no_inf!" begin
        @eval global _coeff::Float64=24.5
        plus_a_global(x) = x + _coeff

        ir = first(only(Base.code_ircode(plus_a_global, Tuple{Float64})))
        Diffractor.forward_diff_no_inf!(ir, [SSAValue(1) => 1]; transform! = identity_transform!)
        ir2 = CC.compact!(ir)
        CC.verify_ir(ir2)  # This would error if we were not handling nonconst globals correctly
        # Assert that the reference to `Main._coeff` is properly typed
        stmt_idx = findfirst(stmt -> isa(stmt[:inst], GlobalRef), collect(ir2.stmts))
        stmt = ir2.stmts[stmt_idx]
        @test stmt[:inst].name == :_coeff
        @test stmt[:type] == Float64
        f = Core.OpaqueClosure(ir2; do_compile=false)
        @test f(3.5) == 28.0
    end

    @testset "runs of phi nodes" begin
        function phi_run(x::Float64)
            a = 2.0
            b = 2.0
            if (@noinline rand()) < 0  # this branch will never actually be taken
                a = -100.0
                b = 200.0
            end
            return x - a + b
        end

        input_ir = first(only(Base.code_ircode(phi_run, Tuple{Float64})))
        ir = copy(input_ir)
        #Workout where to diff to trigger error
        diff_ssa = SSAValue[]
        for idx in 1:length(ir.stmts)
            if ir.stmts[idx][:inst] isa Core.PhiNode
                push!(diff_ssa, SSAValue(idx))
            end
        end

        Diffractor.forward_diff_no_inf!(ir, diff_ssa .=> 1; transform! = identity_transform!)
        ir2 = CC.compact!(ir)
        CC.verify_ir(ir2)  # This would error if we were not handling nonconst phi nodes correctly (after https://github.com/JuliaLang/julia/pull/50158)
        f = Core.OpaqueClosure(ir2; do_compile=false)
        @test f(3.5) == 3.5  # this will segfault if we are not handling phi nodes correctly
    end

    #only test this on new enough julia versions as exactly what infers can be fussy, as is running inference manually
    VERSION >= v"1.12.0-DEV.283" && @testset "Eras mode: $eras_mode" for eras_mode in (false, true)
        foo(x, y) = x*x + y*y
        ir = first(only(Base.code_ircode(foo, Tuple{Any, Any})))
        mul1_ssa = findfirst_ssa(x->x.args[1].name==:*, ir)
        Diffractor.forward_diff_no_inf!(ir, [mul1_ssa] .=> 1; transform! = identity_transform!, eras_mode)
        ir = CC.compact!(ir)
        ir.argtypes[2:end] .= Float64
        ir = CC.compact!(ir)
        infer_ir!(ir)
        CC.verify_ir(ir)
        @test isfully_inferred(ir)  # passes with and without eras mode

        add_ssa = findfirst_ssa(x->x.args[1].name==:+, ir)
        Diffractor.forward_diff_no_inf!(ir, [add_ssa] .=> 1; transform! = identity_transform!, eras_mode)
        ir = CC.compact!(ir)
        infer_ir!(ir)
        CC.verify_ir(ir)
        if eras_mode
            @test isfully_inferred(ir)
        else
            # if this passes outside era mode then this test is wrong
            @assert !isfully_inferred(ir)
        end
    end
end  # module

