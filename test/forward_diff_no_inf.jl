
module forward_diff_no_inf
    using Diffractor, Test
    # this is needed as transform! is *always* called on Arguments regardless of what visit_custom says
    identity_transform!(ir, ssa::Core.SSAValue, order) = ir[ssa]
    function identity_transform!(ir, arg::Core.Argument, order)
        return Core.Compiler.insert_node!(ir, Core.SSAValue(1), Core.Compiler.NewInstruction(Expr(:call, Diffractor.ZeroBundle{1}, arg), Any))
    end
    
    @testset "Constructors in forward_diff_no_inf!" begin
        struct Bar148
            v
        end
        foo_148(x) = Bar148(x)

        ir = first(only(Base.code_ircode(foo_148, Tuple{Float64})))
        Diffractor.forward_diff_no_inf!(ir, [Core.SSAValue(1) => 1]; transform! = identity_transform!)
        ir2 = Core.Compiler.compact!(ir)
        f = Core.OpaqueClosure(ir2; do_compile=false)
        @test f(1.0) == Bar148(1.0)  # This would error if we were not handling constructors (%new) right
    end

    @testset "nonconst globals in forward_diff_no_inf!" begin
        @eval global _coeff::Float64=24.5
        plus_a_global(x) = x + _coeff

        ir = first(only(Base.code_ircode(plus_a_global, Tuple{Float64})))
        Diffractor.forward_diff_no_inf!(ir, [Core.SSAValue(1) => 1]; transform! = identity_transform!)
        ir2 = Core.Compiler.compact!(ir)
        Core.Compiler.verify_ir(ir2)  # This would error if we were not handling nonconst globals correctly
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
        diff_ssa = Core.SSAValue[]
        for idx in 1:length(ir.stmts)
            if ir.stmts[idx][:inst] isa Core.PhiNode
                push!(diff_ssa, Core.SSAValue(idx))
            end
        end
    
        Diffractor.forward_diff_no_inf!(ir, diff_ssa .=> 1; transform! = identity_transform!)
        ir2 = Core.Compiler.compact!(ir)
        Core.Compiler.verify_ir(ir2)  # This would error if we were not handling nonconst phi nodes correctly (after https://github.com/JuliaLang/julia/pull/50158)
        f = Core.OpaqueClosure(ir2; do_compile=false)
        @test f(3.5) == 3.5  # this will segfault if we are not handling phi nodes correctly
    end    
end
