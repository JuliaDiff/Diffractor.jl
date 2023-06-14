module stage2_fwd
    using Diffractor, Test, ChainRulesCore

    mysin(x) = sin(x)
    let sin′ = Diffractor.dontuse_nth_order_forward_stage2(Tuple{typeof(mysin), Float64})
        @test sin′(1.0) == cos(1.0)
    end
    let sin′′ = Diffractor.dontuse_nth_order_forward_stage2(Tuple{typeof(mysin), Float64}, 2)
        @test isa(sin′′, Core.OpaqueClosure{Tuple{Float64}, Float64})
        @test sin′′(1.0) == -sin(1.0)
    end

    myminus(a, b) = a - b
    ChainRulesCore.@scalar_rule myminus(x, y) (true, -1)

    self_minus(a) = myminus(a, a)
    let self_minus′′ = Diffractor.dontuse_nth_order_forward_stage2(Tuple{typeof(self_minus), Float64}, 2)
        @test isa(self_minus′′, Core.OpaqueClosure{Tuple{Float64}, Float64})
        @test self_minus′′(1.0) == 0.
    end

    @testset "structs" begin
        struct Foo
            x
            y
        end
        foo_dub(x) = Foo(x, 2x)
        dz = Diffractor.∂☆{1}()(Diffractor.ZeroBundle{1}(foo_dub), Diffractor.TaylorBundle{1}(10.0, (π,)))
        @test Diffractor.first_partial(dz) == Tangent{Foo}(;x=π, y=2π)
    end

    @testset "mix of vararg and positional args" begin
        cc(a, x::Vararg) = nothing
        Diffractor.∂☆{1}()(Diffractor.ZeroBundle{1}(cc), Diffractor.TaylorBundle{1}(10f0, (10.0,)), Diffractor.TaylorBundle{1}(10f0, (10.0,)))

        gg(a, xs...) = nothing
        Diffractor.∂☆{1}()(Diffractor.ZeroBundle{1}(gg), Diffractor.TaylorBundle{1}(10f0, (1.2,)), Diffractor.TaylorBundle{1}(20f0, (1.1,)))
    end


    @testset "nontrivial nested" begin
        f(x) = 3x^2
        g(x) = Diffractor.∂☆{1}()(Diffractor.ZeroBundle{1}(f), Diffractor.TaylorBundle{1}(x, (1.0,)))
        Diffractor.∂☆{1}()(Diffractor.ZeroBundle{1}(g), Diffractor.TaylorBundle{1}(10f0, (1.0,)))
    end
end


module forward_diff_no_inf  # todo: move this to a seperate file
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
        f = Core.OpaqueClosure(ir2; do_compile=false)
        @test f(3.5) == 28.0
    end

    @testset "runs of phi nodes" begin
        function phi_run(x::Float64)
            a = 2.0
            b = 2.0
            c = 0.0
            if (@noinline rand()) < 0  # this branch will never actually be taken
                a = -100.0
                b = 200.0
                c = 300.0
            end
            return x - a + b + c
        end
    
        input_ir = first(only(Base.code_ircode(phi_run, Tuple{Float64})))
        ir = copy(input_ir)
        #Workout where to diff to trigger error
        diff_ssa = Core.SSAValue[]
        for idx in 1:length(ir.stmts)
            if ir.stmts[idx][:inst] isa Core.PhiNode
                push!(diff_ssa, Core.SSAValue(idx))
                break
            end
        end
    
        Diffractor.forward_diff_no_inf!(ir, diff_ssa .=> 1; transform! = identity_transform!)
        ir2 = Core.Compiler.compact!(ir)
        Core.Compiler.verify_ir(ir2)  # This would error if we were not handling nonconst phi nodes correctly (after https://github.com/JuliaLang/julia/pull/50158)
        f = Core.OpaqueClosure(ir2; do_compile=false)
        @test f(3.5) == 3.5  # this will segfault if we are not handling phi nodes correctly
    end    
end
