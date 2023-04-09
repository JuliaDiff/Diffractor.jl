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
