module stage2_fwd
    using Diffractor, Test, ChainRulesCore
    using Diffractor: ZeroBundle, CompositeBundle, TaylorBundle
    
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
        dz = Diffractor.∂☆{1}()(ZeroBundle{1}(foo_dub), TaylorBundle{1}(10.0, (π,)))
        @test Diffractor.first_partial(dz) == Tangent{Foo}(;x=π, y=2π)
    end

    @testset "mix of vararg and positional args" begin
        cc(a, x::Vararg) = nothing
        Diffractor.∂☆{1}()(ZeroBundle{1}(cc), TaylorBundle{1}(10f0, (10.0,)), TaylorBundle{1}(10f0, (10.0,)))

        gg(a, xs...) = nothing
        Diffractor.∂☆{1}()(ZeroBundle{1}(gg), TaylorBundle{1}(10f0, (1.2,)), TaylorBundle{1}(20f0, (1.1,)))
    end

    

    @testsset "nontrivial nested" begin
        f(x) = 3x^2
        g(x) = Diffractor.∂☆{1}()(ZeroBundle{1}(f), TaylorBundle{1}(x, (1.0,)))
        # End goal:
        #Diffractor.∂☆{1}()(Diffractor.ZeroBundle{1}(g), Diffractor.TaylorBundle{1}(10f0, (1.0,)))

        # primal call is frule((ZeroTangent(), 1.0), f, 10f0)
        Diffractor.∂☆{1}()(
            Diffractor.ZeroBundle{1}(frule),
            CompositeBundle{1, Tuple{ChainRulesCore.ZeroTangent, Float64}}((ZeroBundle{1}(ZeroTangent()), ZeroBundle{1}(1.0))),
            ZeroBundle{1}(f),
            TaylorBundle{1}(10f0, (1.0,)),
        )

        
    end

CompositeBundle
    
end
