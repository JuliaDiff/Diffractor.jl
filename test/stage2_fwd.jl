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

    Diffractor.reload() # XXX we should remove this

    self_minus(a) = myminus(a, a)
    let self_minus′′ = Diffractor.dontuse_nth_order_forward_stage2(Tuple{typeof(self_minus), Float64}, 2)
        @test isa(self_minus′′, Core.OpaqueClosure{Tuple{Float64}, Float64})
        @test self_minus′′(1.0) == 0.
    end
end
