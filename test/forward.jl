#module forward_tests
using Diffractor
using Diffractor: TaylorBundle, ZeroBundle, ∂☆
using ChainRules
using ChainRulesCore
using ChainRulesCore: ZeroTangent, NoTangent, frule_via_ad, rrule_via_ad
using LinearAlgebra
using Test



# Minimal 2-nd order forward smoke test
let var"'" = Diffractor.PrimeDerivativeFwd
    @test Diffractor.∂☆{2}()(ZeroBundle{2}(sin),
        Diffractor.ExplicitTangentBundle{2}(1.0, (1.0, 1.0, 0.0)))[Diffractor.CanonicalTangentIndex(1)] == sin'(1.0)
end

# Simple Forward Mode tests
let var"'" = Diffractor.PrimeDerivativeFwd
    recursive_sin(x) = sin(x)
    ChainRulesCore.frule(∂, ::typeof(recursive_sin), x) = frule(∂, sin, x)

    # Integration tests
    @test recursive_sin'(1.0) == cos(1.0)
    @test recursive_sin''(1.0) == -sin(1.0)
    
    @test_broken recursive_sin'''(1.0) == -cos(1.0)
    @test_broken recursive_sin''''(1.0) == sin(1.0)
    @test_broken recursive_sin'''''(1.0) == cos(1.0)
    @test_broken recursive_sin''''''(1.0) == -sin(1.0)

    # Test the special rules for sin/cos/exp
    @test sin''''''(1.0) == -sin(1.0)
    @test cos''''''(1.0) == -cos(1.0)
    @test exp''''''(1.0) == exp(1.0)
    @test (x->prod([x, 4]))'(3) == 4
end

# Some Basic Mixed Mode tests
function sin_twice_fwd(x)
    let var"'" = Diffractor.PrimeDerivativeFwd
            sin''(x)
    end
end
let var"'" = Diffractor.PrimeDerivativeFwd
    @test sin_twice_fwd'(1.0) == sin'''(1.0)
end


@testset "No partials" begin
    primal_calls = Ref(0)
    function foo(x, y)
        primal_calls[]+=1
        return x+y
    end
    
    frule_calls = Ref(0)
    function ChainRulesCore.frule((_, ẋ, ẏ), ::typeof(foo), x, y)
        frule_calls[]+=1
        return x+y, ẋ+ẏ
    end

    # Special case if there is no derivative information at all:
    @test ∂☆{1}()(ZeroBundle{1}(foo), ZeroBundle{1}(2.0), ZeroBundle{1}(3.0)) == ZeroBundle{1}(5.0)
    @test frule_calls[] == 0
    @test primal_calls[] == 1
end

@testset "indexing" begin
    # Test to make sure that `:boundscheck` and such are properly handled
    function foo(x)
        t = (x, x)
        return t[1] + 1
    end

    let var"'" = Diffractor.PrimeDerivativeFwd
        @test foo'(1.0) == 1.0
    end

    # Test that `@inbounds` is ignored by Diffractor
    function foo_errors(x)
        t = (x, x)
        @inbounds return t[3] + 1
    end
    let var"'" = Diffractor.PrimeDerivativeFwd
        @test_throws BoundsError foo_errors'(1.0) == 1.0
    end
end


@testset "map" begin
    @test ==(
        ∂☆{1}()(ZeroBundle{1}(xs->(map(x->2*x, xs))), TaylorBundle{1}([1.0, 2.0], ([10.0, 100.0],))),
        TaylorBundle{1}([2.0, 4.0], ([20.0, 200.0],))
    )


    # map over all closure, wrt the closed variable
    mulby(x) = y->x*y
    🐇 = ∂☆{1}()(
        ZeroBundle{1}(x->(map(mulby(x), [2.0, 4.0]))), 
        TaylorBundle{1}(2.0, (10.0,))
    )
    @test 🐇 == TaylorBundle{1}([4.0, 8.0], ([20.0, 40.0],))

end


@testset "structs" begin
    struct IDemo
        x::Float64
        y::Float64
    end

    function foo(a)
        obj = IDemo(2.0, a)
        return obj.x * obj.y
    end

    let var"'" = Diffractor.PrimeDerivativeFwd
        @test foo'(100.0) == 2.0
        @test foo''(100.0) == 0.0
    end
end

@testset "tuples" begin
    function foo(a)
        tup = (2.0, a)
        return first(tup) * tup[2]
    end

    let var"'" = Diffractor.PrimeDerivativeFwd
        @test foo'(100.0) == 2.0
        @test foo''(100.0) == 0.0
    end
end

@testset "vararg" begin
    function foo(a)
        tup = (2.0, a)
        return *(tup...)
    end

    let var"'" = Diffractor.PrimeDerivativeFwd
        @test foo'(100.0) == 2.0
        @test foo''(100.0) == 0.0
    end
end


@testset "taylor_compatible" begin
    taylor_compatible = Diffractor.taylor_compatible

    @test taylor_compatible(
        TaylorBundle{1}(10.0, (20.0,)),
        TaylorBundle{1}(20.0, (30.0,))
    )
    @test !taylor_compatible(
        TaylorBundle{1}(10.0, (20.0,)),
        TaylorBundle{1}(21.0, (30.0,))
    )
    @test taylor_compatible(
        TaylorBundle{2}(10.0, (20.0, 30.)),
        TaylorBundle{2}(20.0, (30.0, 40.))
    )
    @test !taylor_compatible(
        TaylorBundle{2}(10.0, (20.0, 30.0)),
        TaylorBundle{2}(20.0, (31.0, 40.0))
    )


    tuptan(args...) = Tangent{typeof(args)}(args...)
    @test taylor_compatible(
        TaylorBundle{1}((10.0, 20.0), (tuptan(20.0, 30.0),)),
    )
    @test taylor_compatible(
        TaylorBundle{2}((10.0, 20.0), (tuptan(20.0, 30.0),tuptan(30.0, 40.0))),
    )
    @test !taylor_compatible(
        TaylorBundle{1}((10.0, 20.0), (tuptan(21.0, 30.0),)),
    )
    @test !taylor_compatible(
        TaylorBundle{2}((10.0, 20.0), (tuptan(20.0, 31.0),tuptan(30.0, 40.0))),
    )
end

end
