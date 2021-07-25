using Diffractor
using Diffractor: var"'", ∂⃖
using ChainRules
using ChainRulesCore
using ChainRules: ZeroTangent, NoTangent
using Symbolics

using Test

# Unit tests
function tup2(f)
    a, b = ∂⃖{2}()(f, 1)
    c, d = b((2,))
    e, f = d(ZeroTangent(), 3)
    f((4,))
end

@test tup2(tuple) == (NoTangent(), 4)

my_tuple(args...) = args
ChainRules.rrule(::typeof(my_tuple), args...) = args, Δ->Core.tuple(NoTangent(), Δ...)
@test tup2(my_tuple) == (ZeroTangent(), 4)

# Check characteristic of exp rule
@variables ω α β γ δ ϵ ζ η
(x1, c1) = ∂⃖{3}()(exp, ω)
@test simplify(x1 == exp(ω)).val
((_, x2), c2) = c1(α)
@test simplify(x2 == α*exp(ω)).val
(x3, c3) = c2(ZeroTangent(), β)
@test simplify(x3 == β*exp(ω)).val
((_, x4), c4) = c3(γ)
@test simplify(x4 == exp(ω)*(γ + (α*β))).val
(x5, c5) = c4(ZeroTangent(), δ)
@test simplify(x5 == δ*exp(ω)).val
((_, x6), c6) = c5(ϵ)
@test simplify(x6 == ϵ*exp(ω) + α*δ*exp(ω)).val
(x7, c7) = c6(ZeroTangent(), ζ)
@test simplify(x7 == ζ*exp(ω) + β*δ*exp(ω)).val
(_, x8) = c7(η)
@test simplify(x8 == (η + (α*ζ) + (β*ϵ) + (δ*(γ + (α*β))))*exp(ω)).val

# Minimal 2-nd order forward smoke test
@test Diffractor.∂☆{2}()(Diffractor.ZeroBundle{2}(sin),
    Diffractor.TangentBundle{2}(1.0, (1.0, 1.0, 0.0)))[Diffractor.CanonicalTangentIndex(1)] == sin'(1.0)

function simple_control_flow(b, x)
    if b
        return sin(x)
    else
        return cos(x)
    end
end

function myprod(xs)
    s = 1
    for x in xs
      s *= x
    end
    return s
end

function times_three_while(x)
    z = x
    i = 3
    while i > 1
        z += x
        i -= 1
    end
    z
end

isa_control_flow(::Type{T}, x) where {T} = isa(x, T) ? x : T(x)

# Simple Reverse Mode tests
let var"'" = Diffractor.PrimeDerivativeBack
    # Integration tests
    @test @inferred(sin'(1.0)) == cos(1.0)
    @test @inferred(sin''(1.0)) == -sin(1.0)
    @test sin'''(1.0) == -cos(1.0)
    @test sin''''(1.0) == sin(1.0)
    @test sin'''''(1.0) == cos(1.0)
    @test sin''''''(1.0) == -sin(1.0)

    f_getfield(x) = getfield((x,), 1)
    @test f_getfield'(1) == 1
    @test f_getfield''(1) == 0
    @test f_getfield'''(1) == 0

    # Higher order mixed mode tests

    complicated_2sin(x) = (x = map(sin, Diffractor.xfill(x, 2)); x[1] + x[2])
    @test @inferred(complicated_2sin'(1.0)) == 2sin'(1.0)
    @test @inferred(complicated_2sin''(1.0)) == 2sin''(1.0)
    @test @inferred(complicated_2sin'''(1.0)) == 2sin'''(1.0)
    @test @inferred(complicated_2sin''''(1.0)) == 2sin''''(1.0)

    @test @inferred((x->simple_control_flow(true, x))'(1.0)) == sin'(1.0)
    @test @inferred((x->simple_control_flow(false, x))'(1.0)) == cos'(1.0)
    @test (x->sum(isa_control_flow(Matrix{Float64}, x)))'(Float32[1 2;]) == [1.0 1.0;]
    @test times_three_while'(1.0) == 3.0
end

# Simple Forward Mode tests
let var"'" = Diffractor.PrimeDerivativeFwd
    recursive_sin(x) = sin(x)
    ChainRulesCore.frule(∂, ::typeof(recursive_sin), x) = frule(∂, sin, x)

    # Integration tests
    @test recursive_sin'(1.0) == cos(1.0)
    @test recursive_sin''(1.0) == -sin(1.0)
    @test recursive_sin'''(1.0) == -cos(1.0)
    @test recursive_sin''''(1.0) == sin(1.0)
    @test recursive_sin'''''(1.0) == cos(1.0)
    @test recursive_sin''''''(1.0) == -sin(1.0)

    # Test the special rules for sin/cos/exp
    @test sin''''''(1.0) == -sin(1.0)
    @test cos''''''(1.0) == -cos(1.0)
    @test exp''''''(1.0) == exp(1.0)
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

# Regression tests
@test gradient(x -> sum(abs2, x .+ 1.0), zeros(3))[1] == [2.0, 2.0, 2.0]

const fwd = Diffractor.PrimeDerivativeFwd
const bwd = Diffractor.PrimeDerivativeFwd

function f_broadcast(a)
    l = a / 2.0 * [[0. 1. 1.]; [1. 0. 1.]; [1. 1. 0.]]
    return sum(l)
end
@test fwd(f_broadcast)(1.0) == bwd(f_broadcast)(1.0)

include("pinn.jl")
