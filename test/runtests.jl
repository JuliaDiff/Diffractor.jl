using Diffractor
using Diffractor: var"'", ∂⃖, DiffractorRuleConfig
using ChainRules
using ChainRulesCore
using ChainRulesCore: ZeroTangent, NoTangent, frule_via_ad, rrule_via_ad
using Symbolics
using LinearAlgebra

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
@test isequal(simplify(x1), simplify(exp(ω)))
((_, x2), c2) = c1(α)
@test isequal(simplify(x2), simplify(α*exp(ω)))
(x3, c3) = c2(ZeroTangent(), β)
@test isequal(simplify(x3), simplify(β*exp(ω)))
((_, x4), c4) = c3(γ)
@test isequal(simplify(x4), simplify(exp(ω)*(γ + (α*β))))
(x5, c5) = c4(ZeroTangent(), δ)
@test isequal(simplify(x5), simplify(δ*exp(ω)))
((_, x6), c6) = c5(ϵ)
@test isequal(simplify(x6), simplify(ϵ*exp(ω) + α*δ*exp(ω)))
(x7, c7) = c6(ZeroTangent(), ζ)
@test isequal(simplify(x7), simplify(ζ*exp(ω) + β*δ*exp(ω)))
(_, x8) = c7(η)
@test isequal(simplify(x8), simplify((η + (α*ζ) + (β*ϵ) + (δ*(γ + (α*β))))*exp(ω)))

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

function mypow(x, n)
    r = one(x)
    while n > 0
        n -= 1
        r *= x
    end
    return r
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

    # Control flow cases
    @test @inferred((x->simple_control_flow(true, x))'(1.0)) == sin'(1.0)
    @test @inferred((x->simple_control_flow(false, x))'(1.0)) == cos'(1.0)
    @test_broken (x->sum(isa_control_flow(Matrix{Float64}, x)))'(Float32[1 2;]) == [1.0 1.0;]
    @test times_three_while'(1.0) == 3.0

    pow5p(x) = (x->mypow(x, 5))'(x)
    @test pow5p(1.0) == 5.0
end

# Simple Forward Mode tests
let var"'" = Diffractor.PrimeDerivativeFwd
    recursive_sin(x) = sin(x)
    ChainRulesCore.frule(∂, ::typeof(recursive_sin), x) = frule(∂, sin, x)

    # Integration tests
    @test recursive_sin'(1.0) == cos(1.0)
    @test recursive_sin''(1.0) == -sin(1.0)
    # Error: ArgumentError: Tangent for the primal Tangent{Tuple{Float64, Float64}, Tuple{Float64, Float64}}
    # should be backed by a NamedTuple type, not by Tuple{Tangent{Tuple{Float64, Float64}, Tuple{Float64, Float64}}}.
    @test_broken recursive_sin'''(1.0) == -cos(1.0)
    @test_broken recursive_sin''''(1.0) == sin(1.0)
    @test_broken recursive_sin'''''(1.0) == cos(1.0)
    @test_broken recursive_sin''''''(1.0) == -sin(1.0)

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
const bwd = Diffractor.PrimeDerivativeBack

function f_broadcast(a)
    l = a / 2.0 * [[0. 1. 1.]; [1. 0. 1.]; [1. 1. 0.]]
    return sum(l)
end
@test fwd(f_broadcast)(1.0) == bwd(f_broadcast)(1.0)

# Make sure that there's no infinite recursion in kwarg calls
g_kw(;x=1.0) = sin(x)
f_kw(x) = g_kw(;x)
@test bwd(f_kw)(1.0) == bwd(sin)(1.0)

function f_crit_edge(a, b, c, x)
    # A function with two critical edges. This used to trigger an issue where
    # Diffractor would fail to insert edges for the second split critical edge.
    y = 1x
    if a && b
        y = 2x
    end
    if b && c
        y = 3x
    end

    if c
        y = 4y
    end

    return y
end
@test bwd(x->f_crit_edge(false, false, false, x))(1.0) == 1.0
@test bwd(x->f_crit_edge(true, true, false, x))(1.0) == 2.0
@test bwd(x->f_crit_edge(false, true, true, x))(1.0) == 12.0
@test bwd(x->f_crit_edge(false, false, true, x))(1.0) == 4.0

# Issue #27 - Mixup in lifting of getfield
let var"'" = bwd
    @test (x->x^5)''(1.0) == 20.
    @test (x->(x*x)*(x*x)*x)'''(1.0) == 60.
    # Higher order control flow not yet supported (https://github.com/JuliaDiff/Diffractor.jl/issues/24)
    @test_broken (x->x^5)'''(1.0) == 60.
end

# Issue #38 - Splatting arrays
@test gradient(x -> max(x...), (1,2,3))[1] == (0.0, 0.0, 1.0)
@test gradient(x -> max(x...), [1,2,3])[1] == [0.0, 0.0, 1.0]

# Issue #40 - Symbol type parameters not properly quoted
@test Diffractor.∂⃖recurse{1}()(Val{:transformations})[1] === Val{:transformations}()

# PR #43
loss(res, z, w) = sum(res.U * Diagonal(res.S) * res.V) + sum(res.S .* w)
x43 = rand(10, 10)
@test Diffractor.gradient(x->loss(svd(x), x[:,1], x[:,2]), x43) isa Tuple{Matrix{Float64}}

# PR # 45 - Calling back into AD from ChainRules
y45, back45 = rrule_via_ad(DiffractorRuleConfig(), x -> log(exp(x)), 2)
@test y45 ≈ 2.0
@test back45(1) == (ZeroTangent(), 1.0)

z45, delta45 = frule_via_ad(DiffractorRuleConfig(), (0,1), x -> log(exp(x)), 2)
@test z45 ≈ 2.0
@test delta45 ≈ 1.0

# Higher order control flow not yet supported (https://github.com/JuliaDiff/Diffractor.jl/issues/24)
#include("pinn.jl")
