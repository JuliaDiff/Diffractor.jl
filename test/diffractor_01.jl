# This file has tests written specifically for Diffractor v0.1,
# which were in runtests.jl before PR 73 moved them all.
# (This commit has all changes to 27 Dec 2022.)

using Test

using Diffractor
using Diffractor: ∂⃖, DiffractorRuleConfig

using ChainRules
using ChainRulesCore
using ChainRulesCore: ZeroTangent, NoTangent, frule_via_ad, rrule_via_ad
using Symbolics

using LinearAlgebra

# Loading Diffractor: var"'" globally will break many tests above, which use it for adjoint.
const fwd = Diffractor.PrimeDerivativeFwd
const bwd = Diffractor.PrimeDerivativeBack

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
let var"'" = Diffractor.PrimeDerivativeFwd
    @test Diffractor.∂☆{2}()(Diffractor.ZeroBundle{2}(sin),
        Diffractor.ExplicitTangentBundle{2}(1.0, (1.0, 1.0, 0.0)))[Diffractor.CanonicalTangentIndex(1)] == sin'(1.0)
end

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
    @test @inferred(complicated_2sin''(1.0)) == 2sin''(1.0)  broken=true
    @test @inferred(complicated_2sin'''(1.0)) == 2sin'''(1.0)  broken=true
    @test @inferred(complicated_2sin''''(1.0)) == 2sin''''(1.0)  broken=true

    # Control flow cases
    @test @inferred((x->simple_control_flow(true, x))'(1.0)) == sin'(1.0)
    @test @inferred((x->simple_control_flow(false, x))'(1.0)) == cos'(1.0)
    @test (x->sum(isa_control_flow(Matrix{Float64}, x)))'(Float32[1 2;]) == [1.0 1.0;]
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
    @test_broken sin_twice_fwd'(1.0) == sin'''(1.0)
end

# Regression tests
@test gradient(x -> sum(abs2, x .+ 1.0), zeros(3))[1] == [2.0, 2.0, 2.0]

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
@test bwd(bwd(x->5))(1.0) == ZeroTangent()
@test fwd(fwd(x->5))(1.0) == ZeroTangent()

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
r45 = rrule_via_ad(DiffractorRuleConfig(), x -> log(exp(x)), 2)
@test r45 isa Tuple
y45, back45 = r45
@test y45 ≈ 2.0
@test back45(1) == (ZeroTangent(), 1.0)

z45, delta45 = frule_via_ad(DiffractorRuleConfig(), (0,1), x -> log(exp(x)), 2)
@test z45 ≈ 2.0
@test delta45 ≈ 1.0

# PR #82 - getindex on non-numeric arrays
@test gradient(ls -> ls[1](1.), [Base.Fix1(*, 1.)])[1][1] isa Tangent{<:Base.Fix1}

@testset "broadcast" begin
    @test gradient(x -> sum(x ./ x), [1,2,3]) == ([0,0,0],)  # derivatives_given_output
    @test gradient(x -> sum(sqrt.(atan.(x, transpose(x)))), [1,2,3])[1] ≈ [0.2338, -0.0177, -0.0661] atol=1e-3
    @test gradient(x -> sum(exp.(log.(x))), [1,2,3]) == ([1,1,1],)

    @test gradient(x -> sum((exp∘log).(x)), [1,2,3]) == ([1,1,1],)  # frule_via_ad
    exp_log(x) = exp(log(x))
    @test gradient(x -> sum(exp_log.(x)), [1,2,3]) == ([1,1,1],) 
    @test gradient((x,y) -> sum(x ./ y), [1 2; 3 4], [1,2]) == ([1 1; 0.5 0.5], [-3, -1.75])
    @test gradient((x,y) -> sum(x ./ y), [1 2; 3 4], 5) == ([0.2 0.2; 0.2 0.2], -0.4)
    @test gradient(x -> sum((y -> y/x).([1,2,3])), 4) == (-0.375,)  # closure

    @test gradient(x -> sum(sum, (x,) ./ x), [1,2,3])[1] ≈ [-4.1666, 0.3333, 1.1666] atol=1e-3  # array of arrays
    @test gradient(x -> sum(sum, Ref(x) ./ x), [1,2,3])[1] ≈ [-4.1666, 0.3333, 1.1666] atol=1e-3
    @test gradient(x -> sum(sum, (x,) ./ x), [1,2,3])[1] ≈ [-4.1666, 0.3333, 1.1666] atol=1e-3
    @test gradient(x -> sum(sum, (x,) .* transpose(x)), [1,2,3])[1] ≈ [12, 12, 12]  # must not take the * fast path

    @test gradient(x -> sum(x ./ 4), [1,2,3]) == ([0.25, 0.25, 0.25],)
    @test gradient(x -> sum([1,2,3] ./ x), 4) == (-0.375,)  # x/y rule
    @test gradient(x -> sum(x.^2), [1,2,3]) == ([2.0, 4.0, 6.0],)  # x.^2 rule
    @test gradient(x -> sum([1,2,3] ./ x.^2), 4) == (-0.1875,)  # scalar^2 rule

    @test gradient(x -> sum((1,2,3) .- x), (1,2,3)) == (Tangent{Tuple{Int,Int,Int}}(-1.0, -1.0, -1.0),)
    @test gradient(x -> sum(transpose([1,2,3]) .- x), (1,2,3)) == (Tangent{Tuple{Int,Int,Int}}(-3.0, -3.0, -3.0),)
    @test gradient(x -> sum([1 2 3] .+ x .^ 2), (1,2,3)) == (Tangent{Tuple{Int,Int,Int}}(6.0, 12.0, 18.0),)

    @test gradient(x -> sum(x .> 2), [1,2,3]) |> only |> iszero  # Bool output
    @test gradient(x -> sum(1 .+ iseven.(x)), [1,2,3]) |> only |> iszero
    @test gradient((x,y) -> sum(x .== y), [1,2,3], [1 2 3]) == (NoTangent(), NoTangent())
    @test gradient(x -> sum(x .+ [1,2,3]), true) |> only |> iszero  # Bool input
    @test gradient(x -> sum(x ./ [1,2,3]), [true false]) |> only |> iszero
    @test gradient(x -> sum(x .* transpose([1,2,3])), (true, false)) |> only |> iszero

    tup_adj = gradient((x,y) -> sum(2 .* x .+ log.(y)), (1,2), transpose([3,4,5]))
    @test tup_adj[1] == Tangent{Tuple{Int64, Int64}}(6.0, 6.0)
    @test tup_adj[2] ≈ [0.6666666666666666 0.5 0.4]
    @test tup_adj[2] isa Transpose
    @test gradient(x -> sum(atan.(x, (1,2,3))), Diagonal([4,5,6]))[1] isa Diagonal

    @test gradient(x -> sum((y -> (x*y)).([1,2,3])), 4.0) == (6.0,)  # closure
end

@testset "broadcast, 2nd order" begin
    @test gradient(x -> gradient(y -> sum(y .* y), x)[1] |> sum, [1,2,3.0])[1] == [2,2,2]  # calls "split broadcasting generic" with f = unthunk
    @test gradient(x -> gradient(y -> sum(y .* x), x)[1].^3 |> sum, [1,2,3.0])[1] == [3,12,27]
    @test_broken gradient(x -> gradient(y -> sum(y .* 2 .* y'), x)[1] |> sum, [1,2,3.0])[1] == [12, 12, 12]

    @test_broken gradient(x -> sum(gradient(x -> sum(x .^ 2 .+ x'), x)[1]), [1,2,3.0])[1] == [6,6,6]  # BoundsError: attempt to access 18-element Vector{Core.Compiler.BasicBlock} at index [0]
    @test_broken gradient(x -> sum(gradient(x -> sum((x .+ 1) .* x .- x), x)[1]), [1,2,3.0])[1] == [2,2,2]
    @test_broken gradient(x -> sum(gradient(x -> sum(x .* x ./ 2), x)[1]), [1,2,3.0])[1] == [1,1,1]

    @test_broken gradient(x -> sum(gradient(x -> sum(exp.(x)), x)[1]), [1,2,3])[1] ≈ exp.(1:3)  # MethodError: no method matching copy(::Nothing)
    @test_broken gradient(x -> sum(gradient(x -> sum(atan.(x, x')), x)[1]), [1,2,3.0])[1] ≈ [0,0,0]
    @test_broken gradient(x -> sum(gradient(x -> sum(transpose(x) .* x), x)[1]), [1,2,3]) == ([6,6,6],) # accum(a::Transpose{Float64, Vector{Float64}}, b::ChainRulesCore.Tangent{Transpose{Int64, Vector{Int64}}, NamedTuple{(:parent,), Tuple{ChainRulesCore.NoTangent}}})
    @test_broken gradient(x -> sum(gradient(x -> sum(transpose(x) ./ x.^2), x)[1]), [1,2,3])[1] ≈ [27.675925925925927, -0.824074074074074, -2.1018518518518516]

    @test_broken gradient(z -> gradient(x -> sum((y -> (x^2*y)).([1,2,3])), z)[1], 5.0) == (12.0,)
end

# Issue 67, due to https://github.com/JuliaDiff/ChainRulesCore.jl/pull/495
@test gradient(identity∘sqrt, 4.0) == (0.25,)

# Issue #70 - Complex & getproperty
@test_broken gradient(x -> x.re, 2+3im)[1] == 1  # Tangent{Complex{Int64}}(re = 1,)
@test_broken gradient(x -> abs2(x * x.re), 4+5im)[1] == 456 + 160im  # accum(a::ComplexF64, b::Tangent)
@test gradient(x -> abs2(x * real(x)), 4+5im)[1] == 456 + 160im
