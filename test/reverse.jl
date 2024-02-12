module reverse_tests
using Diffractor
using Diffractor: ∂⃖, DiffractorRuleConfig
using ChainRules
using ChainRulesCore
using ChainRulesCore: ZeroTangent, NoTangent, frule_via_ad, rrule_via_ad
using Symbolics
using LinearAlgebra

using Test

const fwd = Diffractor.PrimeDerivativeFwd
const bwd = Diffractor.PrimeDerivativeBack


# Check characteristic of exp rule
# broken see: https://github.com/JuliaDiff/Diffractor.jl/pull/142
@variables ω α β γ δ ϵ ζ η
@test_broken ((x1, c1) = ∂⃖{3}()(exp, ω)) isa Any
#==
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
==#


function simple_control_flow(b, x)
    if b
        return sin(x)
    else
        return cos(x)
    end
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
    @test @inferred(sin'''(1.0)) == -cos(1.0)
    # FIXME: These error with:
    #   Control flow support not fully implemented yet for higher-order reverse mode (TODO)
    @test_broken @inferred(sin''''(1.0)) == sin(1.0)
    @test_broken @inferred(sin'''''(1.0)) == cos(1.0)
    @test_broken @inferred(sin''''''(1.0)) == -sin(1.0)

    f_getfield(x) = getfield((x,), 1)
    @test f_getfield'(1) == 1
    @test f_getfield''(1) == NoTangent()
    @test f_getfield'''(1) == NoTangent()

    # Higher order mixed mode tests

    complicated_2sin(x) = (x = map(sin, Diffractor.xfill(x, 2)); x[1] + x[2])
    @test @inferred(complicated_2sin'(1.0)) == 2sin'(1.0)
    # FIXME: These error with:  Control flow support not fully implemented yet for higher-order reverse mode (TODO)
    @test_broken @inferred(complicated_2sin''(1.0)) == 2sin''(1.0)
    @test_broken @inferred(complicated_2sin'''(1.0)) == 2sin'''(1.0)
    @test_broken @inferred(complicated_2sin''''(1.0)) == 2sin''''(1.0)

    # Control flow cases:
    # if
    @test @inferred((x->simple_control_flow(true, x))'(1.0)) == sin'(1.0)
    @test @inferred((x->simple_control_flow(false, x))'(1.0)) == cos'(1.0)
    @test (x->sum(isa_control_flow(Matrix{Float64}, x)))'(Float32[1 2;]) == [1.0 1.0;]
    
    # while
    # @test times_three_while'(1.0) == 3.0  # hangs in 1.11
    pow5p(x) = (x->mypow(x, 5))'(x)
    #@test pow5p(1.0) == 5.0  # hangs in 1.11
end

end
