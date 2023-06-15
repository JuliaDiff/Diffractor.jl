using Diffractor
using Diffractor: var"'", ∂⃖, DiffractorRuleConfig
using ChainRules
using ChainRulesCore
using ChainRulesCore: ZeroTangent, NoTangent, frule_via_ad, rrule_via_ad
using Symbolics
using LinearAlgebra

using Test

const fwd = Diffractor.PrimeDerivativeFwd
const bwd = Diffractor.PrimeDerivativeBack

@testset verbose=true "Diffractor.jl" begin  # overall testset, ensures all tests run

@testset "$file" for file in (
    "stage2_fwd.jl",
    "tangent.jl",
    "forward_diff_no_inf.jl",
    "forward.jl",
    "reverse.jl",
    "regression.jl",
    #"pinn.jl",  # Higher order control flow not yet supported (https://github.com/JuliaDiff/Diffractor.jl/issues/24)
)
    include(file)
end

# 👻 The following code belongs in "test/reverse.jl" but if moved there it does not work anymore 👻
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






end  # overall testset
