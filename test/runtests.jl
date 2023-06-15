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
)
    include(file)
end



# Higher order control flow not yet supported (https://github.com/JuliaDiff/Diffractor.jl/issues/24)
#include("pinn.jl")

end  # overall testset
