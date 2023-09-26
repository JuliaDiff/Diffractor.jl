import AbstractDifferentiation as AD
struct DiffractorForwardBackend <: AD.AbstractForwardMode
end

"""
    bundle(primal, tangent)

Wraps a primal up with a tangent into the appropriate kind of `AbstractBundle{1}`.
This is more or less the Diffractor equivelent of ForwardDiff.jl's `Dual` type.
"""
function bundle end
bundle(x, dx::ChainRulesCore.AbstractZero) = UniformBundle{1, typeof(x), typeof(dx)}(x, dx)
bundle(x, dx) = TaylorBundle{1}(x, (dx,))

AD.@primitive function pushforward_function(b::DiffractorForwardBackend, f, args...)
    return function pushforward(vs)
        z = ∂☆{1}()(ZeroBundle{1}(f), map(bundle, args, vs)...)
        z[TaylorTangentIndex(1)]
    end
end
