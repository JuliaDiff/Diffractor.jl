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
bundle(x::Number, dx::Number) = TaylorBundle{1}(x, (dx,))
bundle(x::AbstractArray{<:Number}, dx) = TaylorBundle{1}(x, (dx,))
bundle(x::AbstractArray, dx) = error("Nonnumeric arrays not implemented, that type is a mess")
bundle(x::P, dx::Tangent{P}) where P = _bundle(x, ChainRulesCore.canonicalize(dx))

"helper that assumes tangent is in canonical form"
function _bundle(x::P, dx::Tangent{P}) where P
    # SoA to AoS flip (hate this, hate it even more cos we just undo it later when we hit chainrules)
    the_bundle = ntuple(Val{fieldcount(P)}()) do ii
        bundle(getfield(x, ii), getproperty(dx, ii))
    end
    return CompositeBundle{1, P}(the_bundle)
end


AD.@primitive function pushforward_function(b::DiffractorForwardBackend, f, args...)
    return function pushforward(vs)
        z = ∂☆{1}()(ZeroBundle{1}(f), map(bundle, args, vs)...)
        z[TaylorTangentIndex(1)]
    end
end
