import AbstractDifferentiation as AD
struct DiffractorForwardBackend <: AD.AbstractForwardMode
end

bundle(x::Number, dx) = TaylorBundle{1}(x, (dx,))
bundle(x::Tuple, dx) = CompositeBundle{1}(x, dx)
bundle(x::AbstractArray{<:Number}, dx::AbstractArray{<:Number}) = TaylorBundle{1}(x, (dx,))  # TODO check me
# TODO: other types of primal


AD.@primitive function pushforward_function(b::DiffractorForwardBackend, f, args...)
    return function pushforward(vs)
        z = ∂☆{1}()(ZeroBundle{1}(f), map(bundle, args, vs)...)
        z[TaylorTangentIndex(1)]
    end
end
