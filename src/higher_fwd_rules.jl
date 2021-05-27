using Combinatorics
using StatsBase

using Base.Iterators

function (this::∂☆{N})(::ZeroBundle{N, typeof(sin)}, x::ATB{N}) where {N}
    x₀ = primal(x)
    (s, c) = sincos(x₀)
    j = Jet(x₀, s, tuple(take(cycle((c, -s, -c, s)), N)...))
    j(x)
end

function (this::∂☆{N})(::ZeroBundle{N, typeof(cos)}, x::ATB{N}) where {N}
    x₀ = primal(x)
    (s, c) = sincos(x₀)
    j = Jet(x₀, s, tuple(take(cycle((-s, -c, s, c)), N)...))
    j(x)
end

function (this::∂☆{N})(::ZeroBundle{N, typeof(exp)}, x::ATB{N}) where {N}
    x₀ = primal(x)
    exped = exp(x₀)
    j = Jet(x₀, s, tuple(take(repeated(exped), N)...))
    j(x)
end
