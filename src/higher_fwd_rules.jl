using Combinatorics
using StatsBase

using Base.Iterators

function njet(::Val{N}, ::typeof(sin), x₀) where {N}
    (s, c) = sincos(x₀)
    Jet(x₀, s, tuple(take(cycle((c, -s, -c, s)), N)...))
end

function njet(::Val{N}, ::typeof(cos), x₀) where {N}
    (s, c) = sincos(x₀)
    Jet(x₀, s, tuple(take(cycle((-s, -c, s, c)), N)...))
end

function njet(::Val{N}, ::typeof(exp), x₀) where {N}
    exped = exp(x₀)
    Jet(x₀, exped, tuple(take(repeated(exped), N)...))
end

jeval(j, x) = j(x)
for f in (sin, cos, exp)
    function (∂☆ₙ::∂☆{N})(fb::ZeroBundle{N, typeof(f)}, x::TaylorBundle{N}) where {N}
        njet(Val{N}(), primal(fb), primal(x))(x)
    end
    function (∂⃖ₙ::∂⃖{N})(∂☆ₘ::∂☆{M}, fb::ZeroBundle{M, typeof(f)}, x::TaylorBundle{M}) where {N, M}
        ∂⃖ₙ(jeval, njet(Val{N+M}(), primal(fb), primal(x)), x)
    end
end

# TODO: It's a bit embarassing that we need to write these out, but currently the
# compiler is not strong enough to automatically lift the frule. Let's hope we
# can delete these in the near future.
function (∂☆ₙ::∂☆{N})(fb::ZeroBundle{N, typeof(+)}, a::TaylorBundle{N}, b::TaylorBundle{N}) where {N}
    TaylorBundle{N}(primal(a) + primal(b),
        map(+, a.tangent.coeffs, b.tangent.coeffs))
end

function (∂☆ₙ::∂☆{N})(fb::ZeroBundle{N, typeof(+)}, a::TaylorBundle{N}, b::ZeroBundle{N}) where {N}
    TaylorBundle{N}(primal(a) + primal(b), a.tangent.coeffs)
end

function (∂☆ₙ::∂☆{N})(fb::ZeroBundle{N, typeof(-)}, a::TaylorBundle{N}, b::TaylorBundle{N}) where {N}
    TaylorBundle{N}(primal(a) - primal(b),
        map(-, a.tangent.coeffs, b.tangent.coeffs))
end

function (::Diffractor.∂☆new{N})(B::ATB{N, Type{T}}, args::ATB{N}...) where {N, T<:SArray}
    error("Should have intercepted the constructor")
end
