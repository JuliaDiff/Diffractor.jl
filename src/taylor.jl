using TaylorSeries

"""
    struct Jet{T, N}

Represents the truncated (N-1)-th order Taylor series

    f(a) + (x-a)f'(a) + 1/2(x-a)^2f''(a) + ...

Coefficients are stored in unscaled form.
For a jet `j`, several operations are supported:

1. Indexing `j[i]` returns fᵢ
2. Evaluation j(x) semantically evaluates the truncated taylor series at
   `x`. However, evaluation is restricted to be precisely at `a` - the
   additional information in the taylor series is only available through
   derivatives. Mathematically this corresponds to an infinitessimal ball
   around `a`.
"""
struct Jet{T, N}
    a::T
    fₙ::NTuple{N, T}
end

function ChainRulesCore.rrule(::typeof(Base.getproperty), j::Jet, s)
    error("Raw getproperty not allowed in AD code")
end

function Base.:+(j1::Jet{T, N}, j2::Jet{T, N}) where {T, N}
    @assert j1.a === j2.a
    Jet{T, N}(j1.a, map(+, j1.fₙ, j2.fₙ))
end

function Base.:+(j::Jet{T, N}, x::T) where {T, N}
    Jet{T, N}(j.a, tuple(j[0] + x, j.fₙ[2:end]...))
end

function Base.:+(j::Jet, x::One)
    j + one(j[0])
end

function ChainRulesCore.rrule(::typeof(+), j::Jet, x::One)
    j + x, Δ->(NO_FIELDS, One(), ZeroTangent())
end

function Base.zero(j::Jet{T, N}) where {T, N}
    Jet{T, N}(j.a, let z = zero(j[0])
        ntuple(_->z, N)
    end)
end
function ChainRulesCore.rrule(::typeof(Base.zero), j::Jet)
    zero(j), Δ->(NO_FIELDS, ZeroTangent())
end

function Base.getindex(j::Jet{T, N}, i::Integer) where {T, N}
    (0 <= i <= N-1) || throw(BoundsError(j, i))
    @inbounds j.fₙ[i + 1]
end

function deriv(j::Jet{T, N}) where {T, N}
    Jet{T, N-1}(j.a, j.fₙ[2:end])
end

function integrate(j::Jet{T, N}) where {T, N}
    Jet{T, N+1}(j.a, tuple(zero(j.fₙ[1]), j.fₙ...))
end

deriv(::DoesNotExist) = DoesNotExist()
integrate(::DoesNotExist) = DoesNotExist()

antideriv(j, Δ) = integrate(zero(j) + Δ)

function ChainRulesCore.rrule(::typeof(deriv), j::Jet)
    deriv(j), Δ->(NO_FIELDS, antideriv(j, Δ))
end

function ChainRulesCore.rrule(::typeof(integrate), j::Jet)
    integrate(j), Δ->(NO_FIELDS, deriv(Δ))
end

function Base.show(io::IO, j::Jet)
    print(io, join(map(enumerate(j.fₙ)) do (n, x)
        n -= 1
        buf = IOBuffer()
        if n != 0
            if n != 1
                print(buf, "1/",factorial(n))
            end
            print(buf, "(x - ", j.a, ")")
            if n != 1
                print(buf, "^", n)
            end
        end
        if x < 0
            print(buf, "(", x, ")")
        else
            print(buf, x)
        end
        String(take!(buf))
    end, " + "))
end

function domain_check(j::Jet, x)
    if j.a !== x
        throw(DomainError("Evaluation is only valid at a"))
    end
end

function ChainRulesCore.rrule(::typeof(domain_check), j::Jet, x)
    domain_check(j, x), Δ->(ZeroTangent(), ZeroTangent(), ZeroTangent())
end

function (j::Jet)(x)
    domain_check(j, x)
    j[0]
end

function ChainRulesCore.rrule(j::Jet, x)
    j(x), let dj = deriv(j)
        Δ->(DoesNotExist(), Δ*dj(x))
    end
end

function ChainRulesCore.rrule(::typeof(map), ::typeof(*), a, b)
    map(*, a, b), Δ->(NO_FIELDS, NO_FIELDS, map(*, Δ, b), map(*, a, Δ))
end

ChainRulesCore.rrule(::typeof(map), ::typeof(integrate), js::Array{<:Jet}) =
    map(integrate, js), Δ->(NO_FIELDS, NO_FIELDS, map(deriv, Δ))

struct derivBack
    js
end
(d::derivBack)(Δ::Union{ZeroTangent, DoesNotExist}) = (NO_FIELDS, NO_FIELDS, Δ)
(d::derivBack)(Δ::Array) = (NO_FIELDS, NO_FIELDS, broadcast(antideriv, d.js, Δ))

ChainRulesCore.rrule(::typeof(map), ::typeof(deriv), js::Array{<:Jet}) =
    map(deriv, js), derivBack(js)

ChainRulesCore.rrule(::typeof(map), ::typeof(antideriv), js::Array{<:Jet}, Δ) =
    map(antideriv, js, Δ), Δ->(NO_FIELDS, NO_FIELDS, map(deriv, Δ), One())

function mapev(js::Array{<:Jet}, xs::Array)
    map((j,x)->j(x), js, xs)
end

function ChainRulesCore.rrule(::typeof(mapev), js::Array{<:Jet}, xs::Array)
    mapev(js, xs), let djs=map(deriv, js)
        Δ->(NO_FIELDS, DoesNotExist(), map(*, Δ, mapev(djs, xs)))
    end
end

function (∂⃖ₙ::∂⃖{N})(::typeof(map), f, a::Array) where {N}
    @assert Base.issingletontype(typeof(f))
    # TODO: Built-in taylor mode
    t = map(a) do x
        f(x + Taylor1(typeof(x), N))
    end
    js = map((x, t)->Jet{typeof(x), N+1}(
        x, tuple(map(i->t.coeffs[i]*factorial(i-1), 1:(N+1))...)), a, t)
    ∂⃖ₙ(mapev, js, a)
end

# Exempt these from the definition above
for f in (integrate, deriv, antideriv)
    function (∂⃖ₙ::∂⃖{N})(m::typeof(map), f::typeof(f), a::Array) where {N}
        invoke(∂⃖ₙ, Tuple{Any, Vararg{Any}}, m, f, a)
    end
end
