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
    f₀::T
    fₙ::NTuple{N, T}
end

function ChainRulesCore.rrule(::typeof(Base.getproperty), j::Jet, s)
    error("Raw getproperty not allowed in AD code")
end

function Base.:+(j1::Jet{T, N}, j2::Jet{T, N}) where {T, N}
    @assert j1.a === j2.a
    Jet{T, N}(j1.a, j1.f₀ + j2.f₀, map(+, j1.fₙ, j2.fₙ))
end

function Base.:+(j::Jet{T, N}, x::T) where {T, N}
    Jet{T, N}(j.a, j.f₀+x, j.fₙ)
end

struct One; end

function Base.:+(j::Jet, x::One)
    j + one(j[0])
end

function ChainRulesCore.rrule(::typeof(+), j::Jet, x::One)
    j + x, Δ->(NoTangent(), One(), ZeroTangent())
end

function Base.zero(j::Jet{T, N}) where {T, N}
    let z = zero(j[0])
        Jet{T, N}(j.a, z,
            ntuple(_->z, N))
    end
end
function ChainRulesCore.rrule(::typeof(Base.zero), j::Jet)
    zero(j), Δ->(NoTangent(), ZeroTangent())
end

function Base.getindex(j::Jet{T, N}, i::Integer) where {T, N}
    (0 <= i <= N) || throw(BoundsError(j, i))
    i == 0 && return j.f₀
    @inbounds j.fₙ[i]
end

function deriv(j::Jet{T, N}) where {T, N}
    Jet{T, N-1}(j.a, j.fₙ[1], Base.tail(j.fₙ))
end

function integrate(j::Jet{T, N}) where {T, N}
    Jet{T, N+1}(j.a, zero(j.f₀), tuple(j.f₀, j.fₙ...))
end

deriv(::NoTangent) = NoTangent()
integrate(::NoTangent) = NoTangent()

antideriv(j, Δ) = integrate(zero(j) + Δ)

function ChainRulesCore.rrule(::typeof(deriv), j::Jet)
    deriv(j), Δ->(NoTangent(), antideriv(j, Δ))
end

function ChainRulesCore.rrule(::typeof(integrate), j::Jet)
    integrate(j), Δ->(NoTangent(), deriv(Δ))
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
domain_check(j::Jet, x::ATB) = domain_check(j, primal(x))

function ChainRulesCore.rrule(::typeof(domain_check), j::Jet, x)
    domain_check(j, x), Δ->(ZeroTangent(), ZeroTangent(), ZeroTangent())
end

function (j::Jet)(x)
    domain_check(j, x)
    j[0]
end

function tderiv(bb::TaylorBundle{N}, i) where {N}
    i == 0 && return bb
    N-i == 0 && return bb.coeffs[i]
    TaylorBundle{N-i}(bb.coeffs[i], bb.coeffs[i+1:end])
end

function aldot(a::TaylorBundle{N}, b::TaylorBundle{M}) where {M,N}
    a.primal * b.primal +
        sum(1:min(M,N)) do i
            a.coeffs[i] * b.coeffs[i]
        end
end
aldot(a, b::TaylorBundle) = a * b.primal

*ₐₗ(a, b) = a * b
function *ₐₗ(a::Tangent{T}, b::TaylorBundle) where {N,B,T<:TaylorBundle{N,B}}
    # TODO: More general implementation of this
    bb = TaylorBundle{N}(a.primal, a.coeffs)
    cs = ntuple(1 + length(bb.coeffs)) do i
        aldot(tderiv(bb, i-1), b)
    end
    Tangent{T}(;primal = cs[1], coeffs = cs[2:end])
end

function ChainRulesCore.rrule(j::Jet, x)
    z = j(x)
    z, let djx = deriv(j)(x)
        function (Δ)
            (NoTangent(), Δ *ₐₗ djx)
        end
    end
end

function ChainRulesCore.rrule(::typeof(map), ::typeof(*), a, b)
    map(*, a, b), Δ->(NoTangent(), NoTangent(), map(*, Δ, b), map(*, a, Δ))
end

ChainRulesCore.rrule(::typeof(map), ::typeof(integrate), js::Array{<:Jet}) =
    map(integrate, js), Δ->(NoTangent(), NoTangent(), map(deriv, Δ))

struct derivBack
    js
end
(d::derivBack)(Δ::Union{ZeroTangent, NoTangent}) = (NoTangent(), NoTangent(), Δ)
(d::derivBack)(Δ::Array) = (NoTangent(), NoTangent(), broadcast(antideriv, d.js, Δ))

ChainRulesCore.rrule(::typeof(map), ::typeof(deriv), js::Array{<:Jet}) =
    map(deriv, js), derivBack(js)

ChainRulesCore.rrule(::typeof(map), ::typeof(antideriv), js::Array{<:Jet}, Δ) =
    map(antideriv, js, Δ), Δ->(NoTangent(), NoTangent(), map(deriv, Δ), One())

function mapev(js::Array{<:Jet}, xs::AbstractArray)
    map((j,x)->j(x), js, xs)
end

function ChainRulesCore.rrule(::typeof(mapev), js::Array{<:Jet}, xs::AbstractArray)
    mapev(js, xs), let djs=map(deriv, js)
        Δ->(NoTangent(), NoTangent(), map(*, unthunk(Δ), mapev(djs, xs)))
    end
end

function (∂⃖ₙ::∂⃖{N})(::typeof(map), f, a::Array) where {N}
    @assert Base.issingletontype(typeof(f))
    js = map(a) do x
        ∂f = ∂☆{N}()(ZeroBundle{N}(f),
                     TaylorBundle{N}(x,
                       (one(x), (zero(x) for i = 1:(N-1))...,)))
        @assert isa(∂f, TaylorBundle) || isa(∂f, ExplicitTangentBundle{1})
        Jet{typeof(x), N}(x, ∂f.primal,
            isa(∂f, ExplicitTangentBundle) ? ∂f.tangent.partials : ∂f.tangent.coeffs)
    end
    ∂⃖ₙ(mapev, js, a)
end

# Exempt these from the definition above
for f in (integrate, deriv, antideriv)
    function (∂⃖ₙ::∂⃖{N})(m::typeof(map), f::typeof(f), a::Array) where {N}
        invoke(∂⃖ₙ, Tuple{Any, Vararg{Any}}, m, f, a)
    end
end

"""
    jet_taylor_ev(::Val{}, jet, taylor)

Generates a closed form arithmetic expression for the N-th component
of the action of a 1d jet (of order at least N) on a maximally symmetric
(i.e. taylor) tangent bundle element. In particular, if we represent both
the `jet` and the `taylor` tangent bundle element by their associated canonical
taylor series:

    j = j₀ + j₁ (x - a) + j₂ 1/2 (x - a)^2 + ... + jₙ 1/n! (x - a)^n
    t = t₀ + t₁ (x - t₀) + t₂ 1/2 (x - t₀)^2 + ... + tₙ 1/n! (x - t₀)^n

then the action of evaluating `j` on `t`, is some other taylor series

    t′ = a + t′₁ (x - a) + t′₂ 1/2 (x - a)^2 + ... + t′ₙ 1/n! (x - a)^n

The t′ᵢ can be found by explicitly plugging in `t` for every `x` and expanding
out, dropping terms of orders that are higher. This computes closed form
expressions for the t′ᵢ that are hopefully easier on the compiler.
"""
@generated function jet_taylor_ev(::Val{N}, jet, taylor) where {N}
    elements = Any[nothing for _ = 1:N]
    for part in partitions(N)
        coeff = (factorial(N) ÷ (prod(factorial, values(countmap(part))) *
            prod(factorial, part)))
        factor = mapreduce((a,b)->:($a*$b), part) do i
            :(taylor[$i])
        end
        summand = coeff == 1 ? factor : :($coeff*$factor)
        let e = elements[length(part)]
            elements[length(part)] = e === nothing ? summand : :($summand + $e)
        end
    end
    return Expr(:call, +, map(enumerate(elements)) do (i, e)
        :(jet[$i]*$e)
    end...)
end

@generated function (j::Jet{T, N} where T)(x::TaylorBundle{M}) where {N, M}
    O = min(M,N)
    quote
        domain_check(j, x.primal)
        coeffs = x.tangent.coeffs
        TaylorBundle{$O}(j[0],
            ($((:(jet_taylor_ev(Val{$i}(), coeffs, j)) for i = 1:O)...),))
    end
end

function (j::Jet{T, 1} where T)(x::ExplicitTangentBundle{1})
    domain_check(j, x.primal)
    coeffs = x.tangent.partials
    ExplicitTangentBundle{1}(j[0], (jet_taylor_ev(Val{1}(), coeffs, j),))
end

function (j::Jet{T, N} where T)(x::ExplicitTangentBundle{N, M}) where {N, M}
    error("TODO")
end
