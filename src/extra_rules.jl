using StructArrays
using ChainRulesCore: NO_FIELDS

struct ∇getindex{T,S}
    xs::T
    i::S
end

function (g::∇getindex)(Δ)
    Δ′ = zero(g.xs)
    Δ′[g.i...] = Δ
    (ChainRulesCore.NO_FIELDS, Δ′, map(_ -> nothing, g.i)...)
end

function ChainRulesCore.rrule(g::∇getindex, Δ)
    g(Δ), Δ′′->(nothing, Δ′′[1][g.i...])
end

function ChainRulesCore.rrule(::typeof(getindex), xs::Array, i...)
    xs[i...], ∇getindex(xs, i)
end

function reversediff(f, xs...)
    y, f☆ = ∂⃖(f, xs...)
    return tuple(y, tail(f☆(dx(y)))...)
end

function reversediff_array(f, xs::Vector...)
    fieldarrays(StructArray(reversediff(f, x...) for x in zip(xs...)))
end

function reversediff_array(f, xs::Vector)
    fieldarrays(StructArray(reversediff(f, x) for x in xs))
end

function assert_gf(f)
    @assert sizeof(sin) == 0
end

function ChainRulesCore.rrule(::typeof(assert_gf), f)
    assert_gf(f), Δ->begin
        (NO_FIELDS, NO_FIELDS)
    end
end

#=
function ChainRulesCore.rrule(::typeof(map), f, xs::Vector...)
    assert_gf(f)
    primal, dual = reversediff_array(f, xs...)
    primal, Δ->begin
        (NO_FIELDS, NO_FIELDS, ntuple(i->map(*, getfield(dual, i), Δ), length(dual))...)
    end
end
=#

function ChainRulesCore.rrule(::typeof(*), A::AbstractMatrix{<:Real}, B::AbstractVector{<:Real})
    function times_pullback(Ȳ)
        return (NO_FIELDS, Ȳ * Base.adjoint(B), Base.adjoint(A) * Ȳ)
    end
    return A * B, times_pullback
end


#=
function ChainRulesCore.rrule(::typeof(map), f, xs::Vector)
    assert_gf(f)
    arrs = reversediff_array(f, xs)
    primal = getfield(arrs, 1)
    primal, let dual = getfield(arrs, 2)
        Δ->(NO_FIELDS, NO_FIELDS, map(*, dual, unthunk(Δ)))
    end
end
=#

function ChainRulesCore.rrule(::typeof(map), f, xs::Vector, ys::Vector)
    assert_gf(f)
    arrs = reversediff_array(f, xs, ys)
    primal = getfield(arrs, 1)
    primal, let dual = tail(arrs)
        Δ->(NO_FIELDS, NO_FIELDS, map(*, getfield(dual, 1), Δ), map(*, getfield(dual, 2), Δ))
    end
end

xsum(x::Vector) = sum(x)
function ChainRulesCore.rrule(::typeof(xsum), x::Vector)
    xsum(x), let xdims=size(x)
        Δ->(NO_FIELDS, fill(Δ, xdims...))
    end
end

struct NonDiffEven{N, O, P}; end
struct NonDiffOdd{N, O, P}; end

(::NonDiffOdd{N, O, P})(Δ) where {N, O, P} = (ntuple(_->Zero(), N), NonDiffEven{N, plus1(O), P}())
(::NonDiffEven{N, O, P})(Δ...) where {N, O, P} = (Zero(), NonDiffOdd{N, plus1(O), P}())
(::NonDiffOdd{N, O, O})(Δ) where {N, O} = ntuple(_->Zero(), N)

# This should not happen
(::NonDiffEven{N, O, O})(Δ...) where {N, O} = error()

@Base.pure function ChainRulesCore.rrule(::typeof(Core.apply_type), head, args...)
    Core.apply_type(head, args...), NonDiffOdd{plus1(plus1(length(args))), 1, 1}()
end

function ChainRulesCore.rrule(::typeof(Core.tuple), args...)
    Core.tuple(args...), Δ->Core.tuple(NO_FIELDS, Δ...)
end

ChainRulesCore.canonicalize(::ChainRulesCore.Zero) = ChainRulesCore.Zero()

# Skip AD'ing through the axis computation
function ChainRules.rrule(::typeof(Base.Broadcast.instantiate), bc::Base.Broadcast.Broadcasted)
    return Base.Broadcast.instantiate(bc), Δ->begin
        Core.tuple(NO_FIELDS, Δ)
    end
end


using StaticArrays

# Force the static arrays constructor to use a vector representation of
# the cotangent space.

struct to_tuple{N}; end
@generated function (::to_tuple{N})(Δ) where {N}
    :( (NO_FIELDS, Core.tuple( $( ( :(Δ[$i]) for i = 1:N )...) )) )
end

function ChainRules.rrule(::Type{SArray{S, T, N, L}}, x::NTuple{L,T}) where {S, T, N, L}
    SArray{S, T, N, L}(x), to_tuple{L}()
end

function ChainRules.rrule(::typeof(map), ::typeof(+), A::Array, B::Array)
    map(+, A, B), Δ->(NO_FIELDS, NO_FIELDS, Δ, Δ)
end

function ChainRules.rrule(::typeof(map), ::typeof(+), A::Vector, B::Vector)
    map(+, A, B), Δ->(NO_FIELDS, NO_FIELDS, Δ, Δ)
end
