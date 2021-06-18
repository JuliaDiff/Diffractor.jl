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

function ChainRulesCore.rrule(::typeof(*), A::AbstractVecOrMat, B::AbstractVecOrMat)
    function times_pullback(Ȳ)
        return (NO_FIELDS, Ȳ * Base.adjoint(B), Base.adjoint(A) * Ȳ)
    end
    return A * B, times_pullback
end

function ChainRulesCore.rrule(::typeof(*), A::AbstractVector{<:ChainRules.CommutativeMulNumber}, B::AbstractMatrix{<:ChainRules.CommutativeMulNumber})
    function times_pullback(Ȳ)
        return (NO_FIELDS, Ȳ * Base.adjoint(B), Base.adjoint(A) * Ȳ)
    end
    return A * B, times_pullback
end


function ChainRulesCore.frule((_, ∂A, ∂B), ::typeof(*), A::AbstractMatrix{<:Real}, B::AbstractVector{<:Real})
    return (A * B, ∂A * B + A * ∂B)
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

#=
function ChainRulesCore.rrule(::typeof(map), f, xs::Vector, ys::Vector)
    assert_gf(f)
    arrs = reversediff_array(f, xs, ys)
    primal = getfield(arrs, 1)
    primal, let dual = tail(arrs)
        Δ->(NO_FIELDS, NO_FIELDS, map(*, getfield(dual, 1), Δ), map(*, getfield(dual, 2), Δ))
    end
end
=#

xsum(x::Vector) = sum(x)
function ChainRulesCore.rrule(::typeof(xsum), x::Vector)
    xsum(x), let xdims=size(x)
        Δ->(NO_FIELDS, xfill(Δ, xdims...))
    end
end

xfill(x, dims...) = fill(x, dims...)
function ChainRulesCore.rrule(::typeof(xfill), x, dim)
    xfill(x, dim), Δ->(NO_FIELDS, xsum(Δ), NO_FIELDS)
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
(::to_tuple)(Δ::SArray) = getfield(Δ, :data)

function ChainRules.rrule(::Type{SArray{S, T, N, L}}, x::NTuple{L,T}) where {S, T, N, L}
    SArray{S, T, N, L}(x), to_tuple{L}()
end

function ChainRules.rrule(::Type{SArray{S, T, N, L}}, x::NTuple{L,Any}) where {S, T, N, L}
    SArray{S, T, N, L}(x), to_tuple{L}()
end

function ChainRules.frule((_, ∂x), ::Type{SArray{S, T, N, L}}, x::NTuple{L,T}) where {S, T, N, L}
    SArray{S, T, N, L}(x), SArray{S, T, N, L}(∂x)
end

function ChainRules.frule((_, ∂x), ::Type{SArray{S, T, N, L}}, x::NTuple{L,Any}) where {S, T, N, L}
    SArray{S, T, N, L}(x), SArray{S}(∂x)
end

@ChainRulesCore.non_differentiable StaticArrays.promote_tuple_eltype(T)

function ChainRules.frule((_, ∂A), ::typeof(getindex), A::AbstractArray, args...)
    getindex(A, args...), getindex(∂A, args...)
end

function ChainRules.rrule(::typeof(map), ::typeof(+), A::AbstractArray, B::AbstractArray)
    map(+, A, B), Δ->(NO_FIELDS, NO_FIELDS, Δ, Δ)
end

function ChainRules.rrule(::typeof(map), ::typeof(+), A::AbstractVector, B::AbstractVector)
    map(+, A, B), Δ->(NO_FIELDS, NO_FIELDS, Δ, Δ)
end

function ChainRules.rrule(AT::Type{<:Array{T,N}}, x::AbstractArray{S,N}) where {T,S,N}
    # We're leaving these in the eltype that the cotangent vector already has.
    # There isn't really a good reason to believe we should convert to the
    # original array type, so don't unless explicitly requested.
    AT(x), Δ->(NO_FIELDS, Δ)
end

function ChainRules.rrule(AT::Type{<:Array}, undef::UndefInitializer, args...)
    # We're leaving these in the eltype that the cotangent vector already has.
    # There isn't really a good reason to believe we should convert to the
    # original array type, so don't unless explicitly requested.
    AT(undef, args...), Δ->(NO_FIELDS, NO_FIELDS, ntuple(_->NO_FIELDS, length(args))...)
end

function unzip_tuple(t::Tuple)
    map(x->x[1], t), map(x->x[2], t)
end

function ChainRules.rrule(::typeof(unzip_tuple), args::Tuple)
    unzip_tuple(args), Δ->(NO_FIELDS, map((x,y)->(x,y), Δ...))
end

struct BackMap{T}
    f::T
end
(f::BackMap{N})(args...) where {N} = ∂⃖¹(getfield(f, :f), args...)
back_apply(x, y) = x(y)
back_apply_zero(x) = x(Zero())

function ChainRules.rrule(::typeof(map), f, args::Tuple)
    a, b = unzip_tuple(map(BackMap(f), args))
    function back(Δ)
        (fs, xs) = unzip_tuple(map(back_apply, b, Δ))
        (NO_FIELDS, sum(fs), xs)
    end
    function back(Δ::Zero)
        (fs, xs) = unzip_tuple(map(back_apply_zero, b))
        (NO_FIELDS, sum(fs), xs)
    end
    a, back
end

function ChainRules.rrule(::typeof(Base.ntuple), f, n)
    a, b = unzip_tuple(ntuple(BackMap(f), n))
    a, function (Δ)
        (NO_FIELDS, sum(map(back_apply, b, Δ)), DoesNotExist())
    end
end

function ChainRules.frule(_, ::Type{Vector{T}}, undef::UndefInitializer, dims::Int...) where {T}
    Vector{T}(undef, dims...), zeros(T, dims...)
end

@ChainRules.non_differentiable Base.:(|)(a::Integer, b::Integer)
@ChainRules.non_differentiable Base.throw(err)
@ChainRules.non_differentiable Core.Compiler.return_type(args...)
ChainRulesCore.canonicalize(::DoesNotExist) = DoesNotExist()
