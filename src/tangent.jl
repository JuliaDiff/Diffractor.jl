"""
    abstract type AbstractTangentBundle{N, B}; end

This type represents the `N`-th order (iterated) tangent bundle [1] `TⁿB` over some base
(Riemannian) manifold `B`. Note that `TⁿB` is itself another manifold and thus
in particular a vector space (over ℝ). As such, subtypes of this abstract type
are expected to support the usual vector space operations.

However, beyond that, this abstract type makes no guarantee about the
representation. That said, to gain intution for what this object is,
it makes sense to pick some explicit bases and write down examples.

To that end, suppose that B=ℝ. Then `T¹B=T¹ℝ` is just our usual notion of a dual
number, i.e. for some element `η ∈ T¹ℝ`, we may consider `η = a + bϵ` for
real numbers `(a, b)` and `ϵ` an infinitessimal differential such that `ϵ^2 = 0`.

Equivalently, we may think of `η` as being identified with the vector
`(a, b) ∈ ℝ²` with some additional structure.
The same form essentially holds for general `B`, where we may write (as sets):

    T¹B = {(a, b) | a ∈ B, b ∈ Tₐ B }

Note that these vectors are orthogonal to those in the underlying base space.
For example, if `B=ℝ²`, then we have:

    T¹ℝ² = {([aₓ, a_y], [bₓ, b_y]) | [aₓ, a_y] ∈ ℝ², [bₓ, b_y] ∈ Tₐ ℝ² }

For convenience, we will sometimes writes these in one as:

    η ∈ T ℝ²  = aₓ x̂ + a_y ŷ + bₓ ∂/∂x|_aₓ x + b_y ∂/∂y|_{a_y}
             := aₓ x̂ + a_y ŷ + bₓ ∂/∂¹x + b_y ∂/∂¹y
             := [aₓ, a_y] + [bₓ, b_y] ∂/∂¹
             := a + b ∂/∂¹
             := a + b ∂₁

These are all definitional equivalences and we will mostly work with the final
form. An important thing to keep in mind though is that the subscript on ∂₁ does
not refer to a dimension of the underlying base manifold (for which we will
rarely pick an explicit basis here), but rather tags the basis of the tangent
bundle.

Let us iterate this construction to second order. We have:


    T²B = T¹(T¹B) = { (α, β) | α ∈ T¹B, β ∈ T_α T¹B }
                  = { ((a, b), (c, d)) | a ∈ B, b ∈ Tₐ B, c ∈ Tₐ B, d ∈ T²ₐ B}

(where in the last equality we used the linearity of the tangent vector).

Following our above notation, we will canonically write such an element as:

      a + b ∂₁ + c ∂₂ + d ∂₂ ∂₁
    = a + b ∂₁ + c ∂₂ + d ∂₁ ∂₂

It is worth noting that there still only is one base point `a` of the
underlying manifold and thus `TⁿB` is a vector bundle over `B` for all `N`.

# Further Reading

[1] https://en.wikipedia.org/wiki/Tangent_bundle
"""
abstract type AbstractTangentBundle{N, B}; end
basespace(::Type{<:AbstractTangentBundle{N, B} where N}) where {B} = B

const ATB = AbstractTangentBundle

abstract type TangentIndex; end

struct CanonicalTangentIndex <: TangentIndex
    i::Int
end

struct TaylorTangentIndex <: TangentIndex
    i::Int
end

abstract type AbstractTangentSpace; end
Base.:(==)(x::AbstractTangentSpace, y::AbstractTangentSpace) = ==(promote(x, y)...)

"""
    struct ExplicitTangent{P}

A fully explicit coordinate representation of the tangent space,
represented by a vector of `2^N-1` partials.
"""
struct ExplicitTangent{P <: Tuple} <: AbstractTangentSpace
    partials::P
end
Base.:(==)(a::ExplicitTangent, b::ExplicitTangent) = a.partials == b.partials
Base.hash(tt::ExplicitTangent, h::UInt64) = hash(tt.partials, h)

Base.getindex(tangent::ExplicitTangent, b::CanonicalTangentIndex) = tangent.partials[b.i]
function Base.getindex(tangent::ExplicitTangent, b::TaylorTangentIndex)
    if lastindex(tangent.partials) == exp2(b.i) - 1
        return tangent.partials[end]
    end
    # TODO: should we also allow other indexes if all the partials at that level are equal up regardless of order?
    throw(DomainError(b, "$(typeof(tangent)) is not taylor-like. Taylor indexing is ambiguous"))
end


@eval struct TaylorTangent{C <: Tuple} <: AbstractTangentSpace
    coeffs::C
    TaylorTangent(coeffs) = $(Expr(:new, :(TaylorTangent{typeof(coeffs)}), :coeffs))
end

"""
    struct TaylorTangent{C}

The taylor bundle construction mods out the full N-th order tangent bundle
by the equivalence relation that coefficients of like-order basis elements be
equal, i.e. rather than a generic element

    a + b ∂₁ + c ∂₂ + d ∂₃ + e ∂₂ ∂₁ + f ∂₃ ∂₁ + g ∂₃ ∂₂ + h ∂₃ ∂₂ ∂₁

we have a tuple (c₀, c₁, c₂, c₃) corresponding to the full element

    c₀ + c₁ ∂₁ + c₁ ∂₂ + c₁ ∂₃ + c₂ ∂₂ ∂₁ + c₂ ∂₃ ∂₁ + c₂ ∂₃ ∂₂ + c₃ ∂₃ ∂₂ ∂₁

i.e.

    c₀ + c₁ (∂₁ + ∂₂ + ∂₃) + c₂ (∂₂ ∂₁ + ∂₃ ∂₁ + ∂₃ ∂₂) + c₃ ∂₃ ∂₂ ∂₁


This restriction forms a submanifold of the original manifold. The naming is
by analogy with the (truncated) Taylor series

    c₀ + c₁ x + 1/2 c₂ x² + 1/3! c₃ x³ + O(x⁴)
"""
TaylorTangent

Base.:(==)(a::TaylorTangent, b::TaylorTangent) = a.coeffs == b.coeffs
Base.hash(tt::TaylorTangent, h::UInt64) = hash(tt.coeffs, h)


Base.getindex(tangent::TaylorTangent, tti::TaylorTangentIndex) = tangent.coeffs[tti.i]
Base.getindex(tangent::TaylorTangent, tti::CanonicalTangentIndex) = tangent.coeffs[count_ones(tti.i)]


"""
    struct ProductTangent{T <: Tuple{Vararg{AbstractTangentSpace}}}

Represents the product space of the given representations of the
tangent space.
"""
struct ProductTangent{T <: Tuple} <: AbstractTangentSpace
    factors::T
end

"""
    struct UniformTangent

Represents an N-th order tangent bundle with all unform partials. Particularly
useful for representing singleton values.
"""
struct UniformTangent{U} <: AbstractTangentSpace
    val::U
end
Base.hash(t::UniformTangent, h::UInt64) = hash(t.val, h)
Base.:(==)(t1::UniformTangent, t2::UniformTangent) = t1.val == t2.val

Base.getindex(tangent::UniformTangent, ::Any) = tangent.val

# Conversion and promotion
Base.promote_rule(et::Type{<:ExplicitTangent}, ::Type{<:AbstractTangentSpace}) = et
Base.promote_rule(tt::Type{<:TaylorTangent}, ::Type{<:AbstractTangentSpace}) = tt
Base.promote_rule(et::Type{<:ExplicitTangent}, ::Type{<:TaylorTangent}) = et
Base.promote_rule(::Type{<:TaylorTangent}, et::Type{<:ExplicitTangent}) = et

num_partials(::Type{TaylorTangent{P}}) where P = fieldcount(P)
num_partials(::Type{ExplicitTangent{P}}) where P = fieldcount(P)
Base.eltype(::Type{TaylorTangent{P}}) where P = eltype(P)
Base.eltype(::Type{ExplicitTangent{P}}) where P = eltype(P)
function Base.convert(::Type{T}, ut::UniformTangent) where {T<:Union{TaylorTangent, ExplicitTangent}}
    # can't just use T to construct as the inner constructor doesn't accept type params. So get T_wrapper
    T_wrapper = T<:TaylorTangent ? TaylorTangent : ExplicitTangent  
    T_wrapper(ntuple(_->convert(eltype(T), ut.val), num_partials(T)))
end
Base.convert(T::Type{<:ExplicitTangent},  tt::TaylorTangent) = ExplicitTangent(ntuple(i->tt[CanonicalTangentIndex(i)], num_partials(T)))
#TODO: Should we define the reverse: Explict->Taylor for the cases where that is actually defined?

function _TangentBundle end

@eval struct TangentBundle{N, B, P <: AbstractTangentSpace} <: AbstractTangentBundle{N, B}
    primal::B
    tangent::P
    global _TangentBundle(::Val{N}, primal::B, tangent::P) where {N, B, P} = $(Expr(:new, :(TangentBundle{N, Core.Typeof(primal), typeof(tangent)}), :primal, :tangent))
end

"""
    struct TangentBundle{N, B, P}

Represents a tangent bundle as an explicit primal together
with some representation of (potentially a product of) the tangent space.
"""
TangentBundle

TangentBundle{N}(primal::B, tangent::P) where {N, B, P<:AbstractTangentSpace} =
    _TangentBundle(Val{N}(), primal, tangent)

Base.hash(tb::TangentBundle, h::UInt64) = hash(tb.primal, h)
Base.:(==)(a::TangentBundle, b::TangentBundle) = false  # different orders
Base.:(==)(a::TangentBundle{N}, b::TangentBundle{N}) where {N} = (a.primal == b.primal) && (a.tangent == b.tangent)
Base.getindex(tbun::TangentBundle, x) = getindex(tbun.tangent, x)

const ExplicitTangentBundle{N, B, P} = TangentBundle{N, B, ExplicitTangent{P}}

check_tangent_invariant(lp, N) = @assert lp == 2^N - 1
@ChainRulesCore.non_differentiable check_tangent_invariant(lp, N)

function ExplicitTangentBundle{N}(primal::B, partials::P) where {N, B, P}
    check_tangent_invariant(length(partials), N)
    _TangentBundle(Val{N}(), primal, ExplicitTangent{P}(partials))
end

function ExplicitTangentBundle{N,B}(primal::B, partials::P) where {N, B, P}
    check_tangent_invariant(length(partials), N)
    _TangentBundle(Val{N}(), primal, ExplicitTangent{P}(partials))
end

function ExplicitTangentBundle{N,B,P}(primal::B, partials::P) where {N, B, P}
    check_tangent_invariant(length(partials), N)
    _TangentBundle(Val{N}(), primal, ExplicitTangent{P}(partials))
end

function Base.show(io::IO, x::ExplicitTangentBundle)
    print(io, x.primal)
    print(io, " + ")
    x = x.tangent
    print(io, x.partials[1], " ∂₁")
    length(x.partials) >= 2 && print(io, " + ", x.partials[2], " ∂₂")
    length(x.partials) >= 3 && print(io, " + ", x.partials[3], " ∂₁ ∂₂")
    length(x.partials) >= 4 && print(io, " + ", x.partials[4], " ∂₃")
    length(x.partials) >= 5 && print(io, " + ", x.partials[5], " ∂₁ ∂₃")
    length(x.partials) >= 6 && print(io, " + ", x.partials[6], " ∂₂ ∂₃")
    length(x.partials) >= 7 && print(io, " + ", x.partials[7], " ∂₁ ∂₂ ∂₃")
end



const TaylorBundle{N, B, P} = TangentBundle{N, B, TaylorTangent{P}}


function TaylorBundle{N, B, P}(primal::B, coeffs::P) where {N, B, P}
    check_taylor_invariants(coeffs, primal, N)
    _TangentBundle(Val{N}(), primal, TaylorTangent(coeffs))
end
function TaylorBundle{N, B}(primal::B, coeffs) where {N, B}
    check_taylor_invariants(coeffs, primal, N)
    _TangentBundle(Val{N}(), primal, TaylorTangent(coeffs))
end
function TaylorBundle{N}(primal, coeffs) where {N}
    check_taylor_invariants(coeffs, primal, N)
    _TangentBundle(Val{N}(), primal, TaylorTangent(coeffs))
end

function check_taylor_invariants(coeffs, primal, N)
    @assert length(coeffs) == N
end
@ChainRulesCore.non_differentiable check_taylor_invariants(coeffs, primal, N)


function Base.show(io::IO, x::TaylorBundle{1})
    print(io, x.primal)
    print(io, " + ")
    x = x.tangent
    print(io, x.coeffs[1], " ∂₁")
end

"for a TaylorTangent{N, <:Tuple} this breaks it up unto 1 TaylorTangent{N} for each element of the primal tuple"
function destructure(r::TaylorBundle{N, B}) where {N, B<:Tuple}
    return ntuple(fieldcount(B)) do field_ii
        the_primal = primal(r)[field_ii]
        the_partials = ntuple(N) do  order_ii
            partial(r, order_ii)[field_ii]
        end
        return TaylorBundle{N}(the_primal, the_partials)
    end
end


function truncate(tt::TaylorTangent, order::Val{N}) where {N}
    TaylorTangent(tt.coeffs[1:N])
end

function truncate(ut::UniformTangent, order::Val)
    ut
end

function truncate(tb::TangentBundle, order::Val)
    _TangentBundle(order, tb.primal, truncate(tb.tangent, order))
end

function truncate(tb::ExplicitTangent, order::Val{N}) where {N}
    ExplicitTangent(tb.partials[1:2^N-1])
end

function truncate(et::ExplicitTangent, order::Val{1})
    TaylorTangent(et.partials[1:1])
end

const UniformBundle{N, B, U} = TangentBundle{N, B, UniformTangent{U}}
UniformBundle{N, B, U}(primal::B, partial::U) where {N,B,U} = _TangentBundle(Val{N}(), primal, UniformTangent{U}(partial))
UniformBundle{N, B, U}(primal::B) where {N,B,U} = _TangentBundle(Val{N}(), primal, UniformTangent{U}(U.instance))
UniformBundle{N, B}(primal::B, partial::U) where {N,B,U} = _TangentBundle(Val{N}(),primal, UniformTangent{U}(partial))
UniformBundle{N}(primal, partial::U) where {N,U} = _TangentBundle(Val{N}(), primal, UniformTangent{U}(partial))
UniformBundle{N, <:Any, U}(primal, partial::U) where {N, U} = _TangentBundle(Val{N}(), primal, UniformTangent{U}(U.instance))
UniformBundle{N, <:Any, U}(primal) where {N, U} = _TangentBundle(Val{N}(), primal, UniformTangent{U}(U.instance))


const ZeroBundle{N, B} = UniformBundle{N, B, ZeroTangent}
const DNEBundle{N, B} = UniformBundle{N, B, NoTangent}
const AbstractZeroBundle{N, B} = UniformBundle{N, B, <:AbstractZero}

wrapper_name(::Type{<:ZeroBundle}) = "ZeroBundle"
wrapper_name(::Type{<:DNEBundle}) = "DNEBundle"
wrapper_name(::Type{<:AbstractZeroBundle}) = "AbstractZeroBundle"

function Base.show(io::IO, T::Type{<:AbstractZeroBundle{N, B}}) where {N,B}
    print(io, wrapper_name(T))
    print(io, @isdefined(N) ? "{$N, " : "{N, ")
    @isdefined(B) ? show(io, B) : print(io, "B")
    print(io, "}")
end

function Base.show(io::IO, T::Type{<:AbstractZeroBundle{N}}) where {N}
    print(io, wrapper_name(T))
    @isdefined(N) && print(io, "{$N}")
end

function Base.show(io::IO, t::AbstractZeroBundle{N}) where N
    print(io, wrapper_name(typeof(t)))
    @isdefined(N) && print(io, "{$N}")
    print(io, "(")
    show(io, t.primal)
    print(io, ")")
end

# Conversion and promotion
function Base.promote_rule(::Type{TangentBundle{N, B, P1}}, ::Type{TangentBundle{N, B, P2}}) where {N,B,P1,P2}
    return TangentBundle{N, B, promote_type(P1, P2)}
end

function Base.convert(::Type{T}, tbun::TangentBundle{N, B}) where {N, B, P, T<:TangentBundle{N,B,P}}
    the_primal = convert(B, primal(tbun))
    the_partials = convert(P, tbun.tangent)
    return _TangentBundle(Val{N}(), the_primal, the_partials)
end

# StructureArrays helpers

expand_singleton_to_array(asize, a::AbstractZero) = fill(a, asize...)
expand_singleton_to_array(asize, a::AbstractArray) = a

function unbundle(atb::ExplicitTangentBundle{Order, A}) where {Order, Dim, T, A<:AbstractArray{T, Dim}}
    asize = size(atb.primal)
    StructArray{ExplicitTangentBundle{Order, T}}((atb.primal, map(a->expand_singleton_to_array(asize, a), atb.tangent.partials)...))
end

function StructArrays.staticschema(::Type{<:ExplicitTangentBundle{N, B, T}}) where {N, B, T}
    Tuple{B, T.parameters...}
end

function StructArrays.component(m::ExplicitTangentBundle{N, B, T}, i::Int) where {N, B, T}
    i == 1 && return m.primal
    return m.tangent.partials[i - 1]
end

function StructArrays.createinstance(T::Type{<:ExplicitTangentBundle}, args...)
    T(first(args), Base.tail(args))
end

function unbundle(atb::TaylorBundle{Order, A}) where {Order, Dim, T, A<:AbstractArray{T, Dim}}
    StructArray{TaylorBundle{Order, T}}((atb.primal, atb.tangent.coeffs...))
end

function StructArrays.staticschema(::Type{<:TaylorBundle{N, B, T}}) where {N, B, T}
    Tuple{B, T.parameters...}
end

function StructArrays.staticschema(::Type{<:TaylorBundle{N, B}}) where {N, B}
    Tuple{B, Vararg{Any, N}}
end

function StructArrays.component(m::TaylorBundle{N, B}, i::Int) where {N, B}
    i == 1 && return m.primal
    return m.tangent.coeffs[i - 1]
end

function StructArrays.createinstance(T::Type{<:TaylorBundle}, args...)
    T(first(args), Base.tail(args))
end

function unbundle(u::UniformBundle{N, A}) where {N,T,Dim,A<:AbstractArray{T, Dim}}
    StructArray{UniformBundle{N, T}}((u.primal, fill(u.tangent.val, size(u.primal)...)))
end

function ChainRulesCore.rrule(::typeof(unbundle), atb::AbstractTangentBundle)
    unbundle(atb), Δ->throw(Δ)
end

function StructArrays.createinstance(T::Type{<:UniformBundle}, args...)
    T(args[1], args[2])
end

function rebundle(A::AbstractArray{<:ExplicitTangentBundle{N}}) where {N}
    ExplicitTangentBundle{N}(
        map(x->x.primal, A),
        ntuple(2^N-1) do i
            map(x->x.tangent.partials[i], A)
        end)
end

function rebundle(A::AbstractArray{<:TaylorBundle{N}}) where {N}
    TaylorBundle{N}(
        map(x->x.primal, A),
        ntuple(N) do i
            map(x->x.tangent.coeffs[i], A)
        end)
end

function rebundle(A::AbstractArray{<:UniformBundle{N}}) where {N}
    @assert all(x->getfield(x, :tangent)==(first(A).tangent), A)
    UniformBundle{N}(map(x->x.primal, A), first(A).tangent.val)
end

function ChainRulesCore.rrule(::typeof(rebundle), atb)
    rebundle(atb), Δ->throw(Δ)
end
