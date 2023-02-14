"""
    abstract type TangentBundle{N, B}; end

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

function Base.getindex(a::AbstractTangentBundle, b::TaylorTangentIndex)
    error("$(typeof(a)) is not taylor-like. Taylor indexing is ambiguous")
end

abstract type AbstractTangentSpace; end

struct TaylorTangent{C <: Tuple} <: AbstractTangentSpace
    coeffs::C
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

"""
    struct TangentBundle{N, B, P}

Represents a tangent bundle as an explicit primal together
with some representation of (potentially a product of) the tangent space.
"""
struct TangentBundle{N, B, P <: AbstractTangentSpace} <: AbstractTangentBundle{N, B}
    primal::B
    tangent::P
    TangentBundle{N}(B, P) where {N} = new{N, typeof(B), typeof(P)}(B,P)
end

check_tangent_invariant(lp, N) = @assert lp == 2^N - 1
@ChainRulesCore.non_differentiable check_tangent_invariant(lp, N)

const TaylorBundle{N, B, P} = TangentBundle{N, B, TaylorTangent{P}}

function TaylorBundle{N, B}(primal::B, coeffs) where {N, B}
    check_taylor_invariants(coeffs, primal, N)
    TangentBundle{N}(primal, TaylorTangent(coeffs))
end

function check_taylor_invariants(coeffs, primal, N)
    @assert length(coeffs) == N
    if isa(primal, TangentBundle)
        @assert isa(coeffs[1], TangentBundle)
    end
end
@ChainRulesCore.non_differentiable check_taylor_invariants(coeffs, primal, N)

function TaylorBundle{N}(primal, coeffs) where {N}
    TangentBundle{N}(primal, TaylorTangent(coeffs))
end

function Base.show(io::IO, x::TaylorBundle{1})
    print(io, x.primal)
    print(io, " + ")
    x = x.tangent
    print(io, x.coeffs[1], " ∂₁")
end

Base.getindex(tb::TaylorBundle, tti::TaylorTangentIndex) = tb.tangent.coeffs[tti.i]
function Base.getindex(tb::TaylorBundle, tti::CanonicalTangentIndex)
    tb.tangent.coeffs[count_ones(tti.i)]
end

const UniformBundle{N, B, U} = TangentBundle{N, B, UniformTangent{U}}
UniformBundle{N, B, U}(primal::B, partial::U) where {N,B,U} = TangentBundle{N}(primal, UniformTangent{U}(partial))
UniformBundle{N, B, U}(primal::B) where {N,B,U} = TangentBundle{N}(primal, UniformTangent{U}(U.instance))
UniformBundle{N, B}(primal::B, partial::U) where {N,B,U} = TangentBundle{N}(primal, UniformTangent{U}(partial))
UniformBundle{N}(primal, partial::U) where {N,U} = TangentBundle{N}(primal, UniformTangent{U}(partial))
UniformBundle{N, <:Any, U}(primal, partial::U) where {N, U} = TangentBundle{N}(primal, UniformTangent{U}(U.instance))
UniformBundle{N, <:Any, U}(primal) where {N, U} = TangentBundle{N}(primal, UniformTangent{U}(U.instance))

const ZeroBundle{N, B} = UniformBundle{N, B, ZeroTangent}
const DNEBundle{N, B} = UniformBundle{N, B, NoTangent}

Base.getindex(u::UniformBundle, ::TaylorTangentIndex) = u.tangent.val

"""
    TupleTangentBundle{N, B <: Tuple}

Represents the tagent bundle where the base space is some tuple type.
Mathematically, this tangent bundle is the product bundle of the individual
element bundles.
"""
struct CompositeBundle{N, B, T<:Tuple{Vararg{AbstractTangentBundle{N}}}} <: AbstractTangentBundle{N, B}
    tup::T
end
CompositeBundle{N, B}(tup::T) where {N, B, T} = CompositeBundle{N, B, T}(tup)

function Base.getindex(tb::CompositeBundle{N, B} where N, tti::TaylorTangentIndex) where {B}
    B <: SArray && error()
    Tangent{B}(map(tb.tup) do el
        el[tti]
    end...)
end


primal(b::CompositeBundle{N, <:Tuple} where N) = map(primal, b.tup)
function primal(b::CompositeBundle{N, T} where N) where T<:CompositeBundle
    T(map(primal, b.tup)...)
end
@generated primal(b::CompositeBundle{N, B} where N) where {B} =
    quote
        x = map(primal, b.tup)
        $(Expr(:splatnew, B, :x))
    end

expand_singleton_to_array(asize, a::AbstractZero) = fill(a, asize...)
expand_singleton_to_array(asize, a::AbstractArray) = a

function unbundle(atb::TaylorBundle{Order, A}) where {Order, Dim, T, A<:AbstractArray{T, Dim}}
    StructArray{TaylorBundle{Order, T}}((atb.primal, atb.tangent.coeffs...))
end

function ChainRulesCore.rrule(::typeof(unbundle), atb::TaylorBundle)
    unbundle(atb), Δ->throw(Δ)
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

function unbundle(zb::ZeroBundle{N, A}) where {N,T,Dim,A<:AbstractArray{T, Dim}}
    StructArray{ZeroBundle{N, T}}((zb.primal, fill(zb.tangent.val, size(zb.primal)...)))
end

function ChainRulesCore.rrule(::typeof(unbundle), atb::ZeroBundle)
    unbundle(atb), Δ->throw(Δ)
end

function StructArrays.createinstance(T::Type{<:ZeroBundle}, args...)
    T(args[1], args[2])
end

function rebundle(A::AbstractArray{<:TaylorBundle{N}}) where {N}
    TaylorBundle{N}(
        map(x->x.primal, A),
        ntuple(N) do i
            map(x->x.tangent.coeffs[i], A)
        end)
end

function ChainRulesCore.rrule(::typeof(rebundle), atb)
    rebundle(atb), Δ->throw(Δ)
end
