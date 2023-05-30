using Base: tail
using ChainRules
using ChainRulesCore

"""
    ∂⃖{N}

∂⃖{N} is the reverse-mode AD optic functor of order `N`. A call
`(::∂⃖{N})(f, args...)` corresponds to  ∂⃖ⁿ f(args...) in the linear encoding
of an N-optic (see the terminology guide for definitions of these terms).

In general `(::∂⃖{N})(f, args...)` will return a tuple of the original primal
value `f(args...)` (in rare cases primitives may modify the primal value -
in general we will ignore this rare complication for the purposes of clear
documentation) and an optic continuation `λ`. The interpretation of this
continuation depends on the order of the functor:

For example, ∂⃖{1} computes first derivatives. In particular, for a function `f`,
`∂⃖{1}(f, args...)` will return the tuple `(f(args...), f⋆)` (read "f upper-star").

"""
struct ∂⃖{N}; end
const ∂⃖¹ = ∂⃖{1}()

(::Type{∂⃖})(args...) = ∂⃖¹(args...)

"""
    ∂☆{N}

∂☆{N} is the forward-mode AD functor of order `N`. A call
`(::∂☆{N})(f, args...)` evaluating a function `f: A -> B` is lifted to its
pushforward on the N-th order tangent bundle `f⋆: Tⁿ A -> Tⁿ B`.
"""
struct ∂☆{N}; end
const ∂☆¹ = ∂☆{1}()

"""
    dx(x)

dx represents the trival differential one-form of a one dimensional
Riemannian manifold `M`. In particular, it is a section of the cotangent bundle
of `M`, meaning it may be evaluted at a point `x` of `M` to obtain an element
of the cotangent space `T*ₓ M` to `M` at `x`. We impose no restrictions on
the representations of either the manifold itself or the cotangent space.

By default, the only implementation provided identifies T*ₓ ℝ ≃ ℝ, keeping
watever type is used to represent ℝ. i.e.
```julia
dx(x::Real) = one(x)
```

However, users may provide additional overloads for custom representations of
one dimensional Riemannian manifolds.
"""
dx(x::Real) = one(x)
dx(::NoTangent) = NoTangent()
dx(::ZeroTangent) = ZeroTangent()
dx(x::Complex) = error("Tried to take the gradient of a complex-valued function.")
dx(x) = error("Cotangent space not defined for `$(typeof(x))`. Try a real-valued function.")

"""
    ∂xⁿ{N}(x)

For `x` in a one dimensional manifold, map x to the trivial, unital, Nth order
tangent bundle. It should hold that `∀x ⟨∂ⁿ{1}x(x), dx(x)⟩ = 1`
"""
struct ∂xⁿ{N}; end

(::∂xⁿ{N})(x::Real) where {N} = TaylorBundle{N}(x, ntuple(i->i==1 ? one(x) : zero(x), N))
(::∂xⁿ)(x) = error("Tangent space not defined for `$(typeof(x)).")

function ChainRules.rrule(∂::∂xⁿ, x)
    ∂(x), Δ->(NoTangent(), Δ.primal)
end

"""
    ∇(f, args...)

Computes the gradient ∇f(x, y, z...) (at (x, y, z...)). In particular, the
return value will be a tuple of partial derivatives `(∂f/∂x, ∂f/∂y, ∂f/∂z...)`.

# Curried version

Alternatively, ∇ may be curried, essentially giving the gradient as a function:

## Examples

```jldoctest
julia> using Diffractor: ∇

julia> map(∇(*), (1,2,3), (4,5,6))
((4.0, 1.0), (5.0, 2.0), (6.0, 3.0))
```

# The derivative ∂f/∂f

Note that since in Julia, there is no distinction between functions and values,
there is in principle a partial derivative with respect to the function itself.
However, said partial derivative is dropped by this interface. It is however
available using the lower level ∂⃖ if desired. This interaction can also be used
to obtain gradients with respect to only some of the arguments by using a
closure:

∇((x,z)->f(x,y,z))(x, z) # returns (∂f/∂x, ∂f/∂z)

Though of course the same can be obtained by simply indexing the resulting
tuple (in well-inferred code there should not be a performance difference
between these two options).
"""
struct ∇{T}
    f::T
end

function Base.show(io::IO, f::∇)
    print(io, '∇')
    print(io, f.f)
end

function (f::∇)(args...)
    y, f☆ = ∂⃖(getfield(f, :f), args...)
    return tail(f☆(dx(y)))
end

# N.B: This means the gradient is not available for zero-arg function, but such
# a gradient would be guaranteed to be `()`, which is a bit of a useless thing
function (::Type{∇})(f, x1, args...)
    unthunk.(∇(f)(x1, args...))
end

# Star Trek has their prime directive. We have the...
abstract type AbstractPrimeDerivative{N, T}; end

# Backwards first order derivative
struct PrimeDerivativeBack{N, T} <: AbstractPrimeDerivative{N, T}
    f::T
end
Base.show(io::IO, f::PrimeDerivativeBack{N}) where {N} = print(io, f.f, "'"^N)

# This improves performance for nested derivatives by short cutting some
# recursion into the PrimeDerivative constructor
lower_pd(f::PrimeDerivativeBack{N,T}) where {N,T} = PrimeDerivativeBack{N-1,T}(getfield(f, :f))
lower_pd(f::PrimeDerivativeBack{1}) = getfield(f, :f)
raise_pd(f::PrimeDerivativeBack{N,T}) where {N,T} = PrimeDerivativeBack{N+1,T}(getfield(f, :f))

ChainRulesCore.rrule(::typeof(lower_pd), f) = lower_pd(f), Δ->(ZeroTangent(), Δ)
ChainRulesCore.rrule(::typeof(raise_pd), f) = raise_pd(f), Δ->(ZeroTangent(), Δ)

PrimeDerivativeBack(f) = PrimeDerivativeBack{1, typeof(f)}(f)
PrimeDerivativeBack(f::PrimeDerivativeBack{N, T}) where {N, T} = raise_pd(f)

function (f::PrimeDerivativeBack)(x)
    z = ∂⃖¹(lower_pd(f), x)
    y = getfield(z, 1)
    f☆ = getfield(z, 2)
    return unthunk(getfield(f☆(dx(y)), 2))
end

# Forwards primal derivative
struct PrimeDerivativeFwd{N, T}
    f::T
end

PrimeDerivativeFwd(f) = PrimeDerivativeFwd{1, typeof(f)}(f)
PrimeDerivativeFwd(f::PrimeDerivativeFwd{N, T}) where {N, T} = raise_pd(f)

lower_pd(f::PrimeDerivativeFwd{N,T}) where {N,T} = (error(); PrimeDerivativeFwd{N-1,T}(getfield(f, :f)))
raise_pd(f::PrimeDerivativeFwd{N,T}) where {N,T} = PrimeDerivativeFwd{N+1,T}(getfield(f, :f))

(f::PrimeDerivativeFwd{0})(x) = getfield(f, :f)(x)
function (f::PrimeDerivativeFwd{N})(x) where N
    z = ∂☆{N}()(ZeroBundle{N}(getfield(f, :f)), ∂xⁿ{N}()(x))
    z[TaylorTangentIndex(N)]
end

derivative(f, x) = Diffractor.PrimeDerivativeFwd(f)(x)
function gradient(f, x::AbstractVector)
    map(eachindex(x)) do i
        derivative(ξ -> f(vcat(x[begin:i-1], ξ, x[i+1:end])), x[i])
    end
end
gradient(f, x::AbstractArray) = reshape(gradient(v -> f(reshape(v, size(x))), vec(x)), size(x))
gradient(f, xs...) = unthunk.(∇(f)(xs...))
jacobian(f, x::AbstractArray) = reduce(hcat, vec.(gradient(f, x)))
hessian(f, x::AbstractArray) = jacobian(y -> gradient(f, y), float(x))
