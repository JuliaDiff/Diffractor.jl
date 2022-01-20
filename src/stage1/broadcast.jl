using Base.Broadcast
using Base.Broadcast: broadcasted, Broadcasted

# Forward mode broadcast rule
struct FwdBroadcast{N, T<:AbstractTangentBundle{N}}
  f::T
end
(f::FwdBroadcast{N})(args::AbstractTangentBundle{N}...) where {N} = ∂☆{N}()(f.f, args...)

n_getfield(∂ₙ::∂☆{N}, b::ATB{N}, x::Union{Symbol, Int}) where {N} = ∂ₙ(ZeroBundle{N}(getfield), b, ZeroBundle{N}(x))

function (∂ₙ::∂☆{N})(zc::ZeroBundle{N, typeof(copy)},
                     bc::ATB{N, <:Broadcasted}) where {N}
  bc = ∂ₙ(ZeroBundle{N}(Broadcast.flatten), bc)
  args = n_getfield(∂ₙ, bc, :args)
  r = copy(Broadcasted(
      FwdMap(n_getfield(∂ₙ, bc, :f)),
      ntuple(length(primal(args))) do i
        val = n_getfield(∂ₙ, args, i)
        if ndims(primal(val)) == 0
          return Ref(∂ₙ(ZeroBundle{N}(getindex), val))
        else
          return unbundle(val)
        end
      end))
  if isa(r, AbstractArray)
    r = rebundle(r)
  end
  return r
end

# Reverse mode broadcast rules

using ChainRulesCore: derivatives_given_output

# Broadcast over one element is just map
function (∂⃖ₙ::∂⃖{N})(::typeof(broadcasted), f, a::Array) where {N}
    ∂⃖ₙ(map, f, a)
end

(::∂⃖{1})(::typeof(broadcasted), f, args...) = split_bc_rule(f, args...)
(::∂⃖{1})(::typeof(broadcasted), f, arg::Array) = split_bc_rule(f, arg) # ambiguity
function split_bc_rule(f::F, args::Vararg{Any,N}) where {F,N}
    T = Broadcast.combine_eltypes(f, args)
    TΔ = Core.Compiler._return_type(derivatives_given_output, Tuple{T, F, map(eltype, args)...})
    if T === Bool
        # Trivial case: non-differentiable output, e.g. `x .> 0`
        back_1(_) = ntuple(Returns(ZeroTangent()), length(args)+2)
        return f.(args...), back_1
    elseif T <: Number && isconcretetype(TΔ)
        # Fast path: just broadcast, and use x & y to find derivative.
        ys = f.(args...)
        function back_2_one(dys)  # For f.(x) we do not need StructArrays / unzip at all
            delta = broadcast(unthunk(dys), ys, args...) do dy, y, a
                das = only(derivatives_given_output(y, f, a))
                dy * conj(only(das))
            end
            (NoTangent(), NoTangent(), unbroadcast(only(args), delta))
        end
        function back_2_many(dys)
            deltas = splitcast(unthunk(dys), ys, args...) do dy, y, as...
                das = only(derivatives_given_output(y, f, as...))
                map(da -> dy * conj(da), das)
            end
            dargs = map(unbroadcast, args, deltas)  # ideally sum in unbroadcast could be part of splitcast?
            (NoTangent(), NoTangent(), dargs...)
        end
        return ys, N==1 ? back_2_one : back_2_many
    else
        # Slow path: collect all the pullbacks & apply them later.
        # (Since broadcast makes no guarantee about order of calls, and un-fusing 
        # can change the number of calls, this does not bother to try to reverse.)
        ys, backs = splitcast(∂⃖{1}(), f, args...)
        function back_3(dys)
            deltas = splitmap(backs, unthunk(dys)) do back, dy
                map(unthunk, back(dy))
            end
            dargs = map(unbroadcast, args, Base.tail(deltas))  # no real need to close over args here
            (NoTangent(), sum(first(deltas)), dargs...)
        end
        back_3(::AbstractZero) = (NoTangent(), map(Returns(ZeroTangent()), args)...)
        return ys, back_3
    end
end

# Skip AD'ing through the axis computation
function (::∂⃖{1})(::typeof(Base.Broadcast.instantiate), bc::Base.Broadcast.Broadcasted)
    uninstantiate(Δ) = Core.tuple(NoTangent(), Δ)
    return Base.Broadcast.instantiate(bc), uninstantiate
end

# This uses "multimap"-like constructs:
using StructArrays
splitmap(f, args...) = StructArrays.components(StructArray(Iterators.map(f, args...)))
splitcast(f, args...) = StructArrays.components(StructArray(Broadcast.instantiate(Broadcast.broadcasted(f, args...))))

# For certain cheap operations we can easily allow fused broadcast:

(::∂⃖{1})(::typeof(broadcasted), ::typeof(+), args...) = lazy_bc_plus(args...)
(::∂⃖{1})(::typeof(broadcasted), ::typeof(+), arg::Array) = lazy_bc_plus(arg) # ambiguity
function lazy_bc_plus(xs...) where {F}
    broadcasted(+, xs...), Δraw -> let Δ = unthunk(Δraw)
        (NoTangent(), NoTangent(), map(x -> unbroadcast(x, Δ), xs)...)
    end
end

(::∂⃖{1})(::typeof(copy), bc::Broadcast.Broadcasted) = copy(bc), Δ -> (NoTangent(), Δ)

function (::∂⃖{1})(::typeof(broadcasted), ::typeof(-), x, y)
    broadcasted(-, x, y), Δraw -> let Δ = unthunk(Δraw)
        (NoTangent(), NoTangent(), unbroadcast(x, Δ), -unbroadcast(y, Δ))
        # Ideally you could fuse the - into unbroadcast, mapreduce() not sum, when y is a smaller array
    end
end

using LinearAlgebra: dot
const Numeric{T<:Number} = Union{T, AbstractArray{T}}

function (::∂⃖{1})(::typeof(broadcasted), ::typeof(*), x::Numeric, y::Numeric)
    broadcasted(*, x, y), Δraw -> let Δ = unthunk(Δraw)
        dx = eltype(x)==Bool ? NoTangent() : x isa Number ? dot(y, Δ) : unbroadcast(x, Δ .* conj.(y))
        dy = eltype(y)==Bool ? NoTangent() : y isa Number ? dot(x, Δ) : unbroadcast(y, Δ .* conj.(x))
        # When x is an array but a smaller one, instead of dot you may be able to use mapreduce()
        (NoTangent(), NoTangent(), dx, dy)
    end
end

function (::∂⃖{1})(::typeof(broadcasted), ::typeof(Base.literal_pow), ::typeof(^), x, ::Val{2})
    broadcasted(*, x, x), Δ -> begin
        dx = unbroadcast(x, 2 .* unthunk(Δ) .* conj.(x))
        (NoTangent(), NoTangent(), NoTangent(), dx, NoTangent()) 
    end
end
function (::∂⃖{1})(::typeof(broadcasted), ::typeof(Base.literal_pow), ::typeof(^), x::Number, ::Val{2})
    x^2, Δ -> (NoTangent(), NoTangent(), NoTangent(), 2 * Δ * conj(x), NoTangent())
end

function (::∂⃖{1})(::typeof(broadcasted), ::typeof(/), x::Numeric, y::Number)
    z, back = ∂⃖{1}()(/, x, y)
    z, dz -> begin
        _, dx, dy = back(dz)
        (NoTangent(), NoTangent(), dx, dy)
    end
end

(::∂⃖{1})(::typeof(broadcasted), ::typeof(identity), x) = x, identity_pullback
(::∂⃖{1})(::typeof(broadcasted), ::typeof(identity), x::Array) = x, identity_pullback # ambiguity
identity_pullback(Δ) = (NoTangent(), NoTangent(), Δ)

(::∂⃖{1})(::typeof(broadcasted), ::typeof(conj), x::AbstractArray{Real}) = x, identity_pullback
(::∂⃖{1})(::typeof(broadcasted), ::typeof(conj), x::Array{Real}) = x, identity_pullback
(::∂⃖{1})(::typeof(broadcasted), ::typeof(conj), x) =
    broadcasted(conj, x), Δ -> (NoTangent(), conj(unthunk(Δ)))
(::∂⃖{1})(::typeof(broadcasted), ::typeof(conj), x::Array) =
    broadcasted(conj, x), Δ -> (NoTangent(), conj(unthunk(Δ)))

# All broadcasts use `unbroadcast` to reduce to correct shape:

function unbroadcast(x::Base.AbstractArrayOrBroadcasted, dx)
    N = ndims(dx)
    if length(x) == length(dx)
        ProjectTo(x)(dx)  # handles trivial reshapes, offsets, structured matrices, row vectors
    else
        dims = ntuple(d -> get(size(x), d, 1) == 1 ? d : N+1, N)  # awful hack to get type-stable `dims`
        ProjectTo(x)(sum(dx; dims))
    end
end
unbroadcast(x::Base.AbstractArrayOrBroadcasted, dx::AbstractZero) = dx

unbroadcast(x::T, dx) where {T<:Tuple{Any}} = ProjectTo(x)(Tangent{T}(sum(dx)))
function unbroadcast(x::T, dx) where {T<:Tuple{Vararg{Any,N}}} where {N}
    _print("unbroadcast tuple")
    val = if length(x) == length(dx)
        dx
    else
        sum(dx; dims=2:ndims(dx))
    end
    ProjectTo(x)(NTuple{length(x)}(val)) # Tangent
end

unbroadcast(f::Function, df) = sum(df)
unbroadcast(x::Number, dx) = ProjectTo(x)(sum(dx))
unbroadcast(x::Base.RefValue, dx) = ProjectTo(x)(Ref(sum(dx)))

unbroadcast(::Bool, dx) = NoTangent()
unbroadcast(::AbstractArray{Bool}, dx) = NoTangent()
unbroadcast(::AbstractArray{Bool}, ::NoTangent) = NoTangent()  # ambiguity
unbroadcast(::Val, dx) = NoTangent()

function unbroadcast(x, dx)
    p = ProjectTo(x)
    if dx isa AbstractZero || p isa ProjectTo{<:AbstractZero}
        return NoTangent()
    end
    b = Broadcast.broadcastable(x)
    if b isa Ref  # then x is scalar under broadcast
        return p(sum(dx))
    else
        error("don't know how to handle broadcast gradient for x::$(typeof(x))")
    end
end
