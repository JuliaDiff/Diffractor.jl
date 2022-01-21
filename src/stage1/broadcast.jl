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
# function (∂⃖ₙ::∂⃖{N})(::typeof(broadcasted), f, a::Array) where {N}
#     ∂⃖ₙ(map, f, a)
# end

(::∂⃖{1})(::typeof(copy), bc::Broadcast.Broadcasted) = copy(bc), Δ -> (NoTangent(), Δ)

(::∂⃖{1})(::typeof(broadcasted), f::F, args...) where {F} = split_bc_rule(f, args...)
# (::∂⃖{1})(::typeof(broadcasted), f::F, arg::Array) where {F} = split_bc_rule(f, arg) # ambiguity
function split_bc_rule(f::F, args::Vararg{Any,N}) where {F,N}
    T = Broadcast.combine_eltypes(f, args)
    TΔ = Core.Compiler._return_type(derivatives_given_output, Tuple{T, F, map(eltype, args)...})
    if T === Bool
        # Trivial case: non-differentiable output, e.g. `x .> 0`
        back_1(_) = ntuple(Returns(ZeroTangent()), length(args)+2)
        return f.(args...), back_1
    elseif T <: Number && isconcretetype(TΔ)
        # Fast path: just broadcast, and use arguments & result to find derivatives.
        ys = f.(args...)
        function back_2_one(dys)  # For f.(x) we do not need StructArrays / unzip at all
            delta = broadcast(unthunk(dys), ys, args...) do dy, y, a
                das = only(derivatives_given_output(y, f, a))
                dy * conj(only(das))  # possibly this * should be made nan-safe.
            end
            (NoTangent(), NoTangent(), unbroadcast(only(args), delta))
        end
        function back_2_many(dys)
            deltas = tuplecast(unthunk(dys), ys, args...) do dy, y, as...
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
        ys3, backs = tuplecast(∂⃖{1}(), f, args...)
        function back_3(dys)
            deltas = tuplecast(backs, unthunk(dys)) do back, dy  # could be map, sizes match
                map(unthunk, back(dy))
            end
            dargs = map(unbroadcast, args, Base.tail(deltas))
            (NoTangent(), sum(first(deltas)), dargs...)
        end
        back_3(::AbstractZero) = (NoTangent(), map(Returns(ZeroTangent()), args)...)
        return ys3, back_3
    end
end

# Don't run broadcasting on scalars
function split_bc_rule(f::F, args::Number...) where {F}
    z, back = ∂⃖{1}()(f, args...)
    z, dz -> (NoTangent(), back(dz)...)
end

split_bc_rule(::typeof(identity), x) = x, Δ -> (NoTangent(), NoTangent(), Δ)
split_bc_rule(::typeof(identity), x::Number) = x, Δ -> (NoTangent(), NoTangent(), Δ)

# Skip AD'ing through the axis computation
function (::∂⃖{1})(::typeof(Base.Broadcast.instantiate), bc::Base.Broadcast.Broadcasted)
    uninstantiate(Δ) = Core.tuple(NoTangent(), Δ)
    return Base.Broadcast.instantiate(bc), uninstantiate
end

using StructArrays

function tuplecast(f::F, args...) where {F}
    T = Broadcast.combine_eltypes(f, args)
    if isconcretetype(T)
        T <: Tuple || throw(ArgumentError("tuplecast(f, args) only works on functions returning a tuple."))
    end
    bc = Broadcast.instantiate(Broadcast.broadcasted(f, args...))
    StructArrays.components(StructArray(bc))
end

# For certain cheap operations we can easily allow fused broadcast:
const NumericOrBroadcast = Union{Number, AbstractArray{<:Number}, NTuple{<:Any,Number}, Broadcast.Broadcasted}

(::∂⃖{1})(::typeof(broadcasted), ::typeof(+), args::NumericOrBroadcast...) = lazy_bc_plus(args...)
(::∂⃖{1})(::typeof(broadcasted), ::typeof(+), args::Number) = split_bc_rule(+, args...)
function lazy_bc_plus(xs...) where {F}
    broadcasted(+, xs...), Δraw -> let Δ = unthunk(Δraw)
        (NoTangent(), NoTangent(), map(x -> unbroadcast(x, Δ), xs)...)
    end
end

(::∂⃖{1})(::typeof(broadcasted), ::typeof(-), x::Number, y::Number) = split_bc_rule(-, x, y)
function (::∂⃖{1})(::typeof(broadcasted), ::typeof(-), x::NumericOrBroadcast, y::NumericOrBroadcast)
    broadcasted(-, x, y), Δraw -> let Δ = unthunk(Δraw)
        (NoTangent(), NoTangent(), unbroadcast(x, Δ), -unbroadcast(y, Δ))
    end
end

using LinearAlgebra: dot

(::∂⃖{1})(::typeof(broadcasted), ::typeof(*), x::Number, y::Number) = split_bc_rule(*, x, y)
function (::∂⃖{1})(::typeof(broadcasted), ::typeof(*), x::NumericOrBroadcast, y::NumericOrBroadcast)
    broadcasted(*, x, y), Δraw -> let Δ = unthunk(Δraw)
        (NoTangent(), NoTangent(), _back_star(x, y, Δ), _back_star(y, x, Δ))
    end
end
_back_star(x, y, Δ) = unbroadcast(x, Δ .* conj.(y))
_back_star(x::Number, y, Δ) = dot(y, Δ)
_back_star(x::Bool, y, Δ) = NoTangent()

function (::∂⃖{1})(::typeof(broadcasted), ::typeof(Base.literal_pow), ::typeof(^), x::NumericOrBroadcast, ::Val{2})
    broadcasted(*, x, x), Δ -> begin
        dx = unbroadcast(x, 2 .* unthunk(Δ) .* conj.(x))
        (NoTangent(), NoTangent(), NoTangent(), dx, NoTangent()) 
    end
end
function (::∂⃖{1})(::typeof(broadcasted), ::typeof(Base.literal_pow), ::typeof(^), x::Number, ::Val{2})
    x^2, Δ -> (NoTangent(), NoTangent(), NoTangent(), 2 * Δ * conj(x), NoTangent())
end

(::∂⃖{1})(::typeof(broadcasted), ::typeof(/), x::Number, y::Number) = split_bc_rule(/, x, y)
function (::∂⃖{1})(::typeof(broadcasted), ::typeof(/), x::NumericOrBroadcast, y::Number)
    z = broadcast(/, x, y)
    z, Δth -> let Δ = unthunk(Δth)
        dx = unbroadcast(x, Δ ./ conj.(y))
        dy = -dot(z, Δ) / (conj(y))  # the reason to be eager is to allow dot here
        (NoTangent(), NoTangent(), dx, dy)
    end
end

(::∂⃖{1})(::typeof(broadcasted), ::typeof(identity), x) = split_bc_rule(identity, x)
# (::∂⃖{1})(::typeof(broadcasted), ::typeof(identity), x::Array) = split_bc_rule(identity, x) # ambiguity

(::∂⃖{1})(::typeof(broadcasted), ::typeof(conj), x::AbstractArray{Real}) = split_bc_rule(identity, x)
# (::∂⃖{1})(::typeof(broadcasted), ::typeof(conj), x::Array{Real}) = split_bc_rule(identity, x)  # ambiguity
(::∂⃖{1})(::typeof(broadcasted), ::typeof(conj), x) =
    broadcasted(conj, x), Δ -> (NoTangent(), conj(unthunk(Δ)))
(::∂⃖{1})(::typeof(broadcasted), ::typeof(conj), x::Array) =
    broadcasted(conj, x), Δ -> (NoTangent(), conj(unthunk(Δ)))

# Reverse mode broadcasting uses `unbroadcast` to reduce to correct shape:
function unbroadcast(x::Base.AbstractArrayOrBroadcasted, dx)
    N = ndims(dx)
    if length(x) == length(dx)
        ProjectTo(x)(dx)  # handles trivial reshapes, offsets, structured matrices, row vectors
    else
        dims = ntuple(d -> get(size(x), d, 1) == 1 ? d : N+1, N)  # hack to get type-stable `dims`
        ProjectTo(x)(sum(dx; dims))  # ideally this sum might be thunked?
    end
end
unbroadcast(x::Base.AbstractArrayOrBroadcasted, dx::AbstractZero) = dx

unbroadcast(x::T, dx) where {T<:Tuple{Any}} = ProjectTo(x)(Tangent{T}(sum(dx)))
function unbroadcast(x::T, dx) where {T<:Tuple{Vararg{Any,N}}} where {N}
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
