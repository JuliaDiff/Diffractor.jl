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

_print(s) = nothing
# _print(s) = printstyled(s, "\n"; color=:magenta)

# Broadcast over one element is just map
function (∂⃖ₙ::∂⃖{N})(::typeof(broadcasted), f, a::Array) where {N}
    _print("path 0")
    ∂⃖ₙ(map, f, a)
end

using ChainRulesCore: derivatives_given_output

(::∂⃖{1})(::typeof(broadcasted), f, args...) = split_bc_rule(f, args...)
(::∂⃖{1})(::typeof(broadcasted), f, arg::Array) = split_bc_rule(f, arg) # ambiguity
function split_bc_rule(f::F, args...) where {F}
    T = Broadcast.combine_eltypes(f, args)
    if T == Bool
        # Trivial case
        _print("path 1")
        back_1(_) = ntuple(Returns(ZeroTangent()), length(args)+2)
        return f.(args...), back_1
    elseif isconcretetype(Core.Compiler._return_type(
            derivatives_given_output, Tuple{T, F, map(eltype, args)...}))
        # Fast path: just broadcast, and use x & y to find derivative.
        ys = f.(args...)
        _print("path 2")
        function back_2(dys)
            deltas = splitcast(unthunk(dys), ys, args...) do dy, y, as...
                das = only(derivatives_given_output(y, f, as...))
                map(da -> dy * conj(da), das)
            end
            dargs = map(unbroadcast, args, deltas)
            (NoTangent(), NoTangent(), dargs...)
        end
        return ys, back_2
    else
        # Slow path: collect all the pullbacks & apply them later.
        _print("path 3")
        ys, backs = splitcast(rrule_via_ad, DiffractorRuleConfig(), f, args...)
        function back_3(dys)
            deltas = splitmap(backs, unthunk(dys)) do back, dy
                map(unthunk, back(dy))
            end
            dargs = map(unbroadcast, args, Base.tail(deltas))  # no real need to close over args here
            (NoTangent(), sum(first(deltas)), dargs...)
        end
        return ys, back_3
    end
end

using StructArrays
splitmap(f, args...) = StructArrays.components(StructArray(Iterators.map(f, args...))) # warning: splitmap(identity, [1,2,3,4]) === NamedTuple()
splitcast(f, args...) = StructArrays.components(StructArray(Broadcast.instantiate(Broadcast.broadcasted(f, args...))))

# For certain cheap operations we can easily allow fused broadcast:

(::∂⃖{1})(::typeof(broadcasted), ::typeof(+), args...) = split_bc_plus(args...)
(::∂⃖{1})(::typeof(broadcasted), ::typeof(+), arg::Array) = split_bc_plus(arg) # ambiguity
function split_bc_plus(xs...) where {F}
    broadcasted(+, xs...), Δ -> let Δun = unthunk(Δ)
        _print("broadcast +")
        (NoTangent(), NoTangent(), map(x -> unbroadcast(x, Δun), xs)...)
    end
end
Base.eltype(bc::Broadcast.Broadcasted{<:Any, <:Any, typeof(+), <:Tuple}) = 
    mapreduce(eltype, promote_type, bc.args)  # needed to hit fast path

(::∂⃖{1})(::typeof(copy), bc::Broadcast.Broadcasted) = copy(bc), Δ -> (NoTangent(), Δ)

function (::∂⃖{1})(::typeof(broadcasted), ::typeof(-), x, y)
    broadcasted(-, x, y), Δ -> let Δun = unthunk(Δ)
        _print("broadcast -")
        (NoTangent(), NoTangent(), unbroadcast(x, Δun), -unbroadcast(y, Δun))
        # Ideally you could fuse the - into unbroadcast, mapreduce() not sum, when y is a smaller array
    end
end

using LinearAlgebra: dot

function (::∂⃖{1})(::typeof(broadcasted), ::typeof(*), x, y)
    broadcasted(*, x, y), Δ -> let Δun = unthunk(Δ)
        _print("broadcast *")
        dx = eltype(x)==Bool ? NoTangent() : x isa Number ? dot(y, Δun) : unbroadcast(x, Δun .* conj.(y))
        dy = eltype(y)==Bool ? NoTangent() : y isa Number ? dot(x, Δun) : unbroadcast(y, Δun .* conj.(x))
        # When x is an array but a smaller one, instead of dot you may be able to use mapreduce()
        # Will things like this work? Ref([1,2]) .* [1,2,3]
        (NoTangent(), NoTangent(), dx, dy)
    end
end

(::∂⃖{1})(::typeof(broadcasted), ::typeof(conj), x) =
    broadcasted(conj, x), Δ -> (NoTangent(), conj(unthunk(Δ)))
(::∂⃖{1})(::typeof(broadcasted), ::typeof(conj), x::AbstractArray{Real}) =
    x, Δ -> (NoTangent(), Δ)

function unbroadcast(x::Base.AbstractArrayOrBroadcasted, dx)
    N = ndims(dx)
    if length(x) == length(dx)
        ProjectTo(x)(dx)  # handles trivial reshapes, offsets, structured matrices, row vectors
    else
        # This is an awful hack to get type-stable `dims`
        dims = ntuple(d -> get(size(x), d, 1) == 1 ? d : N+1, N)
        ProjectTo(x)(sum(dx; dims))
    end
end
unbroadcast(x::Base.AbstractArrayOrBroadcasted, dx::NoTangent) = NoTangent()

unbroadcast(x::Number, dx) = ProjectTo(x)(sum(dx))
unbroadcast(f::Function, df) = ProjectTo(x)(sum(df))
unbroadcast(x::Base.RefValue, dx) = ProjectTo(x)(Ref(sum(dx)))

unbroadcast(::Bool, dx) = NoTangent()
unbroadcast(::AbstractArray{Bool}, dx) = NoTangent()
unbroadcast(::AbstractArray{Bool}, ::NoTangent) = NoTangent()  # ambiguity
unbroadcast(::Val, dx) = NoTangent()
# Maybe more non-diff types? Some fallback?

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
