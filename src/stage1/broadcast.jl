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

# Broadcast over one element is just map
function (∂⃖ₙ::∂⃖{N})(::typeof(broadcasted), f, a::Array) where {N}
    ∂⃖ₙ(map, f, a)
end

# The below is from Zygote: TODO: DO we want to do something better here?

accum_sum(xs::Nothing; dims = :) = NoTangent()
accum_sum(xs::AbstractArray{Nothing}; dims = :) = NoTangent()
accum_sum(xs::AbstractArray{<:Number}; dims = :) = sum(xs, dims = dims)
accum_sum(xs::AbstractArray{<:AbstractArray{<:Number}}; dims = :) = sum(xs, dims = dims)
accum_sum(xs::Number; dims = :) = xs

# https://github.com/FluxML/Zygote.jl/issues/594
function Base.reducedim_init(::typeof(identity), ::typeof(accum), A::AbstractArray, region)
  Base.reducedim_initarray(A, region, NoTangent(), Union{Nothing,eltype(A)})
end

trim(x, Δ) = reshape(Δ, ntuple(i -> size(Δ, i), Val(ndims(x))))

unbroadcast(x::AbstractArray, x̄) =
  size(x) == size(x̄) ? x̄ :
  length(x) == length(x̄) ? trim(x, x̄) :
    trim(x, accum_sum(x̄, dims = ntuple(i -> size(x, i) == 1 ? i : ndims(x̄)+1, Val(ndims(x̄)))))

unbroadcast(x::Number, x̄) = accum_sum(x̄)
unbroadcast(x::Tuple{<:Any}, x̄) = (accum_sum(x̄),)
unbroadcast(x::Base.RefValue, x̄) = (x=accum_sum(x̄),)

unbroadcast(x::AbstractArray, x̄::Nothing) = NoTangent()

const Numeric = Union{Number, AbstractArray{<:Number, N} where N}

function ChainRulesCore.rrule(::DiffractorRuleConfig, ::typeof(broadcasted), ::typeof(+), xs::Numeric...)
    broadcast(+, xs...), ȳ -> (NoTangent(), NoTangent(), map(x -> unbroadcast(x, unthunk(ȳ)), xs)...)
end

ChainRulesCore.rrule(::DiffractorRuleConfig, ::typeof(broadcasted), ::typeof(-), x::Numeric, y::Numeric) = x .- y,
  Δ -> let Δ=unthunk(Δ); (NoTangent(), NoTangent(), unbroadcast(x, Δ), -unbroadcast(y, Δ)); end

ChainRulesCore.rrule(::DiffractorRuleConfig, ::typeof(broadcasted), ::typeof(*), x::Numeric, y::Numeric) = x.*y,
  z̄ -> let z̄=unthunk(z̄); (NoTangent(), NoTangent(), unbroadcast(x, z̄ .* conj.(y)), unbroadcast(y, z̄ .* conj.(x))); end