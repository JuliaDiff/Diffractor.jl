using Base.Broadcast
using Base.Broadcast: broadcasted, Broadcasted

# Forward mode broadcast rule
struct FwdBroadcast{N, E, T<:AbstractTangentBundle{N}}
  f::T
end
FwdItFwdBroadcastrate{E}(f::T) where {N, E, T<:AbstractTangentBundle{N}} = FwdBroadcast{N,E,T}(f)

(f::FwdBroadcast{N,E})(args::AbstractTangentBundle{N}...) where {N,E} = ∂☆{N,E}()(f.f, args...)

n_getfield(∂ₙ::∂☆{N}, b::ATB{N}, x::Union{Symbol, Int}) where {N} = ∂ₙ(ZeroBundle{N}(getfield), b, ZeroBundle{N}(x))

function (∂ₙ::∂☆{N,E})(zc::AbstractZeroBundle{N, typeof(copy)},
                     bc::ATB{N, <:Broadcasted}) where {N,E}
  bc = ∂ₙ(ZeroBundle{N}(Broadcast.flatten), bc)
  args = n_getfield(∂ₙ, bc, :args)
  r = copy(Broadcasted(
      FwdMap{E}(n_getfield(∂ₙ, bc, :f)),
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
