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

using ChainRulesCore: derivatives_given_output

(::∂⃖{1})(::typeof(broadcasted), f, args...) = split_bc_rule(f, args...)
(::∂⃖{1})(::typeof(broadcasted), f, arg::Array) = split_bc_rule(f, arg) # ambiguity
function split_bc_rule(f::F, args...) where {F}
    T = Broadcast.combine_eltypes(f, args)
    if T == Bool && Base.issingletontype(F)
        # Trivial case
        back_1(_) = ntuple(Returns(ZeroTangent()), length(args)+2)
        return f.(args...), back_1
    # elseif all(a -> a isa Numeric, args) && isconcretetype(Core.Compiler._return_type(
    elseif isconcretetype(Core.Compiler._return_type(
            derivatives_given_output, Tuple{T, F, map(eltype, args)...}))
        # Fast path: just broadcast, and use x & y to find derivative.
        ys = f.(args...)
        # println("2")
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
        # println("3")
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

unbroadcast(f::Function, x̄) = accum_sum(x̄)
unbroadcast(::Val, _) = NoTangent()
unbroadcast(x::AbstractArray, x̄::NoTangent) = NoTangent()
accum_sum(xs::AbstractArray{<:NoTangent}; dims = :) = NoTangent()

#=

julia> xs = randn(10_000);
julia> @btime Zygote.gradient(x -> sum(abs2, x), $xs)
  4.744 μs (2 allocations: 78.17 KiB)
julia> @btime Diffractor.unthunk.(gradient(x -> sum(abs2, x), $xs));
  3.307 μs (2 allocations: 78.17 KiB)

# Simple function

julia> @btime Zygote.gradient(x -> sum(abs2, exp.(x)), $xs);
  72.541 μs (29 allocations: 391.47 KiB)     # with dual numbers -- like 4 copies

julia> @btime gradient(x -> sum(abs2, exp.(x)), $xs);
  45.875 μs (36 allocations: 235.47 KiB)  # fast path -- one copy forward, one back
  44.042 μs (32 allocations: 313.48 KiB)  # slow path -- 3 copies, extra is closure? 
  61.167 μs (12 allocations: 703.41 KiB)  # with `map` rule as before -- worse


# Composed function, Zygote struggles

julia> @btime Zygote.gradient(x -> sum(abs2, (identity∘cbrt).(x)), $xs);
  97.167 μs (29 allocations: 391.61 KiB)     # with dual numbers (Zygote master)
  93.238 ms (849567 allocations: 19.22 MiB)  # without, thus Zygote.pullback

julia> @btime gradient(x -> sum(abs2, (identity∘cbrt).(x)), $xs);
  55.290 ms (830060 allocations: 49.75 MiB)  # slow path
  14.747 ms (240043 allocations: 7.25 MiB)   # with `map` rule as before -- better!

# Compare unfused

julia> @btime gradient(x -> sum(abs2, identity.(cbrt.(x))), $xs);
  69.458 μs (50 allocations: 392.09 KiB)  # fast path -- two copies forward, two back
  75.041 μs (46 allocations: 470.11 KiB)  # slow path -- 5 copies
  135.541 μs (27 allocations: 1.30 MiB)   # with `map` rule as before -- worse


# Lazy +,-,* for partial fusing

julia> @btime Zygote.gradient(x -> sum(abs2, exp.(2 .* x .- 100)), $xs);
  81.250 μs (21 allocations: 625.47 KiB)  # special rules + dual numbers, 4 more copies than best

julia> @btime gradient(x -> sum(abs2, exp.(2 .* x .- 100)), $xs);
  57.166 μs (49 allocations: 470.22 KiB)  # broadcast in *, -
  54.583 μs (46 allocations: 314.06 KiB)  # broadcasted -- two less copies
  72.958 μs (26 allocations: 1016.38 KiB) # with `map` rule as before

julia> gradient((x,y) -> sum(abs2, exp.(2 .* x .+ y)), xs, (rand(10)'))
ERROR: MethodError: no method matching size(::Base.Broadcast.Broadcasted # hmm

julia> @btime gradient((x,y) -> sum(abs2, exp.(x .+ y)), $xs, $(rand(100)'));
  7.127 ms (75 allocations: 22.97 MiB)   # after
  12.956 ms (57 allocations: 76.37 MiB)  # before

ulia> @btime Zygote.gradient((x,y) -> sum(abs2, exp.(x .+ y)), $xs, $(rand(100)'));
  9.937 ms (48 allocations: 45.86 MiB)

=#

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

unbroadcast(x::Union{AbstractArray, Base.Broadcast.Broadcasted}, x̄) =
  size(x) == size(x̄) ? x̄ :
  length(x) == length(x̄) ? trim(x, x̄) :
    trim(x, accum_sum(x̄, dims = ntuple(i -> size(x, i) == 1 ? i : ndims(x̄)+1, Val(ndims(x̄)))))

unbroadcast(x::Number, x̄) = accum_sum(x̄)
unbroadcast(x::Tuple{<:Any}, x̄) = (accum_sum(x̄),)
unbroadcast(x::Base.RefValue, x̄) = (x=accum_sum(x̄),)

unbroadcast(x::AbstractArray, x̄::Nothing) = NoTangent()

const Numeric = Union{Number, AbstractArray{<:Number, N} where N}

# function ChainRulesCore.rrule(::typeof(broadcasted), ::typeof(+), xs::Numeric...)
#     broadcast(+, xs...), ȳ -> (NoTangent(), NoTangent(), map(x -> unbroadcast(x, unthunk(ȳ)), xs)...)
# end

# Replace Zygote-like fully split broadcasting with one fused over easy operations
(::∂⃖{1})(::typeof(broadcasted), ::typeof(+), args...) = split_bc_plus(args...)
(::∂⃖{1})(::typeof(broadcasted), ::typeof(+), arg::Array) = split_bc_plus(arg) # ambiguity
function split_bc_plus(xs...) where {F}
    broadcasted(+, xs...), Δ -> let Δun = unthunk(Δ)
        # println("+")
        (NoTangent(), NoTangent(), map(x -> unbroadcast(x, Δun), xs)...)
    end
end
Base.eltype(bc::Broadcast.Broadcasted{<:Any, <:Any, typeof(+), <:Tuple}) = 
    mapreduce(eltype, promote_type, bc.args)  # needed to hit fast path

(::∂⃖{1})(::typeof(copy), bc::Broadcast.Broadcasted) = copy(bc), Δ -> (NoTangent(), Δ)

# ChainRulesCore.rrule(::typeof(broadcasted), ::typeof(-), x::Numeric, y::Numeric) = x .- y,
#   Δ -> let Δ=unthunk(Δ); (NoTangent(), NoTangent(), unbroadcast(x, Δ), -unbroadcast(y, Δ)); end

function (::∂⃖{1})(::typeof(broadcasted), ::typeof(-), x, y)
    broadcasted(-, x, y), Δ -> let Δun = unthunk(Δ)
        # println("-")
        (NoTangent(), NoTangent(), unbroadcast(x, Δun), -unbroadcast(y, Δun))
        # Ideally you could fuse the - into unbroadcast, mapreduce() not sum, when y is a smaller array
    end
end

# ChainRulesCore.rrule(::typeof(broadcasted), ::typeof(*), x::Numeric, y::Numeric) = x.*y,
#   z̄ -> let z̄=unthunk(z̄); (NoTangent(), NoTangent(), unbroadcast(x, z̄ .* conj.(y)), unbroadcast(y, z̄ .* conj.(x))); end

using LinearAlgebra: dot

function (::∂⃖{1})(::typeof(broadcasted), ::typeof(*), x, y)
    broadcasted(*, x, y), Δ -> let Δun = unthunk(Δ)
        # println("*")
        dx = x isa Number ? dot(y, Δun) : unbroadcast(x, Δun .* conj.(y))
        dy = y isa Number ? dot(x, Δun) : unbroadcast(y, Δun .* conj.(x))
        # When x is an array but a smaller one, instead of dot you may be able to use mapreduce()
        (NoTangent(), NoTangent(), dx, dy)
    end
end
