using Diffractor
using Diffractor: var"'", ∂⃖
using ForwardDiff
using StaticArrays
using Random
using Test

nfan() = 1, 1 # fan_in, fan_out
nfan(n) = 1, n # A vector is treated as a n×1 matrix
nfan(n_out, n_in) = n_in, n_out # In case of Dense kernels: arranged as matrices
nfan(dims::Tuple) = nfan(dims...)
nfan(dims...) = prod(dims[1:end-2]) .* (dims[end-1], dims[end]) # In case of convolution kernels
glorot_uniform(rng::AbstractRNG, dims...) = (rand(rng, Float64, dims...) .- 0.5) .* sqrt(24.0 / sum(nfan(dims...)))
glorot_uniform(dims...) = glorot_uniform(Random.GLOBAL_RNG, dims...)

struct Chain{T <: Tuple}
    layers::T
end
Chain(layers...) = Chain(layers)

@generated function apply_chain(c::T, x) where {T}
    foldl(1:length(T.parameters), init=:x) do b, i
        :(c[$i]($b))
    end
end

(c::Chain)(x) = apply_chain(c.layers, x)
struct Dense{F,S<:AbstractArray,T<:AbstractArray}
    W::S
    b::T
    σ::F
end

function Dense(in::Integer, out::Integer, σ=identity)
    Dense(glorot_uniform(out, in), glorot_uniform(out), σ)
end
function (a::Dense)(x::AbstractArray)
 W, b, σ = a.W, a.b, a.σ
 z = map(+, W*x, b)
 map(σ, z)
end
#g(NNODE,t,x,y) = ((((t*(1-x))*x)*(1-y))*y)*NNODE(@SVector [t,x,y]) + sin(2π*y)*sin(2π*x)
g(NNODE, t, x, y) = NNODE(@SVector [t,x,y])
loss(NNODE, at=0.5) = (x->g(NNODE, -0.1, 0.1, x))''(at)
let var"'" = Diffractor.PrimeDerivativeFwd
    global loss_fwd_diff
    loss_fwd_diff(NNODE, at=0.5) = (x->g(NNODE, -0.1, 0.1, x))''(at)
end
loss_fwd(NNODE, at=0.5) = ForwardDiff.derivative(x->ForwardDiff.derivative(x->g(NNODE, -0.1, 0.1, x), x), at)
NNODE = Chain(Dense(3,256,tanh),
           Dense(256,256,tanh),
           Dense(256,256,tanh),
           Dense(256,1, identity),x->x[1])
# Don't fall over on this semi-complicated nested AD case
training_step(NNODE) = gradient(NNODE->loss(NNODE), NNODE)

@test loss(NNODE, 0.1) ≈ loss_fwd(NNODE, 0.1)
@test loss(NNODE, 0.5) ≈ loss_fwd(NNODE, 0.5)
@test loss(NNODE, 0.1) ≈ loss_fwd_diff(NNODE, 0.1)
@test loss(NNODE, 0.5) ≈ loss_fwd_diff(NNODE, 0.5)

# How to test that this is actually the right answer?
training_step(NNODE)
#gradient(NNODE->loss_fwd_diff(NNODE), NNODE)
