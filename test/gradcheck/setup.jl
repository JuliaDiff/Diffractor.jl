module gradcheck_tests

# This file contains a selection of tests from Zygote's "gradcheck.jl",
# dealing with Base and standard library functions. Many of these use rules
# which have their own more exhaustive tests in ChainRules.

# Tests for packages (Distances, LogExpFunctions, AbstractFFTs, FillArrays) are not included.

# Ideally this would be extended to take `gradient` both forward and reverse,
# and `jacobicheck` including 2nd derivatives, for every testset. But not yet.

using Test
using Random
using ChainRulesCore
using Diffractor
using Distributed: CachingPool, pmap, workers
using FiniteDifferences
using LinearAlgebra

#####
##### Zygote/test/gradcheck.jl : setup
#####

n_grad(f, x::Real) = (central_fdm(5,1)(f,x),)
n_grad(f, x::AbstractArray{<:Real}) = FiniteDifferences.grad(central_fdm(5,1), f, float(x))
n_grad(f, xs::Vararg{Any,N}) where {N} = ntuple(N) do i
    n_grad(x -> f(ntuple(j -> j==i ? x : xs[j], N)...), xs[i])[1]
end

# check gradients via finite differencing
function gradcheck(f, xs::AbstractArray...)
    gs = unthunk.(gradient(f, xs...))
    all(isapprox.(gs, n_grad(f, xs...), rtol=1e-5, atol=1e-5))
end
gradcheck(f, dims...) = gradcheck(f, rand.(Float64, dims)...)
# @test gradcheck(sqrt, 3.14)  # given number
@test gradcheck(sum, randn(10))  # given array
@test gradcheck(dot, randn(3), rand(3))  # given multiple vectors
@test gradcheck(dot, 3, 3)  # given multiple random vectors

jacobicheck(f, xs::AbstractArray...) = f(xs...) isa Number ? gradcheck(f, xs...) : 
    gradcheck((xs...) -> sum(sin, f(xs...)), xs...)
jacobicheck(f, dims...) = jacobicheck(f, randn.(Float64, dims)...)
@test jacobicheck(identity, [1,2,3])  # one given array
@test jacobicheck(sum, [1,2,3])  # fallback to gradcheck
@test jacobicheck(identity, (4,5))  # one random matrix
@test jacobicheck(+, 3, 3)  # two random vectors

isZero(x) = x isa AbstractZero

function value_and_pullback(f, x...)
    y, b = Diffractor.∂⃖{1}()(f, x...)
    back(dy) = map(unthunk, Base.tail(b(dy)))
    y, back
end


include("base.jl")

end
