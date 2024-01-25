# This file contains a selection of tests from Zygote's "gradcheck.jl",
# dealing with Base and standard library functions. Many of these use rules
# which have their own more exhaustive tests in ChainRules.

# Tests for packages (Distances, LogExpFunctions, AbstractFFTs, FillArrays) are not included.

# Ideally this would be extended to take `gradient` both forward and reverse,
# and `jacobicheck` including 2nd derivatives, for every testset. But not yet.

using Test
using ChainRulesCore
using Diffractor
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


#####
##### Zygote/test/gradcheck.jl : Base
#####

@testset "power" begin
    @test gradient(x -> x^2, -2) == (-4,) # literal_pow
    @test gradient(x -> x^10, -1.0) == (-10,)
    _pow = 10
    @test gradient(x -> x^_pow, -1.0) == (-_pow,)
    @test gradient(p -> real(2^p), 2)[1] ≈ 4*log(2)

    @test gradient(xs ->sum(xs .^ 2), [2, -1]) == ([4, -2],)
    @test gradient(xs ->sum(xs .^ 10), [3, -1]) == ([10*3^9, -10],)
    @test gradient(xs ->sum(xs .^ _pow), [4, -1]) == ([_pow*4^9, -10],)

    @test gradient(x -> real((1+3im) * x^2), 5+7im) == (-32 - 44im,)
    @test gradient(p -> real((1+3im) * (5+7im)^p), 2)[1] ≈ real((-234 + 2im)*log(5 - 7im))
    # D[(1+3I)x^p, p] /. {x->5+7I, p->2} // Conjugate
end

@testset "jacobian" begin
    @test jacobicheck((x, W, b) -> identity.(W*x .+ b), 5, (2,5), 2)
    @test jacobicheck((x, W, b) -> identity.(W*x .+ b), (5,3), (2,5), 2)


    @test jacobicheck((x, W, b) -> tanh.(W*x .+ b), 5, (2,5), 2)
    @test jacobicheck((x, W, b) -> tanh.(W*x .+ b), (5,3), (2,5), 2)

    @test jacobicheck((w, x) -> w'*x, randn(10, 2), randn(10))
    @test jacobicheck((w, x) -> Adjoint(w)*x, randn(10, 2), randn(10))
    @test jacobicheck((w, x) -> transpose(w)*x, randn(5,5), randn(5,5))
    @test jacobicheck((w, x) -> Transpose(w)*x, randn(5,5), randn(5,5))


    # FIXME: fail with:
    #   MethodError: no method matching isapprox(::Tangent{Adjoint{Float64, Matrix{Float64}}, @NamedTuple{parent::Matrix{Float64}}}, ::Adjoint{Float64, Matrix{Float64}}; rtol::Float64, atol::Float64)
    @test_broken jacobicheck((w, x) -> parent(w)*x, randn(5,5)', randn(5,5))
    @test_broken jacobicheck((w, x) -> parent(w)*x, transpose(randn(5,5)), randn(5,5))
end

@testset "sum, prod" begin
    @test gradcheck(x -> sum(abs2, x), randn(4, 3, 2))
    @test gradcheck(x -> sum(x[i] for i in 1:length(x)), randn(10))
    @test gradcheck(x -> sum(i->x[i], 1:length(x)), randn(10)) #  issue #231
    @test gradcheck(x -> sum((i->x[i]).(1:length(x))), randn(10))
    @test gradcheck(X -> sum(x -> x^2, X), randn(10))

    # FIXME: fail with
    #    MethodError: no method matching copy(::Nothing)
    @test_broken jacobicheck(x -> sum(x, dims = (2, 3)), (3,4,5))
    @test_broken jacobicheck(x -> sum(abs2, x; dims=1), randn(4, 3, 2))
    @test_broken gradcheck(X -> sum(sum(x -> x^2, X; dims=1)), randn(10)) # issue #681
  
    # Non-differentiable sum of booleans
    @test gradient(sum, [true, false, true]) == (NoTangent(),)
    @test gradient(x->sum(x .== 0.0), [1.2, 0.2, 0.0, -1.1, 100.0]) == (NoTangent(),)
  
    # https://github.com/FluxML/Zygote.jl/issues/314
    @test gradient((x,y) -> sum(yi -> yi*x, y), 1, [1,1]) == (2, [1, 1])
    @test gradient((x,y) -> prod(yi -> yi*x, y), 1, [1,1]) == (2, [1, 1])
  
    # FIXME: fail with
    #  AssertionError: Base.issingletontype(typeof(f))
    @test_broken gradient((x,y) -> sum(map(yi -> yi*x, y)), 1, [1,1]) == (2, [1, 1])
    @test_broken gradient((x,y) -> prod(map(yi -> yi*x, y)), 1, [1,1]) == (2, [1, 1])
  
    @test gradcheck(x -> prod(x), (3,4))
    @test gradient(x -> prod(x), (1,2,3))[1] == (6,3,2)

    # FIXME: fail with
    #    MethodError: no method matching copy(::Nothing)
    @test_broken jacobicheck(x -> prod(x, dims = (2, 3)), (3,4,5))
end
  
@testset "cumsum" begin
    @test jacobicheck(x -> cumsum(x), (4,))

    # FIXME: fail with
    #   TypeError: in typeassert, expected Int64, got a value of type Nothing
    @test_broken jacobicheck(x -> cumsum(x, dims=2), (3,4,5))
    @test_broken jacobicheck(x -> cumsum(x, dims=3), (3,4)) # trivial

    # FIXME: fail with
    #    MethodError: no method matching copy(::Nothing)
    @test_broken jacobicheck(x -> cumsum(x, dims=1), (3,))

    # FIXME: fail with
    #   Rewrite reached intrinsic function bitcast. Missing rule?
    @test_broken jacobicheck(x -> cumsum(x, dims=3), (5,))  # trivial
end


# FIXME: complex numbers; put somewhere
@test gradcheck((a,b)->sum(reim(acosh(complex(a[1], b[1])))), [-2.0], [1.0])

# FIXME: include those?
# @testset "println, show, string, etc" begin
#     function foo(x)
#         Base.show(x)
#         Base.print(x)
#         Base.print(stdout, x)
#         Base.println(x)
#         Base.println(stdout, x)
#         Core.show(x)
#         Core.print(x)
#         Core.println(x)
#         return x
#     end
#     gradcheck(f, xs::AbstractArray...) = gradcheck((xs...) -> sum(sin.(f(xs...))), xs...)
#     println("The following printout is from testing that `print` doesn't upset gradients:")
#     @test gradcheck(foo, [5.0])
#   
#     function bar(x)
#         string(x)
#         repr(x)
#         return x
#     end
#     @test gradcheck(bar, [5.0])
# end

