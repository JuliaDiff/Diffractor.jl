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

# Zygote's misnamed hobbit function:
function pullback(f, x...)
    y, b = Diffractor.∂⃖{1}()(f, x...)
    back(dy) = map(unthunk, Base.tail(b(dy)))
    y, back
end

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


    # MethodError: no method matching isapprox(::Tangent{Adjoint{Float64, Matrix{Float64}}, @NamedTuple{parent::Matrix{Float64}}}, ::Adjoint{Float64, Matrix{Float64}}; rtol::Float64, atol::Float64)
    @test_broken jacobicheck((w, x) -> parent(w)*x, randn(5,5)', randn(5,5))
    @test_broken jacobicheck((w, x) -> parent(w)*x, transpose(randn(5,5)), randn(5,5))
end

@testset "sum, prod" begin
    @test gradcheck(x -> sum(abs2, x), randn(4, 3, 2))
    @test gradcheck(x -> sum(x[i] for i in 1:length(x)), randn(10))
    @test gradcheck(x -> sum(i->x[i], 1:length(x)), randn(10)) #  issue #231
    @test gradcheck(x -> sum((i->x[i]).(1:length(x))), randn(10))
    @test gradcheck(X -> sum(x -> x^2, X), randn(10))

    # MethodError: no method matching copy(::Nothing)
    @test_broken jacobicheck(x -> sum(x, dims = (2, 3)), (3,4,5))
    @test_broken jacobicheck(x -> sum(abs2, x; dims=1), randn(4, 3, 2))
    @test_broken gradcheck(X -> sum(sum(x -> x^2, X; dims=1)), randn(10)) # issue #681
  
    # Non-differentiable sum of booleans
    @test gradient(sum, [true, false, true]) == (NoTangent(),)
    @test gradient(x->sum(x .== 0.0), [1.2, 0.2, 0.0, -1.1, 100.0]) == (NoTangent(),)
  
    # https://github.com/FluxML/Zygote.jl/issues/314
    @test gradient((x,y) -> sum(yi -> yi*x, y), 1, [1,1]) == (2, [1, 1])
    @test gradient((x,y) -> prod(yi -> yi*x, y), 1, [1,1]) == (2, [1, 1])
  
    # AssertionError: Base.issingletontype(typeof(f))
    @test_broken gradient((x,y) -> sum(map(yi -> yi*x, y)), 1, [1,1]) == (2, [1, 1])
    @test_broken gradient((x,y) -> prod(map(yi -> yi*x, y)), 1, [1,1]) == (2, [1, 1])
  
    @test gradcheck(x -> prod(x), (3,4))
    @test gradient(x -> prod(x), (1,2,3))[1] == (6,3,2)

    # MethodError: no method matching copy(::Nothing)
    @test_broken jacobicheck(x -> prod(x, dims = (2, 3)), (3,4,5))
end
  
@testset "cumsum" begin
    @test jacobicheck(x -> cumsum(x), (4,))

    # TypeError: in typeassert, expected Int64, got a value of type Nothing
    @test_broken jacobicheck(x -> cumsum(x, dims=2), (3,4,5))
    @test_broken jacobicheck(x -> cumsum(x, dims=3), (3,4)) # trivial

    # MethodError: no method matching copy(::Nothing)
    @test_broken jacobicheck(x -> cumsum(x, dims=1), (3,))

    # Rewrite reached intrinsic function bitcast. Missing rule?
    @test_broken jacobicheck(x -> cumsum(x, dims=3), (5,))  # trivial
end

@testset "getindex" begin
    @test jacobicheck(x -> x[:, 2, :], (3, 4, 5))
    @test jacobicheck(x -> x[1:2, 3:4], (3, 4))

    imat = [1 2; 3 4]
    @test jacobicheck(x -> x[:, imat], (3, 4))

    # incorrect gradient
    # julia> gradient(sum ∘ x->x[:,[1,2,2]], x)[1]
    # 3×4 Matrix{Float64}:
    # 1.0  1.0  0.0  0.0
    # 1.0  1.0  0.0  0.0
    # 1.0  1.0  0.0  0.0
    # while it should be
    # 3×4 Matrix{Float64}:
    # 1.0  2.0  0.0  0.0
    # 1.0  2.0  0.0  0.0
    # 1.0  2.0  0.0  0.0
    @test_broken jacobicheck(x -> x[:, [1, 2, 2]], (3, 4))
    # same here
    irep = [1 2; 2 2]
    @test_broken jacobicheck(x -> x[1, irep], (3, 4))

    # https://github.com/invenia/Nabla.jl/issues/139
    x = rand(3)
    z = [1, 2, 3, 3]
    y139(x, z) = dot(ones(4), x[z])
    # Evaluated: ([1.0 0.0 0.0; 1.0 0.0 0.0; 2.0 0.0 0.0], NoTangent()) == ([1, 1, 2], NoTangent())
    @test_broken gradient(y139, x, z) == ([1, 1, 2], NoTangent())

    # https://github.com/FluxML/Zygote.jl/issues/376

    _, back = pullback(x->x[1]*im, randn(2))
    @test back(1.0)[1] == real([-im, 0]) == [0, 0]

    # _droplike
    @test gradient(x -> sum(inv, x[1, :]'), ones(2, 2)) == ([-1 -1; 0 0],)
    @test gradient(x -> sum(inv, transpose(x[1, :])), ones(2, 2)) == ([-1 -1; 0 0],)  # same with transpose, in case ' overloaded!
    @test gradient(x -> sum(inv, x[1:1, :]'), ones(2, 2)) == ([-1 -1; 0 0],)
    @test gradient(x -> sum(inv, transpose(x[1:1, :])), ones(2, 2)) == ([-1 -1; 0 0],)
    @test gradient(x -> sum(inv, transpose(view(x, 1, :))), ones(2, 2)) == ([-1 -1; 0 0],)

    # https://github.com/FluxML/Zygote.jl/issues/513
    # MethodError: no method matching copy(::Nothing)
    @test_broken gradient(p -> sum(Float32[1, 0] - p), [2, 3]) == ([-1, -1],)
    @test_broken gradient(x -> sum(Float32[1, x] .+ x), 4) == (3.0f0,)

    # Ensure that nothings work with numeric types.
    _, back = pullback(getindex, randn(4), [1])
    @test back([ZeroTangent()]) == (zeros(4), NoTangent())
    # Ensure that nothings work with non-numeric types.
    _, back = pullback(getindex, [randn(2) for _ in 1:3], [1])
    @test back([ZeroTangent()]) == (NoTangent(), NoTangent())
end

@testset "view" begin
    @test jacobicheck(x -> view(x,:,2,:), (3,4,5))
    @test jacobicheck(x -> view(x,1:2,3:4), (3,4))
    @test jacobicheck(x -> view(x,:,[1,2,2]), (3,4))

    # https://github.com/FluxML/Zygote.jl/issues/272
    g272(x) = view(x,1:2)[1]
    @test gradient(g272, ones(3)) == ([1,0,0],)
end

# 194
@testset "eachcol" begin
    # MethodError: no method matching one(::SubArray{Float64, 1, Matrix{Float64}, Tuple{Base.Slice{Base.OneTo{Int64}}, Int64}, true})
    @test_broken jacobicheck(x -> map(sum, eachcol(x)), (3,4))
    @test_broken jacobicheck(x -> map(sum, eachcol(transpose(x))), (3,4))
    @test_broken jacobicheck(x -> map(norm, eachcol(x)), (3,4))
    @test_broken jacobicheck(x -> map(norm, eachrow(x)), (3,4))
    @test_broken jacobicheck(x -> map(norm, eachslice(x, dims=3)), (3,4,5))

    # some slices may have gradient nothing
    @test gradient(x -> sum(y -> rand()>0.5 ? 0 : first(y), eachcol(x)), rand(3,10))[1] isa Matrix

    # strange errors (on Zygote)
    @test gradient(x -> sum(norm, eachcol(x)), [1 2 3; 4 5 6])[1] isa Matrix
    @test gradient(x -> sum(norm, eachcol(x)), rand(3,400))[1] isa Matrix
end

@testset "collect" begin
    # MethodError: no method matching copy(::Nothing)
    @test_broken gradient(x -> sum(inv, collect(x)), (1,2)) === ((-1.0, -1/4),)
    @test_broken gradient(xs -> sum(inv, [x^2 for x in xs]), ones(2)) == ([-2, -2],)

    # Rewrite reached intrinsic function bitcast. Missing rule?
    @test_broken gradient(x -> sum(collect(view(x, 1:1))), rand(2)) == ([1,0],)
    @test_broken gradient(x -> sum(inv, collect(view(x', 1,:))), ones(2,2)) == ([-1 0; -1 0],)
end


@testset "reverse" begin
    @test jacobicheck(x -> reverse(x), rand(17))
    @test jacobicheck(x -> reverse(x, 8), rand(17))
    @test jacobicheck(x -> reverse(x, 8, 13), rand(17))
    # Rewrite reached intrinsic function bitcast. Missing rule?
    @test_broken jacobicheck(x -> reverse(x, dims=2), rand(17, 42))
end

@testset "permutedims" begin
    @test jacobicheck(x -> permutedims(x), rand(2))
    @test jacobicheck(x -> permutedims(x), rand(2,3))
    @test jacobicheck(x -> permutedims(x, [3,1,2]), rand(4,5,6))
    @test jacobicheck(x -> PermutedDimsArray(x, (3,1,2)), rand(4,5,6))
    let
      y, back = pullback(permutedims, randn(3))
      @test first(back(randn(1, 3))) isa Vector
    end
end

@testset "repeat" begin
    # MethodError: no method matching copy(::Nothing)
    @test_broken jacobicheck(x -> repeat(x; inner=2), rand(5))
    @test_broken jacobicheck(x -> repeat(x; inner=2, outer=3), rand(5))
    @test_broken jacobicheck(x -> repeat(x; inner=(2,2,1), outer=(1,1,3)), rand(5,4,3))

    @test jacobicheck(x -> repeat(x, 3), rand(5))
    @test jacobicheck(x -> repeat(x, 2, 3), rand(5))
    @test jacobicheck(x -> repeat(x, 5), rand(5,7))
    @test jacobicheck(x -> repeat(x, 3, 2), rand(5,3))
end

@testset "fill" begin
    @test jacobicheck(x->fill(x[1], 3), rand(1))
    @test jacobicheck(x->fill(x[1], 3, 4, 2), rand(1))

    # fill(struct, ...) handled by ChainRules after
    # https://github.com/FluxML/Zygote.jl/pull/1051
    # FIXME: marking as broken, because not sure what the tangent should contain.
    # in the first we have a ZeroTangent, and the second we dont.
    @test_broken gradient(x -> fill(x, 3)[1][1], (1,2)) == (Tangent{Tuple{Int,Int}}(1.0,),)
    @test_broken gradient(x -> fill(x, 3)[1].a, (a=1, b=2)) == (Tangent{@NamedTuple{a::Int64, b::Int64}}(a = 1.0, b=nothing),)  # 1 not 1.0
end

@testset "circshift" begin
    for D in 1:5
        x0 = zeros(ntuple(d->5, D))
        g = gradient(x -> x[1], x0)[1] 
        shift = ntuple(_ -> rand(-5:5), D)
        @test gradient(x -> circshift(x, shift)[1], x0)[1] == circshift(g, map(-, shift))
    end
end

@testset "map" begin
    @testset "bascis" begin
        @test jacobicheck(xs -> map(x -> x^2, xs), rand(2,3))

        # MethodError: no method matching copy(::Nothing)
        @test_broken jacobicheck((xss...) -> sum(map((xs...) -> sqrt(sum(xs.^2)), xss...)), [rand(5) for _ in 1:6]...)

        # MethodError: no method matching copy(::Nothing)
        @test_broken gradcheck(y -> map(x -> x*y, 1:5), 3)
        @test_broken gradient(v -> sum([x for x in v]), [1.1,2.2,3.3]) == ([1, 1, 1],)
    end

    @test_skip @testset "bascis, pmap" begin
        @test jacobicheck(xs -> sum(pmap(x -> x^2, xs)), rand(2,3))
        @test jacobicheck((xss...) -> sum(pmap((xs...) -> sqrt(sum(xs.^2)), xss...)), [rand(5) for _ in 1:6]...)

        function foo(y)
            bar = (x) -> x*y
            sum(pmap(bar, 1:5))
        end
        @test gradtest(foo, 3)
        @test gradient(v -> sum([x for x in v]), [1.1,2.2,3.3]) == ([1, 1, 1],)
    end

    @testset "Tuple adjoint" begin
        x = randn(3)
        _, pb = pullback(x -> map(abs2, x), x)
        Δy = randn(3)
        @test first(pb((Δy..., ))) ≈ first(pb(Δy))
    end

    @testset "empty tuples" begin
        out, pb = pullback(map, -, ())
        @test pb(out) === (NoTangent(), NoTangent())

        out, pb = pullback(map, +, (), ())
        # MethodError: reducing over an empty collection is not allowed, ChainRules.var"#map_pullback#1234"{typeof(+), Tuple{Tuple{}, Tuple{}},
        @test_broken pb(()) === (ZeroTangent(), ZeroTangent(), ZeroTangent())

        function build_foo(z)
            foo(x) = x * z
            return foo
        end
        out, pb = pullback(map, build_foo(5.0), ())
        @test pb(()) === (NoTangent(), NoTangent())
    end

    @testset "Vector{Nothing} cotangent" begin
        Δ = fill(ZeroTangent(), 5)

        # Unary stateless
        out, pb = pullback(map, -, randn(5))
        @test pb(Δ)[2] isa Vector{ZeroTangent}

        # Binary stateless
        out, pb = pullback(map, +, randn(5), randn(5))
        @test pb(Δ)[2] isa Vector{ZeroTangent}
        @test pb(Δ)[3] isa Vector{ZeroTangent}

        # Stateful
        function build_foo(z)
            foo(x) = x * z
            return foo
        end
        # AssertionError: Base.issingletontype(typeof(f))
        @test_broken out, pb = pullback(map, build_foo(5.0), randn(5))
        @test_skip pb(Δ)[2] isa Vector{ZeroTangent}
    end
end




@testset "LinearAlgebra.det" begin
    @test jacobicheck(det, (4, 4))
    @test jacobicheck(logdet, map(x -> x*x'+I, (randn(4, 4),))[1])
    @test jacobicheck(x -> logabsdet(x)[1], (4, 4))
    @test gradient(det, 2.0)[1] == 1
    @test gradient(logdet, 2.0)[1] == 0.5
end

@testset "kron" begin
    # FIXME: fail with
    #  TypeError: in typeassert, expected Int64, got a value of type Nothing
    @test_broken jacobicheck(kron, 5, 3)  
    @test_broken jacobicheck(kron, rand(5), rand(3), rand(8))
    @test_broken jacobicheck(kron, rand(5,1), rand(3,1))
    @test_broken jacobicheck(kron, rand(5,1), rand(3,1), rand(8,1))
    @test_broken jacobicheck(kron, rand(5,2), rand(3,2), rand(8,2))
end



# FIXME: complex numbers; put somewhere
@test gradcheck((a,b)->sum(reim(acosh(complex(a[1], b[1])))), [-2.0], [1.0])

# FIXME: misc tests
@test jacobicheck(x -> x', rand(5))


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

