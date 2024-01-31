module gradcheck_tests
# This file contains a selection of tests from Zygote's "gradcheck.jl",
# dealing with Base and standard library functions. Many of these use rules
# which have their own more exhaustive tests in ChainRules.

# Tests for packages (Distances, LogExpFunctions, AbstractFFTs, FillArrays) are not included.

# Ideally this would be extended to take `gradient` both forward and reverse,
# and `jacobicheck` including 2nd derivatives, for every testset. But not yet.

using Test
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

# Zygote's misnamed hobbit function:
function value_and_pullback(f, x...)
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
    using LinearAlgebra: var"'"
    @test jacobicheck(x -> x', rand(5))
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
    # wrong gradient: Evaluated: ([1.0 0.0 0.0; 1.0 0.0 0.0; 2.0 0.0 0.0], NoTangent()) == ([1, 1, 2], NoTangent())
    @test_broken gradient(y139, x, z) == ([1, 1, 2], NoTangent())

    # https://github.com/FluxML/Zygote.jl/issues/376

    _, back = value_and_pullback(x->x[1]*im, randn(2))
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
    _, back = value_and_pullback(getindex, randn(4), [1])
    @test back([ZeroTangent()]) == (zeros(4), NoTangent())
    # Ensure that nothings work with non-numeric types.
    _, back = value_and_pullback(getindex, [randn(2) for _ in 1:3], [1])
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
      y, back = value_and_pullback(permutedims, randn(3))
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
        @test_broken jacobicheck((xss...) -> map((xs...) -> sqrt(sum(xs.^2)), xss...), [rand(5) for _ in 1:6]...)

        # MethodError: no method matching copy(::Nothing)
        @test_broken gradcheck(y -> map(x -> x*y, 1:5), 3)
        @test_broken gradient(v -> sum([x for x in v]), [1.1,2.2,3.3]) == ([1, 1, 1],)
    end

    @testset "bascis, pmap" begin
        # BoundsError: attempt to access 0-element Core.Compiler.UnitRange{Int64} at index [0]
        @test_broken jacobicheck(xs -> pmap(x -> x^2, xs), rand(2,3))
        @test_broken jacobicheck((xss...) -> pmap((xs...) -> sqrt(sum(xs.^2)), xss...), [rand(5) for _ in 1:6]...)

        function foo(y)
            bar = (x) -> x*y
            sum(pmap(bar, 1:5))
        end
        # BoundsError: attempt to access 0-element Core.Compiler.UnitRange{Int64} at index [0]
        @test_broken jacobicheck(y -> pmap(x -> x*y, 1:5), 3)
    end

    @testset "Tuple adjoint" begin
        x = randn(3)
        _, pb = value_and_pullback(x -> map(abs2, x), x)
        Δy = randn(3)
        @test first(pb((Δy..., ))) ≈ first(pb(Δy))
    end

    @testset "empty tuples" begin
        out, pb = value_and_pullback(map, -, ())
        @test pb(out) === (NoTangent(), NoTangent())

        out, pb = value_and_pullback(map, +, (), ())
        # MethodError: reducing over an empty collection is not allowed, ChainRules.var"#map_value_and_pullback#1234"{typeof(+), Tuple{Tuple{}, Tuple{}},
        @test_broken pb(()) === (ZeroTangent(), ZeroTangent(), ZeroTangent())

        function build_foo(z)
            foo(x) = x * z
            return foo
        end
        out, pb = value_and_pullback(map, build_foo(5.0), ())
        @test pb(()) === (NoTangent(), NoTangent())
    end

    @testset "Vector{Nothing} cotangent" begin
        Δ = fill(ZeroTangent(), 5)

        # Unary stateless
        out, pb = value_and_pullback(map, -, randn(5))
        @test pb(Δ)[2] isa Vector{ZeroTangent}

        # Binary stateless
        out, pb = value_and_pullback(map, +, randn(5), randn(5))
        @test pb(Δ)[2] isa Vector{ZeroTangent}
        @test pb(Δ)[3] isa Vector{ZeroTangent}

        # Stateful
        function build_foo(z)
            foo(x) = x * z
            return foo
        end
        # AssertionError: Base.issingletontype(typeof(f))
        @test_broken out, pb = value_and_pullback(map, build_foo(5.0), randn(5))
        @test_skip pb(Δ)[2] isa Vector{ZeroTangent}
    end

    # Check that map infers correctly. pmap still doesn't infer.
    @testset "map inference" begin
        @testset "$name" for (name, f, ȳ, xs) in [
            ("unary empty vector", sin, Float64[], (Float64[], )),
            ("unary vector", sin, randn(3), (randn(3), )),
            ("unary empty tuple", sin, (), ((), )),
            ("unary tuple", sin, (randn(), randn()), ((randn(), randn()), )),
            ("binary empty vector", +, Float64[], (Float64[], Float64[])),
            ("binary vector", +, randn(2), (randn(2), randn(2))),
        ]
            @inferred value_and_pullback(map, f, xs...)
            y, pb = value_and_pullback(map, f, xs...)
            @inferred pb(ȳ)
        end

        # these are broken
        @test_skip @testset "$name" for (name, f, ȳ, xs) in [
            # MethodError: reducing over an empty collection is not allowed; consider supplying `init` to the reducer
            ("binary empty tuple", +, (), ((), ())),
            # return type Tuple{NoTangent, Tangent{...}...} does not match inferred
            # return type Tuple{NoTangent, {Union{NoTangent, Tangent{...}}}}
            ("binary tuple", +, (randn(), randn()), ((randn(), randn()), (randn(), randn()))),
        ]
            @inferred value_and_pullback(map, f, xs...)
            y, pb = value_and_pullback(map, f, xs...)
            @inferred pb(ȳ)
        end
    end

    @testset "map and tuples" begin
        # arrays of tuples, ChainRules's Tangent should not escape
        # MethodError: no method matching one(::Tuple{Int64, Int64})
        @test_broken gradient(x -> sum(map(first, x)), [(1,2), (3,4)]) == ([(1.0, nothing), (1.0, nothing)],)
        T = Tangent{Tuple{Int64, Int64}}
        @test gradient(x -> sum(first, x), [(1,2), (3,4)]) == (T[T(1.0, ZeroTangent()), T(1.0, ZeroTangent())],)
    
        @test gradient(x -> map(+, x, (1,2,3))[1], (4,5,6)) == (Tangent{Tuple{Int,Int,Int}}(1.0, ZeroTangent(), ZeroTangent()),)
        # MethodError: no method matching copy(::Nothing)
        @test_broken gradient(x -> map(+, x, [1,2,3])[1], (4,5,6)) == ((1.0, 0.0, 0.0),)
        @test_broken gradient(x -> map(+, x, (1,2,3))[1], [4,5,6]) == ([1,0,0],)
    
        # mismatched lengths, should zip
        # MethodError: no method matching copy(::Nothing)
        @test_broken gradient(x -> map(+, x, [1,2,3,99])[1], (4,5,6)) == ((1.0, 0.0, 0.0),)
        @test_broken gradient(x -> map(+, x, [1,2,3])[1], (4,5,6,99)) == ((1.0, 0.0, 0.0, nothing),)
    end

    @testset "Alternative Pmap Dispatch" begin
        cache_and_map(f,xs...) = pmap(f, CachingPool(workers()), xs...; batch_size = 1)
        # BoundsError: attempt to access 0-element Core.Compiler.UnitRange{Int64} at index [0]
        @test_broken jacobicheck(xs -> cache_and_map(x -> x^2, xs), rand(2,3))
        @test_broken jacobicheck((xss...) -> cache_and_map((xs...) -> sqrt(sum(xs.^2)), xss...), [rand(5) for _ in 1:6]...)
        @test_broken jacobicheck(y -> cache_and_map(x->x*y, 1:5), 3)
    end

    @testset "Stateful Map" begin
        s = 0
        f(x) = (s += x)
        # Tuple field type cannot be Union{}
        @test_broken gradient(x -> sum(f.(x)), 1:10) == (10:-1:1,)
        s = 0
        # MethodError: no method matching copy(::Nothing)
        @test_broken gradient(x -> sum(map(f, x)), 1:10) == (10:-1:1,)
    end

    @testset "vararg map" begin
        # early stop
        # MethodError: no method matching length(::InplaceableThunk{...})
        if VERSION >= v"1.5"
          # In Julia 1.4 and earlier, map(*,rand(5),[1,2,3]) is a DimensionMismatch
          @test_broken gradient(x -> sum(map(*,x,[1,2,3])), rand(5)) == ([1,2,3,0,0],)
        end
        @test_broken gradient(x -> sum(map(*,x,(1,2,3))), rand(5)) == ([1,2,3,0,0],)
        @test_broken gradient(x -> sum(map(*,x,[1,2,3])), Tuple(rand(5))) == ((1.0, 2.0, 3.0, nothing, nothing),)
      
        # mixed shapes
        # MethodError: no method matching length(::InplaceableThunk{...})
        @test_broken gradient((x,y) -> sum(map(*,x,y)), [1,2,3,4], [1 2; 3 4]) == ([1,3,2,4], [1 3; 2 4])
        @test_broken gradient((x,y) -> sum(map(*,x,y)), [1,2,3], [1 2; 3 4]) == ([1,3,2], [1 3; 2 0])
        @test_broken gradient((x,y) -> sum(map(*,x,y)), (1,2,3), [1 2; 3 4]) == ((1,3,2), [1 3; 2 0])
        @test_broken gradient((x,y) -> sum(map(*,x,y)), [1,2,3,4,5], [1 2; 3 4]) == ([1,3,2,4,0], [1 3; 2 4])
        @test_broken gradient((x,y) -> sum(map(*,x,y)), (1,2,3,4,5), [1 2; 3 4]) == ((1,3,2,4,nothing), [1 3; 2 4])
    end

    @testset "map: issue 1374" begin
        # https://github.com/FluxML/Zygote.jl/issues/1374
        struct Affine1374
          W
          b
        end
        (m::Affine1374)(x) = [sum(x.*r) for r in eachrow(m.W)] + m.b
        m = Affine1374(zeros(3,3), zeros(3,1))
        x = [ 1.0,  2.0,  3.0]
        y = [-1.0, -2.0, -3.0]
        l1374(y,ŷ) = sum(abs2.(y - ŷ))/2
        @test_broken gradient(m -> l1374(y,m(x)), m)[1].W ≈ [1 2 3; 2 4 6; 3 6 9]
    end
end

@testset "sort" begin
    @test jacobicheck(sort, 5)
    correct = [
        [2,3,1],
        [1, 2, 3],
        [1,2,3],
        [2,1,3],
        [1,3,2],
        [3,2,1]
    ]
    for i = 1:3
        @test gradient(v->sort(v)[i], [3.,1,2])[1][correct[1][i]] == 1
        @test gradient(v->sort(v)[i], [1.,2,3])[1][correct[2][i]] == 1
    end
    for i = 1:3
        # Rewrite reached intrinsic function bitcast. Missing rule?
        @test_broken gradient(v->sort(v,by=x->x%10)[i], [11,2,99])[1][correct[3][i]] == 1
        @test_broken gradient(v->sort(v,by=x->x%10)[i], [2,11,99])[1][correct[4][i]] == 1
        @test_broken gradient(v->sort(v,rev=true)[i], [3.,1,2])[1][correct[5][i]] == 1
        @test_broken gradient(v->sort(v,rev=true)[i], [1.,2,3])[1][correct[6][i]] == 1
    end
end

@testset "filter" begin
    @test jacobicheck(xs -> filter(x -> x > 0.5, xs), rand(20))

    @test gradient(x -> sum(log, filter(iseven, x)), 1:10) ==
        (map(x -> iseven(x) ? 1/x : 0, 1:10),)
    @test gradient(x -> sum(abs2, im .+ filter(iseven, x)), 1:10) ==
        (map(x -> iseven(x) ? 2x : 0, 1:10),)
        # (map(x -> iseven(x) ? 2x+2im : 0, 1:10),)
end

@testset "maximum" begin
    @test jacobicheck(maximum, rand(2, 3))

    # MethodError: no method matching copy(::Nothing)
    @test_broken jacobicheck(x -> maximum(x, dims=1), rand(2, 3))
    @test_broken jacobicheck(x -> maximum(x, dims=3), rand(2, 3, 4))
    @test_broken jacobicheck(x -> maximum(x, dims=[1, 2]), rand(2, 3, 4))

    @test gradient(x -> 1 / maximum(x), [1., 2, 3])[1] == [0, 0, -1/9]
end

@testset "minimum" begin
    @test jacobicheck(minimum, rand(2, 3))

    # MethodError: no method matching copy(::Nothing)
    @test_broken jacobicheck(x -> minimum(x, dims=1), rand(2, 3))
    @test_broken jacobicheck(x -> minimum(x, dims=2), rand(2, 3))
end

@testset "dropdims" begin  # https://github.com/JuliaDiff/Diffractor.jl/issues/72
    # TypeError: in typeassert, expected Int64, got a value of type Nothing
    @test_broken jacobicheck(x -> dropdims(x, dims = 3), rand(2, 2, 1, 2))
    @test_broken jacobicheck(x -> dropdims(x, dims = (2, 3)), rand(2, 1, 1, 3))
end

@testset "vcat" begin
    # Scalar
    @test gradient((x,y) -> sum(vcat(x,y)), 1,2) == (1,1)
    @test gradient((x,y) -> sum([x; y; x]), 1,2) == (2,1)

    # Scalar + Vector
    @test gradient(x -> sum(vcat(x, 1, x)), rand(3)) == ([2,2,2],)
    @test gradient((x,y) -> sum(vcat(x, y, y)), rand(3), 4) == ([1,1,1], 2)

    # Vector-only.
    @test jacobicheck(vcat, randn(10))
    @test jacobicheck(x -> vcat(x, [1,2,3], x), randn(3))

    # Matrix-Vector
    @test jacobicheck(x-> vcat(x, [1,2,3]), rand(2,1))
    @test jacobicheck(x-> vcat(x, ones(3,1)), rand(2))
end

@testset "hcat" begin
    # Scalar
    @test gradient((x,y) -> sum(hcat(x,y)), 1,2) == (1,1)
    @test gradient((x,y) -> sum([x y]), 1,2) == (1,1)
    @test gradient((a,b,c,d) -> sum(sqrt, [a b;c d]), 1,1,1,4) == (0.5, 0.5, 0.5, 0.25)

    # Vector-only
    @test jacobicheck(hcat, rand(3))
    @test jacobicheck(x -> hcat(x, [1,2,3]), rand(3))

    # Matrix-only
    @test jacobicheck(hcat, rand(3,4))
    @test jacobicheck(x -> hcat(x, [1 2; 3 4], x), rand(2,2))

    # Matrix-Scalar
    @test gradient((x,y) -> sum(hcat(x, y)), 1, [2 3 4]) == (1, [1 1 1])
    @test gradient(x -> sum(hcat(1, x, 2)), transpose([3,4,5]))[1] isa Transpose
    @test gradient(x -> sum(hcat(1, x, 2)), [3,4,5]')[1] isa Adjoint
end

@testset "hvcat" begin
    @test gradient(xs -> hvcat((2,2),xs...)[1,1], [1,2,3,4])[1] == [1,0,0,0]
    @test gradient(xs -> hvcat((2,2),xs...)[2,1], [1,2,3,4])[1] == [0,0,1,0]
    @test gradient(xs -> hvcat((2,2),xs...)[1,2], [1,2,3,4])[1] == [0,1,0,0]
    @test gradient(xs -> hvcat((2,2),xs...)[2,2], [1,2,3,4])[1] == [0,0,0,1]
    # https://github.com/FluxML/Zygote.jl/issues/513
    @test gradient(x -> hvcat((2,2),1,2,3,x)[4], 4.0) == (1.0,)
end

@testset "cat(...; dims = $dim)" for dim in 1:3
    # Rewrite reached intrinsic function bitcast. Missing rule?

    catdim = (x...) -> cat(x..., dims = dim)
    @test_broken jacobicheck(catdim, rand(4,1))
    @test_broken jacobicheck(catdim, rand(5), rand(5,1))
    @test_broken jacobicheck(catdim, rand(2,5), rand(2,5), rand(2,5)) 

    catdimval = (x...) -> cat(x...; dims = Val(dim))
    @test_broken jacobicheck(catdimval, rand(5), rand(5))
    @test_broken jacobicheck(catdimval, rand(2,5), rand(2,5,1))

    # one empty
    dim == 1 || continue
    @test_broken jacobicheck(catdim, rand(0,5,3), rand(2,5,3))
end

@testset "one(s) and zero(s)" begin
    # should these be ZeroTangent or NoTangent?
    @test gradient(x->sum(ones(size(x))), randn(5))[1] === NoTangent()
    @test_broken gradient(x->sum(one(x)), randn(3, 3))[1] === NoTangent()
    @test gradient(x->sum(zeros(size(x))), randn(7))[1] === NoTangent()
    @test_broken gradient(x->sum(zero(x)), randn(3))[1] === NoTangent()
end

@testset "fma and muladd" begin
    @test gradcheck(x -> fma(x[1], x[2], x[3]), [2.0, 3.0, 5.0])
    @test gradcheck(x -> muladd(x[1], x[2], x[3]), [2.0, 3.0, 5.0])
end

@testset "broadcast" begin
    @test gradient(x -> sum(sin.(x)), Diagonal([0,pi/2,pi]))[1] ≈ [1 0 0; 0 0 0; 0 0 -1]

    # mixing arrays & Ref(array)
    a = rand(3)
    b = rand(2,2)
    @test jacobicheck(x -> sum(diag.((x,) .* a)), b)
    @test jacobicheck(x -> sum(diag.(Ref(x) .* a)), b)
    @test jacobicheck(x -> sum(diag.([x] .* a)), b)

    # tests for https://github.com/FluxML/Zygote.jl/issues/724
    x1 = rand(3, 3)
    @test gradient(x -> sum(x .== 0.5), x1) |> only |> isZero
    # MethodError: no method matching copy(::Nothing)
    @test_broken gradient(x -> sum(x .* (x .== maximum(x, dims=1))), x1)[1] == (x1 .== maximum(x1, dims=1))

    # tests for un-broadcasting *, / via scalar rules
    @test all(gradient((x,y) -> sum(x .* y), [1,2], 5) .≈ ([5, 5], 3))
    @test all(gradient((x,y) -> sum(x .* y), 5, [1,2]) .≈ (3, [5, 5]))
    @test all(gradient((x,y) -> sum(x .* y), [1,2], [3 4 5]) .≈ ([12, 12], [3 3 3]))
    @test all(gradient((x,y) -> sum(x ./ y), [1,2], 5) .≈ ([0.2, 0.2], -0.12))

    @test_skip begin
        using SparseArrays  # not loaded at present
        # https://github.com/FluxML/Zygote.jl/pull/1171
        sm = sprand(5, 5, 0.5)
        @test gradient(x -> sum(abs2, Float32.(x)), sm)[1] ≈ gradient(x -> sum(abs2, x), Matrix{Float32}(sm))[1]
        @test_broken gradient(x -> real(sum(ComplexF32.(x) .+ 1 .+ im)), sm)[1] isa SparseMatrixCSC{Float64}  # MethodError: no method matching zero(::Type{Any}), in ProjectTo(xs::SparseMatrixCSC{Any, Int64})
    end

    # https://github.com/FluxML/Zygote.jl/issues/1178
    function f1179(x)
        fs = Ref.(x)
        getindex.(fs)
    end
    # wrong gradient: Evaluated: ([1.0, 1.0],) == ([2.0, 2.0],)
    @test_broken gradient(sum∘f1179, ones(2)) == ([2.0, 2.0],)  # MethodError: no method matching one(::Base.RefValue{Float64})
end

@testset "array +,-" begin
    A, B = randn(3, 4, 5), randn(3, 4, 5)
    @test jacobicheck(+, B)
    @test jacobicheck(+, A, B)
    # wrong gradient
    @test_broken jacobicheck(+, A, B, A)
    @test jacobicheck(-, A)
    # in typeassert, expected Int64, got a value of type Nothing
    @test_broken jacobicheck(-, A, B)
end

end
