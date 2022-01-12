
# This file contains a selection of tests from Zygote's "gradcheck.jl",
# dealing with Base and standard library functions. Many of these use rules
# which have their own more exhaustive tests in ChainRules.

# Tests for packages (Distances, LogExpFunctions, AbstractFFTs, FillArrays) are not included.

# Ideally this would be extended to take `gradient` both forward and reverse,
# and `jacobicheck` including 2nd derivatives, for every testset. But not yet.

using Diffractor, ChainRulesCore, FiniteDifferences
using Test, LinearAlgebra, Random, Distributed, Statistics

#####
##### Zygote/test/gradcheck.jl : setup
#####

n_grad(f, x::Real) = (central_fdm(5, 1)(f, x),)
n_grad(f, x::AbstractArray{<:Real}) = FiniteDifferences.grad(central_fdm(5, 1), f, float(x))
n_grad(f, xs::Vararg{Any,N}) where {N} = ntuple(N) do i
    n_grad(x -> f(ntuple(j -> j==i ? x : xs[j], N)...), xs[i])[1]
end

# Zygote's tests define functions like these:
gradcheck(f, xs...) = all(isapprox.(unthunk.(gradient(f, xs...)), n_grad(f, xs...); rtol = 1e-5, atol = 1e-5))
@test gradcheck(sqrt, 3.14)
@test gradcheck(sum, randn(10))
@test gradcheck(dot, randn(3), rand(3))

# ... but this one is called `gradtest` there:
jacobicheck(f, xs::AbstractArray...) = f(xs...) isa Number ? gradcheck(f, xs...) : 
    gradcheck((xs...) -> sum(sin, f(xs...)), xs...)
@test jacobicheck(identity, [1,2,3])  # one given array
@test jacobicheck(sum, [1,2,3])  # fallback to gradcheck

jacobicheck(f, dims...) = jacobicheck(f, randn.(Float64, dims)...)
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

# 73
@testset "power" begin
    @test gradient(x -> x^2, -2) == (-4,) # literal_pow
    @test gradient(x -> x^10, -1.0) == (-10,)
    _pow = 10
    @test gradient(x -> x^_pow, -1.0) == (-_pow,)
    @test unthunk(gradient(p -> real(2^p), 2)[1]) ≈ 4*log(2)

    @test gradient(xs ->sum(xs .^ 2), [2, -1]) == ([4, -2],)
    @test gradient(xs ->sum(xs .^ 10), [3, -1]) == ([10*3^9, -10],)
    @test gradient(xs ->sum(xs .^ _pow), [4, -1]) == ([_pow*4^9, -10],)

    @test gradient(x -> real((1+3im) * x^2), 5+7im) == (-32 - 44im,)
    @test unthunk(gradient(p -> real((1+3im) * (5+7im)^p), 2)[1]) ≈ real((-234 + 2im)*log(5 - 7im))
    # D[(1+3I)x^p, p] /. {x->5+7I, p->2} // Conjugate
end

@testset "sum, prod" begin
    @test jacobicheck(x -> sum(x, dims = (2, 3)), rand(3,4,5))
    @test jacobicheck(x -> sum(abs2, x), randn(4, 3, 2))
    @test jacobicheck(x -> sum(abs2, x; dims=1), randn(4, 3, 2))

    @test_broken gradcheck(x -> sum(x[i] for i in 1:length(x)), randn(10))
    @test_broken gradcheck(x -> sum(i->x[i], 1:length(x)), randn(10)) #  Zygote issue #231
    @test gradcheck(x -> sum((i->x[i]).(1:length(x))), randn(10))
    @test_broken gradcheck(X -> sum(x -> x^2, X), randn(10))  # MethodError: no method matching lastindex(::Diffractor.OpticBundle{Float64})
    @test_broken jacobicheck(X -> sum(x -> x^2, X; dims=1), randn(10)) # Zygote issue #681  # MethodError: no method matching (::Diffractor.∂⃖recurse{1})(::typeof(Core.arrayset), ::Bool, ::Vector{Float64}, ::Float64, ::Int64)

    # Non-differentiable sum of booleans
    @test gradient(sum, [true, false, true]) == (NoTangent(),)
    @test_broken gradient(x->sum(x .== 0.0), [1.2, 0.2, 0.0, -1.1, 100.0]) == (NoTangent(),)  # MethodError: no method matching length(::NoTangent)

    # https://github.com/FluxML/Zygote.jl/issues/314
    @test_broken gradient((x,y) -> sum(yi -> yi*x, y), 1, [1,1]) == (2, [1, 1])
    @test_broken gradient((x,y) -> prod(yi -> yi*x, y), 1, [1,1]) == (2, [1, 1])

    @test_broken gradient((x,y) -> sum(map(yi -> yi*x, y)), 1, [1,1]) == (2, [1, 1])  # AssertionError: Base.issingletontype(typeof(f))
    @test_broken gradient((x,y) -> prod(map(yi -> yi*x, y)), 1, [1,1]) == (2, [1, 1])

    @test jacobicheck(x -> prod(x, dims = (2, 3)), randn(3,4,5))
    @test gradcheck(x -> prod(x), randn(3,4))
    @test gradient(x -> prod(x), (1,2,3))[1] == (6,3,2)
end

@testset "cumsum" begin
    @test_broken jacobicheck(x -> cumsum(x, dims=2), (3,4,5))  #  TypeError: in typeassert, expected Int64, got a value of type Nothing
    @test_broken jacobicheck(x -> cumsum(x, dims=1), (3,))
    @test_broken jacobicheck(x -> cumsum(x), (4,))
    @test_broken jacobicheck(x -> cumsum(x, dims=3), (5,))  # trivial
    @test_broken jacobicheck(x -> cumsum(x, dims=3), (3,4)) # trivial
end

# 146
@testset "getindex" begin
    @test jacobicheck(x -> x[:, 2, :], (3, 4, 5))
    @test jacobicheck(x -> x[1:2, 3:4], (3, 4))

    imat = [1 2; 3 4]
    @test jacobicheck(x -> x[:, imat], (3, 4))
    @test_broken jacobicheck(x -> x[:, [1, 2, 2]], (3, 4))
    irep = [1 2; 2 2]
    @test_broken jacobicheck(x -> x[1, irep], (3, 4))

    # https://github.com/invenia/Nabla.jl/issues/139
    x = rand(3)
    z = [1, 2, 3, 3]
    y139(x, z) = dot(ones(4), x[z])
    @test_broken gradient(y139, x, z) == ([1, 1, 2], nothing)  # ArgumentError: indexed assignment with a single value to possibly many locations is not supported; perhaps use broadcasting `.=` instead?

    # https://github.com/FluxML/Zygote.jl/issues/376
    _, back = pullback(x->x[1]*im, randn(2))
    @test back(1.0)[1] == real([-im, 0]) == [0, 0]

    # _droplike
    @test gradient(x -> sum(inv, x[1, :]'), ones(2, 2)) == ([-1 -1; 0 0],)
    @test gradient(x -> sum(inv, transpose(x[1, :])), ones(2, 2)) == ([-1 -1; 0 0],)  # same with transpose, in case ' overloaded!
    @test gradient(x -> sum(inv, x[1:1, :]'), ones(2, 2)) == ([-1 -1; 0 0],)
    @test gradient(x -> sum(inv, transpose(x[1:1, :])), ones(2, 2)) == ([-1 -1; 0 0],)
    @test_broken gradient(x -> sum(inv, transpose(view(x, 1, :))), ones(2, 2)) == ([-1 -1; 0 0],)

    # https://github.com/FluxML/Zygote.jl/issues/513
    @test_broken gradient(p -> sum(Float32[1, 0] - p), [2, 3]) == ([-1, -1],)
    @test_broken gradient(x -> sum(Float32[1, x] .+ x), 4) == (3.0f0,)  # MethodError: no method matching (::Diffractor.∂⃖recurse{1})(::typeof(Core.arrayset), ::Bool, ::Vector{Float32}, ::Float32, ::Int64)

    # Ensure that nothings work with numeric types.
    _, back = pullback(getindex, randn(4), [1])
    @test back([ZeroTangent()]) == (zeros(4), NoTangent())
    # Ensure that nothings work with non-numeric types.
    _, back = pullback(getindex, [randn(2) for _ in 1:3], [1])
    @test_broken back([ZeroTangent()]) == (NoTangent(), NoTangent())  # MethodError: no method matching zero(::Type{Vector{Float64}})
end

@test_skip @testset "view" begin  # Rewrite reached intrinsic function and_int. Missing rule?
    @test jacobicheck(x -> view(x,:,2,:), (3,4,5))
    @test jacobicheck(x -> view(x,1:2,3:4), (3,4))
    @test jacobicheck(x -> view(x,:,[1,2,2]), (3,4))

    # https://github.com/FluxML/Zygote.jl/issues/272
    g272(x) = view(x,1:2)[1]
    @test gradient(g272, ones(3)) == ([1,0,0],)
end

# 194
@testset "eachcol" begin
    @test_broken jacobicheck(x -> map(sum, eachcol(x)), (3,4))  # MethodError: no method matching one(::SubArray{Float64, 1, Matrix{Float64}, Tuple{Base.Slice{Base.OneTo{Int64}}, Int64}, true})
    @test_broken jacobicheck(x -> map(sum, eachcol(transpose(x))), (3,4))

    @test_broken jacobicheck(x -> map(norm, eachcol(x)), (3,4))
    @test_broken jacobicheck(x -> map(norm, eachrow(x)), (3,4))
    @test_broken jacobicheck(x -> map(norm, eachslice(x, dims=3)), (3,4,5))

    # some slices may have gradient nothing
    @test_broken gradient(x -> sum(y -> rand()>0.5 ? 0 : first(y), eachcol(x)), rand(3,10))[1] isa Matrix  # MethodError: no method matching lastindex(::Diffractor.OpticBundle{Int64})

    # strange errors
    @test gradient(x -> sum(norm, eachcol(x)), [1 2 3; 4 5 6])[1] isa Matrix
    @test gradient(x -> sum(norm, eachcol(x)), rand(3,400))[1] isa Matrix
end

@test_skip @testset "collect" begin
    @test gradient(x -> sum(inv, collect(x)), (1,2)) === ((-1.0, -1/4),)

    @test gradient(x -> sum(collect(view(x, 1:1))), rand(2)) == ([1,0],)
    @test gradient(x -> sum(inv, collect(view(x', 1,:))), ones(2,2)) == ([-1 0; -1 0],)

    @test gradient(xs -> sum(inv, [x^2 for x in xs]), ones(2)) == ([-2, -2],)
end

@testset "reverse" begin
    @test jacobicheck(x -> reverse(x), rand(17))
    @test jacobicheck(x -> reverse(x, 8), rand(17))
    @test jacobicheck(x -> reverse(x, 8, 13), rand(17))
    @test jacobicheck(x -> reverse(x, dims=2), rand(17, 42))
end

@testset "permutedims" begin
    @test jacobicheck(x -> permutedims(x), rand(2))
    @test_broken jacobicheck(x -> permutedims(x), rand(2,3))
    @test_broken jacobicheck(x -> permutedims(x, [3,1,2]), rand(4,5,6))
    @test_broken jacobicheck(x -> PermutedDimsArray(x, (3,1,2)), rand(4,5,6))
    let
      y, back = pullback(permutedims, randn(3))
      @test first(back(randn(1, 3))) isa Vector
    end
end

@testset "repeat" begin
    @test jacobicheck(x -> repeat(x; inner=2), rand(5))
    @test jacobicheck(x -> repeat(x; inner=2, outer=3), rand(5))
    @test jacobicheck(x -> repeat(x; inner=(2,2,1), outer=(1,1,3)), rand(5,4,3))

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
    @test_broken gradient(x -> fill(x, 3)[1][1], (1,2)) === ((1.0, nothing),)  # MethodError: no method matching zero(::Type{Tuple{Int64, Int64}})
    @test_broken gradient(x -> fill(x, 3)[1].a, (a=1, b=2)) == ((a=1.0, b=nothing),)  # 1 not 1.0
end

# 256
@testset "circshift" begin
    for D in 1:5
        x0 = zeros(ntuple(d->5, D))
        g = gradient(x -> x[1], x0)[1] 
        shift = ntuple(_ -> rand(-5:5), D)
        @test gradient(x -> circshift(x, shift)[1], x0)[1] == circshift(g, map(-, shift))
    end
end

# 273
@test_skip @testset "kron" begin
    @test jacobicheck(kron, 5, 3)  # TypeError: in typeassert, expected Int64, got a value of type Nothing
    @test jacobicheck(kron, rand(5), rand(3), rand(8))
    @test jacobicheck(kron, rand(5,1), rand(3,1))
    @test jacobicheck(kron, rand(5,1), rand(3,1), rand(8,1))
    @test jacobicheck(kron, rand(5,2), rand(3,2), rand(8,2))
end

# 279
@testset "map" begin
# ???
end

@testset "map and tuples" begin
    # arrays of tuples
    @test_broken gradient(x -> sum(map(first, x)), [(1,2), (3,4)]) == ([(1.0, NoTangent()), (1.0, NoTangent())],)  # MethodError: no method matching one(::Tuple{Int64, Int64})
    @test_broken gradient(x -> sum(first, x), [(1,2), (3,4)]) == ([(1.0, NoTangent()), (1.0, NoTangent())],)   # MethodError: no method matching lastindex(::Diffractor.OpticBundle{Int64})

    @test gradient(x -> map(+, x, (1,2,3))[1], (4,5,6)) == ((1.0, ZeroTangent(), ZeroTangent()),)
    @test_broken gradient(x -> map(+, x, [1,2,3])[1], (4,5,6)) == ((1.0, 0.0, 0.0),)  # MethodError: no method matching (::Diffractor.∂⃖recurse{1})(::typeof(Core.arrayset), ::Bool, ::Vector{Int64}, ::Int64, ::Int64)
    @test_broken gradient(x -> map(+, x, (1,2,3))[1], [4,5,6]) == ([1,0,0],)

    # mismatched lengths, should zip
    @test_broken gradient(x -> map(+, x, [1,2,3,99])[1], (4,5,6)) == ((1.0, 0.0, 0.0),) 
    @test_broken gradient(x -> map(+, x, [1,2,3])[1], (4,5,6,99)) == ((1.0, 0.0, 0.0, NoTangent()),)
end

# 420
@testset "filter" begin
    @test_broken jacobicheck(xs -> filter(x -> x > 0.5, xs), 20)  # MethodError: no method matching (::Diffractor.∂⃖recurse{1})(::typeof(Core.arrayset), ::Bool, ::Vector{Float64}, ::Float64, ::Int64)

    @test_broken gradient(x -> sum(log, filter(iseven, x)), 1:10) ==
        (map(x -> iseven(x) ? 1/x : 0, 1:10),)
    @test_broken gradient(x -> sum(abs2, im .+ filter(iseven, x)), 1:10) ==
        (map(x -> iseven(x) ? 2x : 0, 1:10),)
        # (map(x -> iseven(x) ? 2x+2im : 0, 1:10),)
end

# 494
@testset "maximum" begin
    @test jacobicheck(maximum, rand(2, 3))

    @test jacobicheck(x -> maximum(x, dims=1), rand(2, 3))
    @test jacobicheck(x -> maximum(x, dims=3), rand(2, 3, 4))
    @test jacobicheck(x -> maximum(x, dims=[1, 2]), rand(2, 3, 4))

    @test gradient(x -> 1 / maximum(x), [1., 2, 3])[1] == [0, 0, -1/9]
end

@testset "minimum" begin
    @test jacobicheck(minimum, rand(2, 3))

    @test jacobicheck(x -> minimum(x, dims=1), rand(2, 3))
    @test jacobicheck(x -> minimum(x, dims=2), rand(2, 3))
end

@testset "dropdims" begin  # https://github.com/JuliaDiff/Diffractor.jl/issues/72
    @test_broken jacobicheck(x -> dropdims(x, dims = 3), rand(2, 2, 1, 2))
    @test_broken jacobicheck(x -> dropdims(x, dims = (2, 3)), rand(2, 1, 1, 3))
end

# 1186
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
    catdim = (x...) -> cat(x..., dims = dim)
    @test jacobicheck(catdim, rand(4,1))
    @test jacobicheck(catdim, rand(5), rand(5,1))
    @test jacobicheck(catdim, rand(2,5), rand(2,5), rand(2,5)) 

    catdimval = (x...) -> cat(x...; dims = Val(dim))
    @test jacobicheck(catdimval, rand(5), rand(5))
    @test jacobicheck(catdimval, rand(2,5), rand(2,5,1))

    # one empty
    dim == 1 || continue
    @test jacobicheck(catdim, rand(0,5,3), rand(2,5,3))
end

# 1278
@testset "one(s) and zero(s)" begin
    @test gradient(x->sum(ones(size(x))), randn(5))[1] === NoTangent()
    @test gradient(x->sum(one(x)), randn(3, 3))[1] === NoTangent()
    @test gradient(x->sum(zeros(size(x))), randn(7))[1] === NoTangent()
    @test gradient(x->sum(zero(x)), randn(3))[1] === NoTangent()
end

@testset "fma and muladd" begin
    @test gradcheck(x -> fma(x[1], x[2], x[3]), [2.0, 3.0, 5.0])
    @test gradcheck(x -> muladd(x[1], x[2], x[3]), [2.0, 3.0, 5.0])
end

# 1388
@testset "broadcast" begin
    @test gradient(x -> sum(sin.(x)), Diagonal([0,pi/2,pi]))[1] ≈ [1 0 0; 0 0 0; 0 0 -1]

    # mixing arrays & Ref(array)
    a = rand(3)
    b = rand(2,2)
    @test_broken jacobicheck(x -> sum(diag.((x,) .* a)), b)
    @test_broken jacobicheck(x -> sum(diag.(Ref(x) .* a)), b)
    @test_broken jacobicheck(x -> sum(diag.([x] .* a)), b)

    # tests for https://github.com/FluxML/Zygote.jl/issues/724
    x1 = rand(3, 3)
    @test_broken gradient(x -> sum(x .== 0.5), x1)[1] === NoTangent()  # MethodError: no method matching length(::NoTangent)
    @test_broken gradient(x -> sum(x .* (x .== maximum(x, dims=1))), x1)[1] == (x1 .== maximum(x1, dims=1))  #  no method matching length(::NoTangent)

    # tests for un-broadcasting *, / via scalar rules
    @test all(gradient((x,y) -> sum(x .* y), [1,2], 5) .≈ ([5, 5], 3))
    @test all(gradient((x,y) -> sum(x .* y), 5, [1,2]) .≈ (3, [5, 5]))
    @test all(gradient((x,y) -> sum(x .* y), [1,2], [3 4 5]) .≈ ([12, 12], [3 3 3]))
    @test all(gradient((x,y) -> sum(x ./ y), [1,2], 5) .≈ ([0.2, 0.2], -0.12))
end

# 1489
@testset "array +,-" begin
  A, B = randn(3, 4, 5), randn(3, 4, 5)
  @test jacobicheck(+, B)
  @test jacobicheck(+, A, B)
  @test_broken jacobicheck(+, A, B, A)
  @test jacobicheck(-, A)
  @test_broken jacobicheck(-, A, B)  # in typeassert, expected Int64, got a value of type Nothing
end

# 1666
@testset "@nograd & friends" begin
    @test_broken gradient(x->eachindex([10,20,30])[1], 11) == (NoTangent(),)  # Rewrite reached intrinsic function and_int. Missing rule?

    @test gradient(x -> findfirst(ismissing, x), [1, missing]) == (NoTangent(),)
    @test gradient(x -> findlast(ismissing, x), [1, missing]) == (NoTangent(),)
    @test gradient(x -> findall(ismissing, x)[1], [1, missing]) == (NoTangent(),)

    # @test gradient(x -> Zygote.ignore(() -> x*x), 1) == (NoTangent(),) ?? replace with CRC versions?
    # @test gradient(x -> Zygote.@ignore(x*x), 1) == (NoTangent(),)
    # @test gradient(1) do x
    #     y = Zygote.@ignore x
    #     x * y
    # end == (1,)
end

# 1683
@testset "fastmath" begin
    @test gradient(x -> begin @fastmath sin(x) end, 1) == gradient(x -> sin(x), 1)
    @test gradient(x -> begin @fastmath tanh(x) end, 1) == gradient(x -> tanh(x), 1)
    @test gradient((x, y) -> begin @fastmath x*y end, 3, 2) == gradient((x, y) -> x*y, 3, 2)
    @test gradient(x -> begin @fastmath real(log(x)) end, 1 + 2im) == gradient(x -> real(log(x)), 1 + 2im)
end

# 1704
@testset "rand" begin
    @test gradient(x -> rand(), 1) == (ZeroTangent(),)
    @test gradient(x -> sum(rand(4)), 1) == (ZeroTangent(),)
    @test gradient(x -> sum(rand(Float32, (1,1))), 1) == (ZeroTangent(),)
    @test gradient(x -> sum(randn(Float32, 1,1)), 1) == (ZeroTangent(),)
    @test gradient(x -> sum(randexp(Float32, (1,1))), 1) == (ZeroTangent(),)

    rng = Random.default_rng()
    @test gradient(x -> sum(rand(rng, 4)), 1) == (ZeroTangent(),)
    @test gradient(x -> sum(rand(rng, Float32, 1,1)), 1) == (ZeroTangent(),)
    @test gradient(x -> sum(randn(rng, Float32, (1,1))), 1) == (ZeroTangent(),)
    @test gradient(x -> sum(randexp(rng, Float32, 1,1)), 1) == (ZeroTangent(),)
end

# 1737
@testset "broadcasted($op, Array, Bool)" for op in (+,-,*)
    @testset "with $bool and sizes $s" for s in ((4,), (2,3)), bool in (true,false)
        r = rand(Int8, s) .+ 0.0
        z = fill(bool, s) .+ 0.0

        fun(args...) = pullback((x, y) -> sum(op.(x,y)), args...)[1]
        gfun(args...) = gradient((x, y) -> sum(op.(x,y)), args...)

        @test fun(r, z) == fun(r, bool)
        @test gfun(r, bool) == (gfun(r, z)[1], NoTangent())

        @test fun(z, r) == fun(bool, r)
        @test gfun(bool, r) == (NoTangent(), gfun(z, r)[2])
    end
end

@testset "misc issues" begin

    # https://github.com/FluxML/Zygote.jl/issues/957
    @test_broken gradcheck(x -> prod(Base.Fix1(+, 1), x), randn(10))
    @test_broken gradcheck(x -> prod(Base.Fix2(+, 1), x), randn(10))

    # https://github.com/FluxML/Zygote.jl/issues/996
    @test gradient(x->sum(x .+ rand.()), rand(3)) == (ones(3),)

    # https://github.com/FluxML/Zygote.jl/pull/660
    function example660(x,N)
        ax = axes(x)
        extraAxe = ax[2+N:end]
        filledLoc = fill(1, N)
        return x[:, filledLoc..., extraAxe...]
    end
    y, back = pullback(example660, randn(5,3,4,3), 2)
    @test back(zero(y) .= 1) isa Tuple{Array{Float64,4}, ZeroTangent}

    # https://github.com/JuliaDiff/ChainRulesCore.jl/issues/440
    f440(x,y) = sum(sum, [[x[i],y[i]] for i=1:length(x)])
    g440(x,y) = sum(sum, [(x[i],y[i]) for i=1:length(x)])
    @test_broken gradient(f440, rand(3), rand(3)) == ([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
    @test_broken gradient(g440, rand(3), rand(3)) == ([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])

@test_skip begin

    # https://github.com/FluxML/Zygote.jl/issues/804
    # Comprehension is used.
    io = IOBuffer()
    s = 0.0
    gs = gradient([1.0, 2.0]) do xs   # UndefVarError: s not defined
        sum([(print(io, x); s += x; s * x) for x in xs])
    end
    @test String(take!(io)) == "1.02.0"
    @test s == 3.0
    @test gs == ([4.0, 5.0],)

    # Comprehension is not used.
    io = IOBuffer()
    s = 0.0
    gs = gradient([1.0, 2.0]) do xs
        sum([(print(io, x); s += x; s * x) for x in xs])
        0.0
    end
    @test String(take!(io)) == "1.02.0"
    @test s == 3.0
    @test gs == (nothing,)

    # Comprehension is empty and not used.
    io = IOBuffer()
    s = 0.0
    gs = gradient([]) do xs
        [(print(io, x); s += x; s * x) for x in xs]
        0.0
    end
    @test String(take!(io)) == ""
    @test s == 0.0
    @test gs == (nothing,)

end # skip

end

#####
##### Zygote/test/gradcheck.jl : LinearAlgebra
#####

@testset "LinearAlgebra misc" begin
    @test jacobicheck(x -> x', rand(5))
    @test jacobicheck(x -> adjoint(x), rand(5))
    @test jacobicheck(tr, rand(4, 4))
end

# 140
@test_skip @testset "LinearAlgebra.det" begin  # ArgumentError: Tangent for the primal Base.Pairs{Symbol, Union{}, Tuple{}, NamedTuple{(), Tuple{}}} should be backed by a AbstractDict type
    @test jacobicheck(det, (4, 4))
    @test jacobicheck(logdet, map(x -> x*x', (rand(4, 4),))[1])
    @test jacobicheck(x -> logabsdet(x)[1], (4, 4))
    @test gradient(det, 2.0)[1] == 1
    @test gradient(logdet, 2.0)[1] == 0.5
end

# 266
@testset "LinearAlgebra.dot" begin
    @test gradcheck((x, y)->dot(x[1], y[1]), [randn()], [randn()])
    @test gradcheck(dot, randn(10), randn(10))
    @test gradcheck(dot, randn(10, 3), randn(10, 3))
end

# 537
@testset "LinearAlgebra.(p)inv" begin
  P, Q = 13, 11
  A, B, C = randn(P, Q), randn(P, P), randn(Q, P)
  @test jacobicheck(pinv, A)
  @test jacobicheck(inv, B)
  @test jacobicheck(pinv, C)

  @test gradient(inv, 2.0)[1] == -0.25
end

@testset "LinearAlgebra: *" begin
    M, P, Q = 13, 7, 11
    @test jacobicheck(*, randn(M, P), randn(P, Q))
    @test jacobicheck(*, randn(M, P), randn(P))
    @test jacobicheck(*, randn(M, 1), randn(1, Q))
    @test jacobicheck(*, randn(M), randn(1, Q))
    @test jacobicheck(*, randn(10)', randn(10))
    @test jacobicheck(*, transpose(randn(10)), randn(10))
    @test jacobicheck(*, randn(10)', randn(10))
    @test jacobicheck(*, transpose(randn(10)), randn(10))
end

# 1383
@testset "matrix multiplication size" begin
    @test size(gradient((x, y)->sum(x * y), randn(1, 1), randn(1, 10))[1]) == (1, 1)
    @test size(gradient((x, y)->sum(x * y), randn(1, 1), randn(1, 10))[2]) == (1, 10)
end

@testset "backsolve" begin
    rng, M, P, Q = MersenneTwister(123456), 13, 10, 9
    X, Y, y = randn(rng, P, P), randn(rng, P, Q), randn(rng, P)
    A, B = randn(rng, P, M), randn(P, Q)
    D = collect(Diagonal(randn(rng, P)))
    U = collect(UpperTriangular(randn(rng, P, P)))
    U[diagind(U)] .= 1 .+ 0.01 .* randn(rng, P)

    # \ (Dense square)
    @test jacobicheck(\, X, Y)
    @test jacobicheck(\, X, y)

    # \ (Dense rectangular)
    @test jacobicheck(\, A, Y)
    @test jacobicheck(\, A, y)
    @test jacobicheck(\, B, Y)
    @test jacobicheck(\, B, y)

    # \ (Diagonal)
    @test jacobicheck(\, D, Y)
    @test jacobicheck(\, D, y)
    @test jacobicheck((D, Y)-> Diagonal(D) \ Y, D, Y)
    @test jacobicheck((D, Y)-> Diagonal(D) \ Y, D, y)

    # \ (UpperTriangular)
    @test jacobicheck(\, U, Y)
    @test jacobicheck(\, U, y)
    @test jacobicheck((U, Y) -> UpperTriangular(U) \ Y, U, Y)
    @test jacobicheck((U, Y) -> UpperTriangular(U) \ Y, U, y)

    # /
    @test jacobicheck(/, Y', X)
    @test jacobicheck((y, X)->y' / X, y, X)

    # / (rectangular)
    @test jacobicheck(/, Y', A')
    @test jacobicheck((y, A)->y' / A', y, B)

    # / (Diagonal)
    @test jacobicheck((D, Y) -> Y' / D, D, Y)
    @test jacobicheck((D, Y)-> Y' / Diagonal(D), D, y)

    # / (UnitUpperTriangular)
    @test jacobicheck((U, Y) -> Y' / U, U, Y)
    @test_broken jacobicheck((U, Y) -> Y' / UnitUpperTriangular(U), U, y)  # MethodError: no method matching isapprox(::ChainRules.var"#1711#1714"{Adjoint{Float64, Vector{Float64}}, UnitUpperTriangular{Float64, Matrix{Float64}}, ProjectTo{UnitUpperTriangular, NamedTuple{(:parent,), Tuple{ProjectTo{AbstractArray, NamedTuple{(:element, :axes), Tuple{ProjectTo{Float64, NamedTuple{(), Tuple{}}}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}}}}}}}, Adjoint{Float64, Vector{Float64}}}, ::Matrix{Float64}; rtol=1.0e-5, atol=1.0e-5)
end

# 667
@testset "LinearAlgebra.Symmetric" begin  # ?? copy something?

end

# 771
@testset "LinearAlgebra.diag" begin
    @test jacobicheck(diag, (10, 10))
    @test jacobicheck(x -> diag(x, 2), (10, 13))
end

@testset "LinearAlgebra.Diagonal" begin
    x = randn(5)
    @test jacobicheck(Diagonal, x)

    y, back = pullback(Diagonal, x)
    D = randn(5,5)
    @test back(D)[1] ≈ back(Diagonal(D))[1]
    @test back(D)[1] ≈ back(Tangent{Diagonal}(; diag=diag(D)))[1]
end

@testset "dense + UniformScaling" begin
    A, λ = randn(10, 10), randn()
    @test_broken jacobicheck(A->A + 5I, A)  # (1, unsafe_copyto!(dest::Array{T}, doffs, src::Array{T}, soffs, n) where T in Base at array.jl:280, :($(Expr(:gc_preserve_end, :(%1)))))
    @test_broken jacobicheck(A->5I - A, A)
    @test_broken jacobicheck(λ->A + λ[1] * I, [λ])
end

# 795
@testset "LinearAlgebra.cholesky" begin

end

# 825
@testset "LinearAlgebra.lyap" begin

end

@testset "matrix exponential" begin
# ??
end

# 1767
@testset "LinearAlgebra.norm" begin
    # check that type is not unnecessarily promoted
    # https://github.com/FluxML/Zygote.jl/issues/663
    @test unthunk.(gradient(norm, randn(Float32, 2, 2))) isa Tuple{Matrix{Float32}}
    @test_broken unthunk.(gradient(norm, randn(Float32, 2, 2), 3)) isa Tuple{Matrix{Float32},Float64}  # Float32 is OK?
    @test unthunk.(gradient(norm, randn(Float32, 2, 2), 3f0)) isa Tuple{Matrix{Float32},Float32}
    @test unthunk.(gradient(norm, randn(ComplexF32, 2, 2), 3.5f0)) isa Tuple{Matrix{ComplexF32},Float32}

    # just check that these do not error
    # https://github.com/FluxML/Zygote.jl/issues/331
    gradient(x->norm(x*[1, 1]), 1.23)
    gradient(x->norm(x*[1 1]), 1.23)
    gradient(x->norm(x*[1im, 1]), 1.23)
    gradient(x->norm(x*[1im 1]), 1.23)
end

# 1690
@testset "LinearAlgebra.I |> Matrix" begin
    @test_broken gradient(x -> sum(Matrix(x*I, 2, 2)), 1.0) == (2.0,)  # MethodError: no method matching (::Diffractor.∂⃖recurse{1})(::typeof(Core.arrayset), ::Bool, ::Matrix{Float64}, ::Float64, ::Int64)

    @test_broken gradient(x -> sum(Matrix(x[1]*I, (2, 2))), [1.0]) == ([2.0],)
    @test_broken gradient(x -> sum(Matrix{Float64}(x[1]*I, 2, 2)), [1.0]) == ([2.0],)

    # Check we haven't broken the forward pass:
    @test_broken first(pullback(x->Matrix(x*I, 2,2), 8.0)) == Matrix(8.0*I, 2,2)
end


#####
##### Zygote/test/gradcheck.jl : Statistics
#####

# 430
@test_skip @testset "Statistics.mean" begin
    @test jacobicheck(mean, rand(2, 3))

    @test jacobicheck(x -> mean(x, dims=1), rand(2, 3))
    @test jacobicheck(x -> mean(x, dims=2), rand(2, 3))
    @test jacobicheck(x -> mean(x, dims=3), rand(2, 3, 4))

    @test jacobicheck(x -> mean(x, dims=[1, 2]), rand(2, 3, 4))
end

@test_skip @testset "Statistics.$var" for var in (std, var)
    @test jacobicheck(var, rand(2, 3))
    @test jacobicheck(x -> var(x, dims=2), rand(2, 3))
    @test jacobicheck(x -> var(x, dims=(1, 2)), rand(2, 3, 4))

    @test jacobicheck(x -> var(x, corrected=false), rand(2, 3))
    @test jacobicheck(x -> var(x, dims=1, corrected=false), rand(2, 3))

    @test jacobicheck(x -> var(x, mean=mean(x)), rand(2, 3))
    @test jacobicheck(x -> var(x, dims=2, mean=mean(x, dims=2)), rand(2, 3))
    @test jacobicheck(x -> var(x, dims=(1, 2), mean=mean(x, dims=(1, 2))), rand(2, 3, 4))

    @test jacobicheck(x -> var(x, corrected=false, mean=mean(x)), rand(2, 3))
    @test jacobicheck(x -> var(x, dims=1, corrected=false, mean=mean(x, dims=1)), rand(2, 3))
end
