
# This file has integration tests for some rules defined in ChainRules.jl,
# especially those which aim to support higher derivatives, as properly
# testing those is difficult. Organised according to the files in CR.jl.

using Diffractor, ForwardDiff, ChainRulesCore
using Test, LinearAlgebra

using Test: Threw, eval_test


#####
##### Base/array.jl
#####





#####
##### Base/arraymath.jl
#####




#####
##### Base/base.jl
#####





#####
##### Base/indexing.jl
#####

@testset "getindex, first" begin
    @test_broken gradient(x -> sum(abs2, gradient(first, x)[1]), [1,2,3])[1] == [0, 0, 0]  # MethodError: no method matching +(::Tuple{ZeroTangent, ZeroTangent}, ::Tuple{ZeroTangent, ZeroTangent})
    @test_broken gradient(x -> sum(abs2, gradient(sqrt∘first, x)[1]), [1,2,3])[1] ≈ [-0.25, 0, 0]  # error() in perform_optic_transform(ff::Type{Diffractor.∂⃖recurse{2}}, args::Any)
    @test gradient(x -> sum(abs2, gradient(x -> x[1]^2, x)[1]), [1,2,3])[1] == [8, 0, 0]
    @test_broken gradient(x -> sum(abs2, gradient(x -> sum(x[1:2])^2, x)[1]), [1,2,3])[1] == [48, 0, 0]  # MethodError: no method matching +(::Tuple{ZeroTangent, ZeroTangent}, ::Tuple{ZeroTangent, ZeroTangent})
end

@testset "eachcol etc" begin
    @test gradient(m -> sum(prod, eachcol(m)), [1 2 3; 4 5 6])[1] == [4 5 6; 1 2 3]
    @test gradient(m -> sum(first, eachcol(m)), [1 2 3; 4 5 6])[1] == [1 1 1; 0 0 0]
    @test gradient(m -> sum(first(eachcol(m))), [1 2 3; 4 5 6])[1] == [1 0 0; 1 0 0]
    @test_skip gradient(x -> sum(sin, gradient(m -> sum(first(eachcol(m))), x)[1]), [1 2 3; 4 5 6])[1]  # MethodError: no method matching one(::Base.OneTo{Int64}), unzip_broadcast, split_bc_forwards
end

#####
##### Base/mapreduce.jl
#####

@testset "sum" begin
    @test gradient(x -> sum(abs2, gradient(sum, x)[1]), [1,2,3])[1] == [0,0,0]
    @test gradient(x -> sum(abs2, gradient(x -> sum(abs2, x), x)[1]), [1,2,3])[1] == [8,16,24]

    @test gradient(x -> sum(abs2, gradient(sum, x .^ 2)[1]), [1,2,3])[1] == [0,0,0]
    @test gradient(x -> sum(abs2, gradient(sum, x .^ 3)[1]), [1,2,3])[1] == [0,0,0]
end

@testset "foldl" begin

    @test gradient(x -> foldl(*, x), [1,2,3,4])[1] == [24.0, 12.0, 8.0, 6.0]
    @test gradient(x -> foldl(*, x; init=5), [1,2,3,4])[1] == [120.0, 60.0, 40.0, 30.0]
    @test gradient(x -> foldr(*, x), [1,2,3,4])[1] == [24, 12, 8, 6]

    @test gradient(x -> foldl(*, x), (1,2,3,4))[1] == Tangent{NTuple{4,Int}}(24.0, 12.0, 8.0, 6.0)
    @test_broken gradient(x -> foldl(*, x; init=5), (1,2,3,4))[1] == Tangent{NTuple{4,Int}}(120.0, 60.0, 40.0, 30.0)  # does not return a Tangent
    @test gradient(x -> foldl(*, x; init=5), (1,2,3,4)) |> only |> Tuple == (120.0, 60.0, 40.0, 30.0)
    @test_broken gradient(x -> foldr(*, x), (1,2,3,4))[1] == Tangent{NTuple{4,Int}}(24, 12, 8, 6)
    @test gradient(x -> foldr(*, x), (1,2,3,4)) |> only |> Tuple == (24, 12, 8, 6)

end


#####
##### LinearAlgebra/dense.jl
#####


@testset "dot" begin

    @test gradient(x -> dot(x, [1,2,3])^2, [4,5,6])[1] == [64,128,192]
    @test_broken gradient(x -> sum(gradient(x -> dot(x, [1,2,3])^2, x)[1]), [4,5,6])[1] == [12,24,36]  # MethodError: no method matching +(::Tuple{Tangent{ChainRules.var

end


#####
##### LinearAlgebra/norm.jl
#####


