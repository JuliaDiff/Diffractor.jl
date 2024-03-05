using Diffractor
using StaticArrays
using ChainRulesCore
using Test

@testset "StaticArrays constructor" begin
    #frule(::Tuple{ChainRulesCore.NoTangent, ChainRulesCore.Tangent{Tuple{Int64, Vararg{Float64, 9}}, Tuple{Int64, Vararg{Float64, 9}}}}, ::Type{StaticArraysCore.SVector{10, Float64}}, x::Tuple{Int64, Vararg{Float64, 9}})
    #     @ Diffractor ~/.julia/packages/Diffractor/yCsbI/src/extra_rules.jl:183

    @testset "homogenious type" begin
        x = (10.0, 20.0, 30.0)
        ẋ = zero_tangent(x)
        y, ẏ = frule((NoTangent(), ẋ), StaticArraysCore.SVector{3, Float64}, x)
        @test y == @SVector [10.0, 20.0, 30.0]
        @test ẏ == @SVector [0.0, 0.0, 0.0]
    end

    @testset "convertable type" begin
        x = (10, 20.0, 30.0)
        ẋ = zero_tangent(x)
        y, ẏ = frule((NoTangent(), ẋ), StaticArraysCore.SVector{3, Float64}, x)
        # all are float
        @test y == @SVector [10.0, 20.0, 30.0]
        @test ẏ == @SVector [0.0, 0.0, 0.0]
    end

    @testset "convertable type with ZeroTangent()" begin
        x = (10, 20.0, 30.0)
        ẋ = Tangent{typeof(x)}(ZeroTangent(), 1.0, 2.0)
        y, ẏ = frule((NoTangent(), ẋ), StaticArraysCore.SVector{3, Float64}, x)
        # all are float
        @test y == @SVector [10.0, 20.0, 30.0]
        @test ẏ == @SVector [0.0, 1.0, 2.0]
    end
end