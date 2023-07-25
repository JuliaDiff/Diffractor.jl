using AbstractDifferentiation, Diffractor, Test
include(joinpath(pathof(AbstractDifferentiation), "..", "..", "test", "test_utils.jl"))
import AbstractDifferentiation as AD

backend = Diffractor.DiffractorForwardBackend()
@testset "ForwardDiffBackend" begin
    backends = [
        @inferred(Diffractor.DiffractorForwardBackend())
    ]
    @testset for backend in backends
        @test backend isa AD.AbstractForwardMode

        @testset "Derivative" begin #setfield!(::Core.Box, ::Symbol, ::Float64)
            @test_broken test_derivatives(backend)
        end
        @testset "Gradient" begin #Diffractor.TangentBundle{1, Float64, Diffractor.TaylorTangent{Tuple{Float64}}}(::Float64, ::Tuple{Float64})
            @test_broken test_gradients(backend)
        end
        @testset "Jacobian" begin #setfield!(::Core.Box, ::Symbol, ::Vector{Float64})
            @test_broken test_jacobians(backend)
        end
        @testset "Hessian" begin #setindex!(::ChainRulesCore.ZeroTangent, ::Float64, ::Int64)
            @test_broken test_hessians(backend)
        end
        @testset "jvp" begin #setfield!(::Core.Box, ::Symbol, ::Vector{Float64})
            @test_broken test_jvp(backend; vaugmented=true)
        end
        @testset "j′vp" begin #setfield!(::Core.Box, ::Symbol, ::Vector{Float64})
            @test_broken test_j′vp(backend)
        end
        @testset "Lazy Derivative" begin
            test_lazy_derivatives(backend)
        end
        @testset "Lazy Gradient" begin #Diffractor.TangentBundle{1, Float64, Diffractor.TaylorTangent{Tuple{Float64}}}(::Float64, ::Tuple{Float64})
            @test_broken test_lazy_gradients(backend)
        end
        @testset "Lazy Jacobian" begin
            test_lazy_jacobians(backend; vaugmented=true)
        end
        @testset "Lazy Hessian" begin # everything everywhere all at once is broken
            @test_broken test_lazy_hessians(backend)
        end
    end
end
