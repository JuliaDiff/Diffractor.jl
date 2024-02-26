module AbstractDifferentiationTests

using AbstractDifferentiation, Diffractor, Test, LinearAlgebra, ChainRulesCore
import AbstractDifferentiation as AD
backend = Diffractor.DiffractorForwardBackend()

@testset "bundle" begin
    bundle = Diffractor.bundle

    @test bundle(1.0, 2.0) isa Diffractor.TaylorBundle{1}
    @test bundle([1.0, 2.0], [2.0, 3.0]) isa Diffractor.TaylorBundle{1}
    @test bundle(1.5=>2.5, Tangent{Pair{Float64, Float64}}(first=1.0, second=2.0)) isa Diffractor.TaylorBundle{1}
    @test bundle(1.1, ChainRulesCore.ZeroTangent()) isa Diffractor.ZeroBundle{1}
    @test bundle(1.5=>2.5=>3.5, Tangent{Pair{Float64, Pair{Float64, Float64}}}(first=1.0, second=Tangent{Pair{Float64, Float64}}(first=1.0, second=2.0))) isa Diffractor.TaylorBundle{1}
    
    # noncanonical structural tangent
    b = bundle(1.5=>2.5=>3.5, Tangent{Pair{Float64, Pair{Float64, Float64}}}(second=Tangent{Pair{Float64, Float64}}(second=2.0, first=1.0)))
    t = Diffractor.first_partial(b)
    @test b isa Diffractor.TaylorBundle{1}
    @test iszero(t.first)
    @test t.second.first == 1.0
    @test t.second.second == 2.0
end

@testset "basics" begin
    @test AD.derivative(backend, +, 1.5, 10.0) == (1.0, 1.0)
    @test AD.derivative(backend, *, 1.5, 10.0) == (10.0, 1.5)
    @test only(AD.jacobian(backend, prod, [1.5, 2.5, 10.0])) == [25.0  15.0  3.75]
    @test only(AD.jacobian(backend, identity, [1.5, 2.5, 10.0])) == Matrix(I, 3, 3)
end

# standard tests from AbstractDifferentiation.test_utils
include(joinpath(pathof(AbstractDifferentiation), "..", "..", "test", "test_utils.jl"))
@testset "Standard AbstractDifferentiation.test_utils tests" begin
    backends = [
        @inferred(Diffractor.DiffractorForwardBackend())
    ]
    @testset for backend in backends
        @test backend isa AD.AbstractForwardMode

        @testset "Derivative" begin 
            test_derivatives(backend)
        end
        @testset "Gradient" begin # no method matching size(::ZeroTangent)
            @test_broken test_gradients(backend)
        end
        @testset "Jacobian" begin #setfield!(::Core.Box, ::Symbol, ::Vector{Float64})
            test_jacobians(backend)
        end
        @testset "Hessian" begin # no method matching (Diffractor.TangentBundle{…} where P)(::Tuple{…}, ::Tuple{…})
            @test_broken test_hessians(backend)
        end
        @testset "jvp" begin # Expression: pf1[1] isa Vector{Float64}
            @test_broken false # test_jvp(backend; vaugmented=true)
        end
        @testset "j′vp" begin # no method matching *(::Diffractor.PrimeDerivativeBack{1, Diagonal{Bool, Vector{Bool}}}, ::Vector{Float64})
            @test_broken false #test_j′vp(backend)
        end
        @testset "Lazy Derivative" begin
            test_lazy_derivatives(backend)
        end
        @testset "Lazy Gradient" begin #MethodError: no method matching size(::ZeroTangent) 
            @test_broken test_lazy_gradients(backend)
        end
        @testset "Lazy Jacobian" begin # no method matching *(::Diffractor.PrimeDerivativeBack{1, Diagonal{Bool, Vector{Bool}}}, ::Vector{Float64})
            @test_broken false #test_lazy_jacobians(backend)
        end
        @testset "Lazy Hessian" begin # ERROR: MethodError: no method matching *(::Vector{Float64}, ::Diffractor.PrimeDerivativeBack{1, Vector{Float64}})
            @test_broken false #test_lazy_hessians(backend)
        end
    end
end

end #module AbstractDifferentiationTests
