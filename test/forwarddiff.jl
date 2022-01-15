
# This file contains tests adapted from FowardDiff.jl, many of them using DiffTests.jl
# Organised here file-by-file in alphabetical order!

#####
##### setup
#####

using Test, LinearAlgebra
using ForwardDiff, DiffTests
using Diffractor, ChainRulesCore

# Define functions which behave like the ones ForwardDiff doesn't export.
# These have plenty of sharp edges!
begin

    fwd_derivative(f, x::Number) = Diffractor.PrimeDerivativeFwd(f)(float(x)) |> unthunk
    function rev_derivative(f, x::Number)
        y = f(x)
        if y isa Number || y isa AbstractZero
            Diffractor.PrimeDerivativeBack(f)(float(x)) |> unthunk
        elseif y isa AbstractArray
            map(CartesianIndices(y)) do I
                Diffractor.PrimeDerivativeBack(x -> f(x)[I])(float(x)) |> unthunk
            end
        else
            throw("rev_derivative can't handle f(x)::$(typeof(y))")
        end
    end

    @test ForwardDiff.derivative(abs2, 3) == 6
    @test fwd_derivative(abs2, 3) == 6
    @test rev_derivative(abs2, 3) == 6

    @test ForwardDiff.derivative(x -> fill(x,2,3), 7) == [1 1 1; 1 1 1]
    @test fwd_derivative(x -> fill(x,2,3), 7) == [1 1 1; 1 1 1]
    @test rev_derivative(x -> fill(x,2,3), 7) == [1 1 1; 1 1 1]

    DERIVATIVES = (ForwardDiff.derivative, fwd_derivative, rev_derivative)

    function fwd_gradient(f, x::AbstractVector)
        map(eachindex(x)) do i
            fwd_derivative(ξ -> f(vcat(x[begin:i-1], ξ, x[i+1:end])), x[i])
        end
    end
    fwd_gradient(f, x::AbstractArray) = reshape(fwd_gradient(v -> f(reshape(v, size(x))), vec(x)), size(x))
    rev_gradient(f, x::AbstractArray) = ChainRulesCore.unthunk(Diffractor.PrimeDerivativeBack(f)(float(x)))

    @test ForwardDiff.gradient(prod, [1,2,3]) == [6,3,2]
    @test fwd_gradient(prod, [1,2,3]) == [6,3,2]
    @test rev_gradient(prod, [1,2,3]) == [6,3,2]

    @test fwd_gradient(sum, [1,2]) == [1,1]
    @test fwd_gradient(first, [1,1]) == [1,0]

    GRADIENTS = (ForwardDiff.gradient, rev_gradient)

    fwd_jacobian(f, x::AbstractArray) = hcat(vec.(fwd_gradient(f, x))...)
    function rev_jacobian(f, x::AbstractArray)
        y = f(x)
        slices = map(LinearIndices(y)) do i  # fails if y isa Number, just like ForwardDiff.jacobian
            vec(rev_gradient(x -> f(x)[i], x))
        end
        vcat(transpose(slices)...)
        # permutedims(hcat(slices...))
    end

    @test ForwardDiff.jacobian(x -> x[1:2], [1,2,3]) == [1 0 0; 0 1 0]
    @test fwd_jacobian(x -> x[1:2], [1,2,3]) == [1 0 0; 0 1 0]
    @test rev_jacobian(x -> x[1:2], [1,2,3]) == [1 0 0; 0 1 0]

    JACOBIANS = (ForwardDiff.jacobian, fwd_jacobian, rev_jacobian)

    fwd_hessian(f, x::AbstractArray) = fwd_jacobian(y -> fwd_gradient(f, y), float(x))
    rev_hessian(f, x::AbstractArray) = rev_jacobian(y -> rev_gradient(f, y), float(x))
    fwd_rev_hessian(f, x::AbstractArray) = fwd_jacobian(y -> rev_gradient(f, y), float(x))
    rev_fwd_hessian(f, x::AbstractArray) = rev_jacobian(y -> fwd_gradient(f, y), float(x))

    @test ForwardDiff.hessian(x -> -log(x[1]), [2,3]) == [0.25 0; 0 0]
    @test_broken fwd_hessian(x -> -log(x[1]), [2,3]) == [0.25 0; 0 0]  # UndefVarError: B not defined
    @test rev_hessian(x -> -log(x[1]), [2,3]) == [0.25 0; 0 0]
    @test_broken rev_fwd_hessian(x -> -log(x[1]), [2,3]) == [0.25 0; 0 0]  # MethodError: no method matching (::Diffractor.∂⃖recurse{1})(::typeof(Core.arrayset), ::Bool, ::Vector{Float64}, ::Float64, ::Int64)
    @test_skip fwd_rev_hessian(x -> -log(x[1]), [2,3])  # Internal error: encountered unexpected error in runtime: AssertionError(msg="argextype only works on argument-position values")  MethodError: no method matching (::Core.OpaqueClosure{Tuple{Any}, Any})(::Float64)

    HESSIANS = (ForwardDiff.hessian, rev_hessian)

end

#####
##### ConfusionTest
#####

@testset verbose=true "ConfusionTest" begin

    # Perturbation Confusion (Issue #83) #
    #------------------------------------#

    @testset "issue 83: perturbation confusion 1" begin # for D in DERIVATIVES

        g = ForwardDiff.gradient(v -> sum(v) * norm(v), [1])
        @testset "ForwardDiff.derivative" begin
            D = ForwardDiff.derivative

            @test D(x -> x * D(y -> x + y, 1), 1) == 1
            
            @test ForwardDiff.gradient(v -> sum(v) * D(y -> y * norm(v), 1), [1]) == g
            @test_broken fwd_gradient(v -> sum(v) * D(y -> y * norm(v), 1), [1]) == g
            @test_broken rev_gradient(v -> sum(v) * D(y -> y * norm(v), 1), [1]) == g

        end
        @testset "fwd_derivative" begin
            D = fwd_derivative

            @test_broken D(x -> x * D(y -> x + y, 1), 1) == 1  # UndefVarError: B not defined

            @test ForwardDiff.gradient(v -> sum(v) * D(y -> y * norm(v), 1), [1]) == g
            @test_broken fwd_gradient(v -> sum(v) * D(y -> y * norm(v), 1), [1]) == g
            @test rev_gradient(v -> sum(v) * D(y -> y * norm(v), 1), [1]) == g

        end
        @testset "rev_derivative" begin
            D = rev_derivative

            @test_broken D(x -> x * D(y -> x + y, 1), 1) == 1  # MethodError: no method matching +(::Float64, ::Tangent{var"#269#271"{Float64}, NamedTuple{(:x,), Tuple{ZeroTangent}}})

            @test ForwardDiff.gradient(v -> sum(v) * D(y -> y * norm(v), 1), [1]) == g  
            @test_broken fwd_gradient(v -> sum(v) * D(y -> y * norm(v), 1), [1]) == g  # Internal error: encountered unexpected error in runtime: AssertionError(msg="argextype only works on argument-position values")
            @test_broken rev_gradient(v -> sum(v) * D(y -> y * norm(v), 1), [1]) == g  # MethodError: no method matching ndims(::ZeroTangent)

        end

    end

    @testset "issue 83: perturbation confusion 2, $jacobian + $gradient" for jacobian in JACOBIANS, gradient in GRADIENTS

        A = rand(10,8)
        y = rand(10)
        x = rand(8)

        @test A == jacobian(x) do x
            gradient(y) do y
                dot(y, A*x)
            end
        end  broken = jacobian != ForwardDiff.jacobian

    end

    # Issue #238                         #
    #------------------------------------#

    @testset "issue 238: legendre transformation 1, $jacobian + $gradient" for jacobian in JACOBIANS, gradient in GRADIENTS

        m,g = 1, 9.8
        t = 1
        q = [1,2]
        q̇ = [3,4]
        L(t,q,q̇) = m/2 * dot(q̇,q̇) - m*g*q[2]

        ∂L∂q̇(L, t, q, q̇) = ForwardDiff.gradient(a->L(t,q,a), q̇)
        Dqq̇(L, t, q, q̇) = ForwardDiff.jacobian(a->∂L∂q̇(L,t,a,q̇), q)
        @test Dqq̇(L, t, q, q̇)  == fill(0.0, 2, 2)

    end
    @testset "issue 238: legendre transformation 2, $hessian + $gradient" for hessian in HESSIANS, gradient in GRADIENTS

        m,g = 1, 9.8
        t = 1
        q = [1,2]
        q̇ = [3,4] .+ 0.0
        L(t,q,q̇) = m/2 * dot(q̇,q̇) - m*g*q[2]

        q = [1,2] .+ 0.0
        p = [5,6] .+ 0.0
        function Legendre_transformation(F, w)
            z = fill(0.0, size(w))
            M = hessian(F, z)
            b = gradient(F, z)
            v = cholesky(M)\(w-b)
            dot(w,v) - F(v)
        end
        function Lagrangian2Hamiltonian(Lagrangian, t, q, p)
            L = q̇ -> Lagrangian(t, q, q̇)
            Legendre_transformation(L, p)
        end

        @test Lagrangian2Hamiltonian(L, t, q, p) isa Number  broken = hessian==rev_hessian  # ?? InexactError: Int64(-9.8)
        @test_broken gradient(a->Lagrangian2Hamiltonian(L, t, a, p), q) == [0.0,g]

    end

    @testset "issue 267: let scoping $hessian" for hessian in HESSIANS

        @noinline f83a(z, x) = x[1]
        z83a = ([(1, (2), [(3, (4, 5, [1, 2, (3, (4, 5), [5])]), (5))])])
        let z = z83a
            g = x -> f83a(z, x)
            h = x -> g(x)
            @test hessian(h, [1.]) == zeros(1, 1)  broken = hessian != ForwardDiff.hessian
        end

    end

    @testset "simple 2nd order $derivative" for derivative in DERIVATIVES

        @test derivative(1.0) do x
            derivative(x) do y
                x
            end
        end == 0.0  broken = derivative != ForwardDiff.derivative
        # Cotangent space not defined for `ZeroTangent`. Try a real-valued function.

    end

end

#####
##### DerivativeTest
#####

@testset verbose=true "DerivativeTest" begin

    x = 1

    @testset "scalar derivative of DiffTests.$f" for f in DiffTests.NUMBER_TO_NUMBER_FUNCS
        v = f(x)
        d = ForwardDiff.derivative(f, x)
        # @test isapprox(d, Calculus.derivative(f, x), atol=FINITEDIFF_ERROR)

        @test d ≈ fwd_derivative(f, x)  broken=(f==DiffTests.num2num_4)
        @test d ≈ rev_derivative(f, x)  broken=(f==DiffTests.num2num_4)
    end

    @testset "array derivative of DiffTests.$f" for f in DiffTests.NUMBER_TO_ARRAY_FUNCS
        v = f(x)
        d = ForwardDiff.derivative(f, x)
        # @test isapprox(d, Calculus.derivative(f, x), atol=FINITEDIFF_ERROR)

        @test d ≈ fwd_derivative(f, x)
        @test d ≈ rev_derivative(f, x)
    end

    # @testset "exponential function at base zero: $derivative" for derivative in DERIVATIVES
    #     @test (x -> derivative(y -> x^y, -0.5))(0.0) === -Inf
    #     @test (x -> derivative(y -> x^y,  0.0))(0.0) === -Inf
    #     @test (x -> derivative(y -> x^y,  0.5))(0.0) === 0.0
    #     @test (x -> derivative(y -> x^y,  1.5))(0.0) === 0.0
    # end

end

#####
##### GradientTest
#####

@testset verbose=true "GradientTest" begin

    @testset "hardcoded rosenbrock gradient" begin
        f = DiffTests.rosenbrock_1
        x = [0.1, 0.2, 0.3]
        v = f(x)
        g = [-9.4, 15.6, 52.0]

        @test g ≈ ForwardDiff.gradient(f, x)
        @test g ≈ fwd_gradient(f, x)
        @test g ≈ rev_gradient(f, x)
    end

    @testset "gradient of DiffTests.$f" for f in DiffTests.VECTOR_TO_NUMBER_FUNCS
        X, Y = rand(13), rand(7)
        FINITEDIFF_ERROR = 3e-5

        v = f(X)
        g = ForwardDiff.gradient(f, X)
        # @test isapprox(g, Calculus.gradient(f, X), atol=FINITEDIFF_ERROR)

        @test_skip g ≈ fwd_gradient(f, X)
        @test_skip g ≈ rev_gradient(f, X)
        # Many of these fail. They don't involve mutation:
        # https://github.com/JuliaDiff/DiffTests.jl/blob/master/src/DiffTests.jl#L64-L121
    end

    # @testset "exponential function at base zero: $gradient" for gradient in GRADIENTS 
    #     @test isequal(gradient(t -> t[1]^t[2], [0.0, -0.5]), [NaN, NaN])
    #     @test isequal(gradient(t -> t[1]^t[2], [0.0,  0.0]), [NaN, NaN])
    #     @test isequal(gradient(t -> t[1]^t[2], [0.0,  0.5]), [Inf, NaN])
    #     @test isequal(gradient(t -> t[1]^t[2], [0.0,  1.5]), [0.0, 0.0])
    # end

    @testset "chunk size zero - issue 399: $gradient" for gradient in GRADIENTS
        f_const(x) = 1.0
        g_grad_const = x -> gradient(f_const, x)
        @test g_grad_const([1.0]) |> iszero
        @test g_grad_const(zeros(Float64, 0)) |> (g -> isempty(g) || g isa AbstractZero)
    end

    # Issue 548
    @testset "ArithmeticStyle: $gradient" for gradient in GRADIENTS
        function f(p)
            sum(collect(0.0:p[1]:p[2]))
        end
        @test gradient(f, [0.2,25.0]) == [7875.0, 0.0]  broken = gradient==rev_gradient  # Rewrite reached intrinsic function fptosi. Missing rule?
    end

    # Issue 197
    @testset "det with branches" begin
        det2(A) = return (
            A[1,1]*(A[2,2]*A[3,3]-A[2,3]*A[3,2]) -
            A[1,2]*(A[2,1]*A[3,3]-A[2,3]*A[3,1]) +
            A[1,3]*(A[2,1]*A[3,2]-A[2,2]*A[3,1])
        )

        A = [1 0 0; 0 2 0; 0 pi 3]
        @test det2(A) == det(A) == 6
        @test istril(A)

        ∇A = [6 0 0; 0 3 -pi; 0 0 2]
        @test ForwardDiff.gradient(det2, A) ≈ ∇A
        @test_broken ForwardDiff.gradient(det, A) ≈ ∇A

        @test fwd_gradient(det2, A) ≈ ∇A
        @test fwd_gradient(det, A) ≈ ∇A

        @test rev_gradient(det2, A) ≈ ∇A
        @test rev_gradient(det, A) ≈ ∇A

        # And issue 407
        @test_broken ForwardDiff.hessian(det, A) ≈ ForwardDiff.hessian(det2, A)

        H = ForwardDiff.hessian(det2, A)

        @test_broken fwd_hessian(det, A) ≈ H
        @test_broken rev_hessian(det, A) ≈ H
        @test_broken fwd_rev_hessian(det, A) ≈ H
        @test_broken rev_fwd_hessian(det, A) ≈ H

        @test_broken fwd_hessian(det2, A) ≈ H  # UndefVarError: B not defined
        @test_broken rev_hessian(det2, A) ≈ H
        @test_broken fwd_rev_hessian(det2, A) ≈ H  # Internal error: encountered unexpected error in runtime: AssertionError(msg="argextype only works on argument-position values")
        @test_broken rev_fwd_hessian(det2, A) ≈ H  # MethodError: no method matching (::Diffractor.∂⃖recurse{1})(::typeof(Core.arrayset)
    end

    @testset "branches in mul!" begin
        a, b = rand(3,3), rand(3,3)

        # Issue 536, version with 3-arg *, Julia 1.7:
        @test_broken ForwardDiff.derivative(x -> sum(x*a*b), 0.0) ≈ sum(a * b)

        @test fwd_derivative(x -> sum(x*a*b), 0.0) ≈ sum(a * b)
        @test rev_derivative(x -> sum(x*a*b), 0.0) ≈ sum(a * b)
    end

end

#####
##### HessianTest
#####

@testset verbose=true "HessianTest" begin

    @testset "hardcoded rosenbrock hessian" begin

        f = DiffTests.rosenbrock_1
        x = [0.1, 0.2, 0.3]
        v = f(x)
        g = [-9.4, 15.6, 52.0]
        h = [-66.0  -40.0    0.0;
             -40.0  130.0  -80.0;
               0.0  -80.0  200.0]

        @test isapprox(h, ForwardDiff.hessian(f, x))

        @test_skip h ≈ fwd_hessian(f, x) 
        @test_broken h ≈ rev_hessian(f, x)  # Control flow support not fully implemented yet for higher-order reverse mode
        @test_skip h ≈ rev_fwd_hessian(f, x)
        @test_skip h ≈ fwd_rev_hessian(f, x)
    end

    @testset "hessians for DiffTests.$f" for f in DiffTests.VECTOR_TO_NUMBER_FUNCS
        X, Y = rand(13), rand(7)

        v = f(X)
        g = ForwardDiff.gradient(f, X)
        h = ForwardDiff.hessian(f, X)

        @test_broken g ≈ rev_gradient(f, x)
        @test_broken h ≈ rev_hessian(f, x)
    end

end

#####
##### JacobianTest
#####

@testset verbose=true "JacobianTest" begin

    @testset "hardcoded jacobian" begin

        f(x) = begin
            y1 = x[1] * x[2] * sin(x[3]^2)
            y2 = y1 + x[3]
            y3 = y1 / y2
            y4 = x[3]
            [y1, y2, y3, y4]
        end
        x = [1, 2, 3]
        v = f(x)
        j = [0.8242369704835132  0.4121184852417566  -10.933563142616123
             0.8242369704835132  0.4121184852417566  -9.933563142616123
             0.169076696546684   0.084538348273342   -2.299173530851733
             0.0                 0.0                 1.0]

        @test isapprox(j, ForwardDiff.jacobian(f, x))
        @test isapprox(j, fwd_jacobian(f, x))
        @test isapprox(j, rev_jacobian(f, x))

    end

    @testset "jacobians of DiffTests.$f" for f in DiffTests.ARRAY_TO_ARRAY_FUNCS
        X, Y = rand(13), rand(7)
        FINITEDIFF_ERROR = 3e-5

        v = f(X)
        j = ForwardDiff.jacobian(f, X)
        # @test isapprox(j, Calculus.jacobian(x -> vec(f(x)), X, :forward), atol=1.3FINITEDIFF_ERROR)

        @test j ≈ fwd_jacobian(f, X)  broken = f ∉ [-, identity, DiffTests.arr2arr_2]
        @test j ≈ rev_jacobian(f, X)  broken = f ∉ [-, identity, DiffTests.arr2arr_2]
        # Most of these involve mutation:
        # https://github.com/JuliaDiff/DiffTests.jl/blob/master/src/DiffTests.jl#L252-L272

    end

    # for f! in DiffTests.INPLACE_ARRAY_TO_ARRAY_FUNCS
    #     v = fill!(similar(Y), 0.0)
    #     f!(v, X)
    #     j = ForwardDiff.jacobian(f!, fill!(similar(Y), 0.0), X)
    #     @test isapprox(j, Calculus.jacobian(x -> (y = fill!(similar(Y), 0.0); f!(y, x); vec(y)), X, :forward), atol=FINITEDIFF_ERROR)
    # end

    # @testset "dimension errors for jacobian" begin
    #     @test_throws DimensionMismatch ForwardDiff.jacobian(identity, 2pi) # input
    #     @test_throws DimensionMismatch ForwardDiff.jacobian(sum, fill(2pi, 2)) # vector_mode_jacobian
    #     @test_throws DimensionMismatch ForwardDiff.jacobian(sum, fill(2pi, 10^6)) # chunk_mode_jacobian
    # end

    @testset "eigen" begin
        @test ForwardDiff.jacobian(x -> eigvals(SymTridiagonal(x, x[1:end-1])), [1.,2.]) ≈ [(1 - 3/sqrt(5))/2 (1 - 1/sqrt(5))/2 ; (1 + 3/sqrt(5))/2 (1 + 1/sqrt(5))/2]
        @test ForwardDiff.jacobian(x -> eigvals(Symmetric(x*x')), [1.,2.]) ≈ [0 0; 2 4]

        @test_broken fwd_jacobian(x -> eigvals(SymTridiagonal(x, x[1:end-1])), [1.,2.]) ≈ [(1 - 3/sqrt(5))/2 (1 - 1/sqrt(5))/2 ; (1 + 3/sqrt(5))/2 (1 + 1/sqrt(5))/2]
        @test_broken fwd_jacobian(x -> eigvals(Symmetric(x*x')), [1.,2.]) ≈ [0 0; 2 4]

        @test_broken rev_jacobian(x -> eigvals(SymTridiagonal(x, x[1:end-1])), [1.,2.]) ≈ [(1 - 3/sqrt(5))/2 (1 - 1/sqrt(5))/2 ; (1 + 3/sqrt(5))/2 (1 + 1/sqrt(5))/2]
        @test rev_jacobian(x -> eigvals(Symmetric(x*x')), [1.,2.]) ≈ [0 0; 2 4]
    end

end

#####
##### MiscTest
#####

@testset verbose=true "MiscTest" begin

    ##########################
    # Nested Differentiation #
    ##########################

    @testset "nested README example, $jacobian + $gradient" for jacobian in JACOBIANS, gradient in GRADIENTS

        # README example #
        #----------------#

        x = rand(5)

        f = x -> sum(sin, x) + prod(tan, x) * sum(sqrt, x)
        g = x -> gradient(f, x)
        j = x -> jacobian(g, x)

        @test isapprox(ForwardDiff.hessian(f, x), j(x))  broken = (jacobian, gradient) != (ForwardDiff.jacobian, ForwardDiff.gradient)
        # MethodError: no method matching +(::Tuple{NoTangent}, ::Tuple{NoTangent})
        # MethodError: no method matching rebundle(::Vector{Diffractor.CompositeBundle{1, Tuple{Float64, ChainRules.var"#sin_pullback#1175"{Float64}}, Tuple{Diffractor.TangentBundle{1, Float64, Tuple{Float64}}, Diffractor.CompositeBundle{1, ChainRules.var"#sin_pullback#1175"{Float64}, Tuple{Diffractor.TangentBundle{1, Float64, Tuple{Float64}}}}}}})
        # Internal error: encountered unexpected error in runtime: AssertionError(msg="argextype only works on argument-position values") argextype at ./compiler/optimize.jl:357
    end

    # higher-order derivatives #
    #--------------------------#

    @test_skip @testset "tensor 3rd order, $jacobian + $hessian" for jacobian in JACOBIANS, hessian in HESSIANS

        function tensor(f, x)
            n = length(x)
            out = jacobian(y -> hessian(f, y), x)
            return reshape(out, n, n, n)
        end

        test_tensor_output = reshape([240.0  -400.0     0.0;
                                     -400.0     0.0     0.0;
                                        0.0     0.0     0.0;
                                     -400.0     0.0     0.0;
                                        0.0   480.0  -400.0;
                                        0.0  -400.0     0.0;
                                        0.0     0.0     0.0;
                                        0.0  -400.0     0.0;
                                        0.0     0.0     0.0], 3, 3, 3)

        @test isapprox(tensor(DiffTests.rosenbrock_1, [0.1, 0.2, 0.3]), test_tensor_output)

    end

    @testset "broadcast 4rd order, $jacobian + $jacobian2" for jacobian in JACOBIANS, jacobian2 in JACOBIANS

        test_nested_jacobian_output = [-sin(1)  0.0     0.0;
                                       -0.0    -0.0    -0.0;
                                       -0.0    -0.0    -0.0;
                                        0.0     0.0     0.0;
                                       -0.0    -sin(2) -0.0;
                                       -0.0    -0.0    -0.0;
                                        0.0     0.0     0.0;
                                       -0.0    -0.0    -0.0;
                                       -0.0    -0.0    -sin(3)]

        sin_jacobian = x -> jacobian2(y -> broadcast(sin, y), x)

        @test isapprox(jacobian(sin_jacobian, [1., 2., 3.]), test_nested_jacobian_output)  broken = jacobian != ForwardDiff.jacobian

    end

    @testset "trig 2rd order, some gradient + $derivative" for derivative in DERIVATIVES
        # Issue #59 example #
        #-------------------#

        x = rand(2)

        f = x -> sin(x)/3 * cos(x)/2
        df = x -> derivative(f, x)
        testdf = x -> (((cos(x)^2)/3) - (sin(x)^2)/3) / 2

        @test df(x[1]) ≈ testdf(x[1])

        f2 = x -> df(x[1]) * f(x[2])
        testf2 = x -> testdf(x[1]) * f(x[2])

        @test isapprox(ForwardDiff.gradient(f2, x), ForwardDiff.gradient(testf2, x))
        g = ForwardDiff.gradient(testf2, x)

        @test g ≈ fwd_gradient(f2, x)  broken = derivative != fwd_derivative
        @test g ≈ rev_gradient(f2, x)  broken = derivative != rev_derivative

        # MethodError: no method matching *(::ForwardDiff.Dual{ForwardDiff.Tag{var"#139#140", Float64}, Float64, 1}, ::Tuple{Float64, Tuple{Tuple{Float64}}})
    end

    ######################################
    # Higher-Dimensional Differentiation #
    ######################################

    @testset "inv & kron, $jacobian" for jacobian in JACOBIANS

        x = rand(5, 5)

        @test isapprox(ForwardDiff.jacobian(inv, x), -kron(inv(x'), inv(x)))

    end

    #########################################
    # Differentiation with non-Array inputs #
    #########################################

    # x = rand(5,5)

    # # Sparse
    # f = x -> sum(sin, x) + prod(tan, x) * sum(sqrt, x)
    # gfx = ForwardDiff.gradient(f, x)
    # @test isapprox(gfx, ForwardDiff.gradient(f, sparse(x)))

    # # Views
    # jinvx = ForwardDiff.jacobian(inv, x)
    # @test isapprox(jinvx, ForwardDiff.jacobian(inv, view(x, 1:5, 1:5)))

    ########################
    # Conversion/Promotion #
    ########################


    @testset "issue 71: target function returns a literal" begin

        # target function returns a literal (Issue #71) #
        #-----------------------------------------------#

        # @test ForwardDiff.derivative(x->zero(x), rand()) == ForwardDiff.derivative(x->1.0, rand())
        # @test ForwardDiff.gradient(x->zero(x[1]), [rand()]) == ForwardDiff.gradient(x->1.0, [rand()])
        # @test ForwardDiff.hessian(x->zero(x[1]), [rand()]) == ForwardDiff.hessian(x->1.0, [rand()])
        # @test ForwardDiff.jacobian(x->[zero(x[1])], [rand()]) == ForwardDiff.jacobian(x->[1.0], [rand()])

        for derivative in DERIVATIVES
            @test derivative(x->zero(x), rand()) |> iszero
        end
        for gradient in GRADIENTS
            @test gradient(x->zero(x[1]), [rand()]) |> iszero
        end
        for hessian in HESSIANS
            @test hessian(x->zero(x[1]), [rand()]) |> iszero
        end
        for jacobian in JACOBIANS
            @test jacobian(x->[zero(x[1])], [rand()]) |> iszero
        end

    end

    @testset "arithmetic element-wise functions, $jacobian" for jacobian in JACOBIANS

        if jacobian == rev_jacobian
            @test_broken false
            # Got exception outside of a @test
            # DimensionMismatch("arrays could not be broadcast to a common size; got a dimension with lengths 2 and 4")
            continue
        end

        # arithmetic element-wise functions #
        #-----------------------------------#

        for op in (:.-, :.+, :./, :.*)
            @eval begin
                N = 4
                a = fill(1.0, N)
                jac0 = reshape(vcat([[fill(0.0, N*(i-1)); a; fill(0.0, N^2-N*i)] for i = 1:N]...), N^2, N)

                f = x -> [$op(x[1], a); $op(x[2], a); $op(x[3], a); $op(x[4], a)]
                jac = $jacobian(f, a)
                @test isapprox(jac0, jac)
            end
        end

    end

    # NaNs #
    #------#

    # @test ForwardDiff.partials(NaNMath.pow(ForwardDiff.Dual(-2.0,1.0),ForwardDiff.Dual(2.0,0.0)),1) == -4.0

    # Partials{0} #
    #-------------#

    @testset "Partials? $hessian" for hessian in HESSIANS

        if hessian == rev_hessian
            x, y = rand(3), rand(3)
            @test_broken  hessian(y -> sum(hypot.(x, y)), y)  isa AbstractMatrix
            # Control flow support not fully implemented yet for higher-order reverse mode (TODO)
            continue
        end

        x, y = rand(3), rand(3)
        h = hessian(y -> sum(hypot.(x, y)), y)

        @test h[1, 1] ≈ (x[1]^2) / (x[1]^2 + y[1]^2)^(3/2)
        @test h[2, 2] ≈ (x[2]^2) / (x[2]^2 + y[2]^2)^(3/2)
        @test h[3, 3] ≈ (x[3]^2) / (x[3]^2 + y[3]^2)^(3/2)
        let i, j
            for i in 1:3, j in 1:3
                i != j && @test h[i, j] ≈ 0.0
            end
        end

    end

    @testset "issue 267: $hessian" for hessian in HESSIANS

        @noinline f267(z, x) = x[1]
        z267 = ([(1, (2), [(3, (4, 5, [1, 2, (3, (4, 5), [5])]), (5))])])
        let z = z267,
            g = x -> f267(z, x),
            h = x -> g(x)
            @test hessian(h, [1.]) == fill(0.0, 1, 1)  broken = hessian == rev_hessian
        end

    end

    @testset "issue 290: rem2pi & rounding modes, $derivative" for derivative in DERIVATIVES

        @test derivative(x -> rem2pi(x, RoundUp), rand()) == 1
        @test derivative(x -> rem2pi(x, RoundDown), rand()) == 1

    end

end
