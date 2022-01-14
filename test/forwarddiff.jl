
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
            Diffractor.PrimeDerivativeBack(f)(float(x))
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

    @test_skip @testset "issue 83: perturbation confusion 1, $D" for D in DERIVATIVES

        @test D(x -> x * D(y -> x + y, 1), 1) == 1  broken = D != ForwardDiff.derivative

        g = ForwardDiff.gradient(v -> sum(v) * norm(v), [1])
        println("FD")
        @test ForwardDiff.gradient(v -> sum(v) * D(y -> y * norm(v), 1), [1]) == g
        println("fwd")
        @test_broken fwd_gradient(v -> sum(v) * D(y -> y * norm(v), 1), [1]) == g
         println("rev")
        @test rev_gradient(v -> sum(v) * D(y -> y * norm(v), 1), [1]) == g # broken = D != rev_derivative
# ?? a mess
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
        q̇ = [3,4]
        L(t,q,q̇) = m/2 * dot(q̇,q̇) - m*g*q[2]

        q = [1,2]
        p = [5,6]
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

        Lagrangian2Hamiltonian(L, t, q, p)
        @test_broken gradient(a->Lagrangian2Hamiltonian(L, t, a, p), q) == [0.0,g]
# ?? InexactError: Int64(-9.8)
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

    @testset "exponential function at base zero: $derivative" for derivative in DERIVATIVES
        @test (x -> derivative(y -> x^y, -0.5))(0.0) === -Inf
        @test (x -> derivative(y -> x^y,  0.0))(0.0) === -Inf
        @test (x -> derivative(y -> x^y,  0.5))(0.0) === 0.0
        @test (x -> derivative(y -> x^y,  1.5))(0.0) === 0.0
    end

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
        @test_broken g ≈ fwd_gradient(f, x)
        @test g ≈ rev_gradient(f, x)
    end

    @testset "gradient of DiffTests.$f" for f in DiffTests.VECTOR_TO_NUMBER_FUNCS
        X, Y = rand(13), rand(7)
        FINITEDIFF_ERROR = 3e-5

        v = f(X)
        g = ForwardDiff.gradient(f, X)
        # @test isapprox(g, Calculus.gradient(f, X), atol=FINITEDIFF_ERROR)

        @test g ≈ fwd_gradient(f, X)
        @test g ≈ rev_gradient(f, X)
    end

    @testset "exponential function at base zero: $gradient" for gradient in GRADIENTS 
        @test isequal(gradient(t -> t[1]^t[2], [0.0, -0.5]), [NaN, NaN])
        @test isequal(gradient(t -> t[1]^t[2], [0.0,  0.0]), [NaN, NaN])
        @test isequal(gradient(t -> t[1]^t[2], [0.0,  0.5]), [Inf, NaN])
        @test isequal(gradient(t -> t[1]^t[2], [0.0,  1.5]), [0.0, 0.0])
    end

    @testset "chunk size zero - issue 399: $gradient" for gradient in GRADIENTS
        f_const(x) = 1.0
        g_grad_const = x -> gradient(f_const, x)
        @test g_grad_const([1.0]) == [0.0]
        @test isempty(g_grad_const(zeros(Float64, 0)))
    end

    @testset "ArithmeticStyle: $gradient" for gradient in GRADIENTS
        function f(p)
            sum(collect(0.0:p[1]:p[2]))
        end
        @test gradient(f, [0.2,25.0]) == [7875.0, 0.0]
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

# ??TODO!

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

        @test isapprox(ForwardDiff.hessian(f, x), j(x))
    end

    # higher-order derivatives #
    #--------------------------#

    @testset "tensor 3rd order, $jacobian + $hessian" for jacobian in JACOBIANS, hessian in HESSIANS

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

        @test isapprox(jacobian(sin_jacobian, [1., 2., 3.]), test_nested_jacobian_output)

    end

    @testset "trig 2rd order, $gradient + $derivative" for gradient in GRADIENTS, derivative in DERIVATIVES
        # Issue #59 example #
        #-------------------#

        x = rand(2)

        f = x -> sin(x)/3 * cos(x)/2
        df = x -> derivative(f, x)
        testdf = x -> (((cos(x)^2)/3) - (sin(x)^2)/3) / 2

        @test df(x) ≈ testdf(x)

        f2 = x -> df(x[1]) * f(x[2])
        testf2 = x -> testdf(x[1]) * f(x[2])

        @test isapprox(gradient(f2, x), ForwardDiff.gradient(testf2, x))

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

        @test ForwardDiff.derivative(x->zero(x), rand()) == ForwardDiff.derivative(x->1.0, rand())
        @test ForwardDiff.gradient(x->zero(x[1]), [rand()]) == ForwardDiff.gradient(x->1.0, [rand()])
        @test ForwardDiff.hessian(x->zero(x[1]), [rand()]) == ForwardDiff.hessian(x->1.0, [rand()])
        @test ForwardDiff.jacobian(x->[zero(x[1])], [rand()]) == ForwardDiff.jacobian(x->[1.0], [rand()])

    end


    @testset "arithmetic element-wise functions, $jacobian" for jacobian in JACOBIANS

        # arithmetic element-wise functions #
        #-----------------------------------#

        N = 4
        a = fill(1.0, N)
        jac0 = reshape(vcat([[fill(0.0, N*(i-1)); a; fill(0.0, N^2-N*i)] for i = 1:N]...), N^2, N)

        for op in (:.-, :.+, :./, :.*)
            @eval begin
                f = x -> [$op(x[1], a); $op(x[2], a); $op(x[3], a); $op(x[4], a)]
                jac = jacobian(f, a)
                @test isapprox(jac0, jac)
            end
        end

    end

    # NaNs #
    #------#

    @test ForwardDiff.partials(NaNMath.pow(ForwardDiff.Dual(-2.0,1.0),ForwardDiff.Dual(2.0,0.0)),1) == -4.0

    # Partials{0} #
    #-------------#

    x, y = rand(3), rand(3)
    h = ForwardDiff.hessian(y -> sum(hypot.(x, y)), y)

    @test h[1, 1] ≈ (x[1]^2) / (x[1]^2 + y[1]^2)^(3/2)
    @test h[2, 2] ≈ (x[2]^2) / (x[2]^2 + y[2]^2)^(3/2)
    @test h[3, 3] ≈ (x[3]^2) / (x[3]^2 + y[3]^2)^(3/2)
    let i, j
        for i in 1:3, j in 1:3
            i != j && @test h[i, j] ≈ 0.0
        end
    end

    @testset "issue 267: hessian" begin

        @noinline f267(z, x) = x[1]
        z267 = ([(1, (2), [(3, (4, 5, [1, 2, (3, (4, 5), [5])]), (5))])])
        let z = z267,
            g = x -> f267(z, x),
            h = x -> g(x)
            @test ForwardDiff.hessian(h, [1.]) == fill(0.0, 1, 1)
        end

    end

    @testset "issue 290: rem2pi & rounding modes, $derivative" for derivative in DERIVATIVES

        @test derivative(x -> rem2pi(x, RoundUp), rand()) == 1
        @test derivative(x -> rem2pi(x, RoundDown), rand()) == 1

    end

end
