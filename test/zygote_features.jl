
# This file contains many examples borrowed from Zygote's tests, 
# all files except "gradcheck.jl" which is elsewhere.

# Ideally this will catch the awkward cases, and should be extended
# to test forward mode and higher derivatives.

using Diffractor, ChainRulesCore
using Test, LinearAlgebra

isZero(x) = x isa AbstractZero

# Zygote's misnamed hobbit function:
function pullback(f, x...)
    y, b = Diffractor.∂⃖{1}()(f, x...)
    back(dy) = map(unthunk, Base.tail(b(dy)))
    y, back
end

#####
##### Zygote/test/complex.jl
#####

gradfwd(f, x::Number) = (Diffractor.PrimeDerivativeFwd(f)(x),)
gradback(f, x::Number) = (Diffractor.PrimeDerivativeBack(f)(x),)

@testset "complex numbers" begin # : $gradient" for gradient in (gradfwd, gradback)

    @test gradient(x -> real(abs(x)*exp(im*angle(x))), 10+20im)[1] ≈ 1
    @test gradient(x -> imag(real(x)+0.3im), 0.3)[1] ≈ 0
    @test gradient(x -> imag(conj(x)+0.3im), 0.3 + 0im)[1] ≈ -1im
    @test gradient(x -> abs((imag(x)+0.3)), 0.3 + 0im)[1] ≈ 1im

    @test gradient(x -> norm((im*x) / (im)), 2)[1] == 1
    @test gradient(x -> norm((im) / (im*x)), 2)[1] == -1/4

    fs_C_to_R = (
        real,
        imag,
        abs,
        abs2,
        z -> abs(z)*cos(im*angle(z)),
        z->abs(cos(exp(z))),
        z->3*real(z)^3-2*imag(z)^5
    )
    @testset "C->R: $i" for (i,f) in enumerate(fs_C_to_R)
        for z in (1.0+2.0im, -2.0+pi*im)
            ε = 1e-8
            grad_fd = (f(z+ε)-f(z))/ε + im*(f(z+ε*im)-f(z))/ε
            @test abs(gradient(x -> real(f(x)), z)[1] - grad_fd) < sqrt(ε)
        end
    end

    fs_C_to_C_holomorphic = (
        cos,
        exp,
        log,
        z->z^2,
        z->(real(z)+im*imag(z))^2,
        z->real(z)^2 - imag(z)^2 +2im*(real(z)*imag(z)),
        z->exp(cos(log(z))),
        z->abs(z)*exp(im*angle(z)),
    )
    @testset "C->C holomorphic: $i" for (i,f) in enumerate(fs_C_to_C_holomorphic)
        for z in (1.0+2.0im, -2.0+pi*im)
            ε = 1e-8
            grad_fd_r = (f(z+ε)-f(z))/ε
            grad_fd_i = (f(z+ε*im)-f(z))/(ε*im)
            @assert abs(grad_fd_r - grad_fd_i) < sqrt(ε) # check the function is indeed holomorphic
            @test abs(gradient(x -> real(f(x)), z)[1] - conj(grad_fd_r)) < sqrt(ε)
        end
    end

    fs_C_to_C_non_holomorphic = (
        conj,
        z->abs(z)+0im,
        z->im*abs(z),
        z->abs2(z)+0im,
        z->im*abs2(z),
        z->z'z,
        z->conj(z)*z^2,
    )
    @testset "C->C non-holomorphic: $i" for (i,f) in enumerate((fs_C_to_C_holomorphic...,fs_C_to_C_holomorphic...))
        for z in (1.0+2.0im, -2.0+pi*im)
            ε = 1e-8
            grad_fd = real(f(z+ε)-f(z))/ε + im*real(f(z+ε*im)-f(z))/ε
            @test abs(gradient(x -> real(f(x)), z)[1] - grad_fd) < sqrt(ε)
        end
    end

    # Zygote issue 342
    @test gradient(x->real(x + 2.0*im), 3.0) == (1.0,)
    @test gradient(x->imag(x + 2.0*im), 3.0) == (0.0,)

end

# Zygote issue 705
@test gradient(x -> imag(sum(exp, x)), [1,2,3])[1] |> isZero
@test gradient(x -> imag(sum(exp, x)), [1+0im,2,3])[1] ≈ im .* exp.(1:3)


#####
##### Zygote/test/features.jl
#####

@testset "features I" begin

    # power functions

    function pow(x, n)
        r = 1
        while n > 0
            n -= 1
            r *= x
        end
        return r
    end
    @test gradient(pow, 2, 3)[1] == 12
    @test gradient(pow, 2, 3)[2] |> isZero

    function pow_mut(x, n)
        r = Ref(one(x))
        while n > 0
            n -= 1
            r[] = r[] * x  # not sure Diffractor supports this, if not it could give a helpful error
        end
        return r[]
    end
    @test_broken gradient(pow_mut, 2, 3)[1] == 12  # no method matching (::Diffractor.∂⃖recurse{1})(::typeof(setfield!), ::Base.RefValue{Int64}, ::Symbol, ::Int64)
    @test_broken gradient(pow_mut, 2, 3)[2] |> isZero

    global r163 = 1
    function pow_global(x, n)
        global r163
        while n > 0
            r163 *= x
            n -= 1
        end
        return r163
    end
    @test_broken gradient(pow_global, 2, 3)[1] == 12  # transform!(ci::Core.CodeInfo, meth::Method, nargs::Int64, sparams::Core.SimpleVector, N::Int64)
    @test_broken gradient(pow_global, 2, 3)[2] |> isZero

    # misc.

    @test gradient(x -> 1, 2)[1] |> isZero

    @test gradient(t -> t[1]*t[2], (2, 3)) == ((3, 2),)
    @test_broken gradient(t -> t[1]*t[2], (2, 3)) isa Tangent  # should be!

    # complex & getproperty -- https://github.com/JuliaDiff/Diffractor.jl/issues/71

    @test_broken gradient(x -> x.re, 2+3im) === (1.0 + 0.0im,)  # one NamedTuple
    @test_broken gradient(x -> x.re*x.im, 2+3im) == (3.0 + 2.0im,)  # two, different fields
    @test_broken gradient(x -> x.re*x.im + x.re, 2+3im) == (4.0 + 2.0im,)  # three, with accumulation

    @test_skip gradient(x -> abs2(x * x.re), 4+5im) == (456.0 + 160.0im,)   # gradient participates
    @test gradient(x -> abs2(x * real(x)), 4+5im) == (456.0 + 160.0im,)   # function not getproperty
    @test_skip gradient(x -> abs2(x * getfield(x, :re)), 4+5im) == (456.0 + 160.0im,)

end
@testset "features II" begin

    # structs

    struct Bar{T}
        a::T
        b::T
    end
    function mul_struct(a, b)
        c = Bar(a, b)
        c.a * c.b
    end
    @test gradient(mul_struct, 2, 3) == (3, 2)

    @test_broken gradient(x -> [x][1].a, Bar(1, 1)) == ((a=1, b=NoTangent()),)  # MethodError: no method matching zero(::Type{Bar{Int64}})

    function mul_tuple(a, b)
        c = (a, b)
        c[1] * c[2]
    end
    @test gradient(mul_tuple, 2, 3) == (3, 2)

    function mul_lambda(x, y)
        g = z -> x * z
        g(y)
    end
    @test gradient(mul_lambda, 2, 3) == (3, 2)

    # splats

    @test gradient((a, b...) -> *(a, b...), 2, 3) == (3, 2)

    @test_broken gradient((x, a...) -> x, 1) == (1,)
    @test gradient((x, a...) -> x, 1, 1) == (1, ZeroTangent())
    @test_broken gradient((x, a...) -> x == a, 1) == (NoTangent(),)
    @test gradient((x, a...) -> x == a, 1, 2) == (NoTangent(), NoTangent())

    # keywords

    kwmul(; a = 1, b) = a*b
    mul_kw(a, b) = kwmul(a = a, b = b)
    @test_broken gradient(mul_kw, 2, 3) == (3, 2)  # passes at REPL, not in testset?

end
@testset "features III" begin

    function myprod(xs)
        s = 1
        for x in xs
            s *= x
        end
        return s
    end
    @test gradient(myprod, [1,2,3])[1] == [6,3,2]

    function mul_vec(a, b)
        xs = [a, b]
        xs[1] * xs[2]
    end
    @test gradient(mul_vec, 2, 3) == (3, 2)

    # dictionary

    @test_skip gradient(2) do x
        d = Dict()
        d[:x] = x
        x * d[:x]
    end == (4,)

    # keywords

    f249(args...; a=nothing, kwargs...) = g250(a,args...; kwargs...)
    g250(args...; x=1, idx=Colon(), kwargs...) = x[idx]
    @test_skip gradient(x -> sum(f249(; x=x, idx=1:1)), ones(2))[1] == [1, 0]

    # recursion

    pow_rec(x, n) = n == 0 ? 1 : x*pow_rec(x, n-1)
    @test gradient(pow_rec, 2, 3)[1] == 12
    @test gradient(pow_rec, 2, 3)[2] |> isZero

    # second derivatives

    function grad258(f, args...)
        y, back = pullback(f, args...)
        return back(1)
    end
    D263(f, x) = grad258(f, x)[1]

    @test D263(x -> D263(sin, x), 0.5) == -sin(0.5)
    @test D263(x -> x*D263(y -> x+y, 1), 1) == 1
    @test D263(x -> x*D263(y -> x*y, 1), 4) == 8

    # throw

    f272(x) = throw(DimensionMismatch("fubar"))
    @test_throws DimensionMismatch gradient(f272, 1)

    # hvcat

    @test gradient(2) do x
        H = [1 x; 3 4]
        sum(H)
    end[1] == 1

    @test gradient(x -> one(eltype(x)), rand(10))[1] |> isZero

    # three-way control flow merge

    @test gradient(1) do x
        if x > 0
            x *= 2
        elseif x < 0
            x *= 3
        end
        x
    end[1] == 2

end
@testset "features IV" begin

    @test gradient(1) do x
        if true
        elseif true
            nothing
        end
        x + x
    end == (2,)

    # try

    function pow_try(x)
        try
            2x
        catch e
            println("error")
        end
    end

    @test_broken gradient(pow_try, 1) == (2,)

    # @simd

    function pow_simd(x, n)
        r = 1
            @simd for i = 1:n
        r *= x
        end
        return r
    end
    @test gradient(pow_simd, 2, 3)[1] == 12
    @test gradient(pow_simd, 2, 3)[2] |> isZero

    # @timed

    @test_broken gradient(x -> first(@timed x), 0) == (1,)  # transform!(ci::Core.CodeInfo, meth::Method, nargs::Int64, sparams::Core.SimpleVector, N::Int64)
    @test_broken gradient(x -> (@time x^2), 3) == (6,)

    # kwarg splat

    g516(; kwargs...) = kwargs[:x] * kwargs[:z]
    h517(somedata) = g516(; somedata...)
    @test_broken gradient(h517, (; x=3.0, y=4.0, z=2.3)) == (Tangent{NamedTuple{(:x, :y, :z), Tuple{Float64,Float64,Float64}}}(; x=2.3, y=0.0, z=3.0),)
    @test_broken gradient(h517, Dict(:x=>3.0, :y=>4.0, :z=>2.3)) == (Tangent{NamedTuple{(:x, :y, :z), Tuple{Float64,Float64,Float64}}}(; x=2.3, y=0.0, z=3.0),)

end

@testset "mutable structs" begin
    
    mutable struct MyMutable
        value::Float64
    end
    function foo!(m::MyMutable, x)
        m.value = x
    end
    function baz(args)
        m = MyMutable(0.0)
        foo!(m, args...)
        m.value
    end
    @test_broken gradient(baz, (1.0,)) == ((1.0,),)  # not sure that's the desired type, but for now an error

    # ChainRules represents these as the same Tangent as immutable structs, but is that ideal?

    @test gradient(x -> x.value^2 + x.value, MyMutable(3)) === (Tangent{MyMutable}(value = 7.0,),)

    @test gradient(x -> x.x^2 + x.x, Ref(3)) == (Tangent{Base.RefValue{Int}}(x = 7.0,),)
    @test gradient(x -> real(x.x^2 + im * x.x), Ref(4)) == (Tangent{Base.RefValue{Int}}(x = 8.0,),)

    # Field access of contents:
    @test_broken gradient(x -> abs2(x.x) + 7 * x.x.re, Ref(1+im)) == ((x = 9.0 + 2.0im,),)
    @test_broken gradient(x -> abs2(x[1].x) + 7 * x[1].x.re, [Ref(1+im)]) == ([(x = 9.0 + 2.0im,)],)
    @test_broken gradient(x -> abs2(x[1].x) + 7 * real(x[1].x), [Ref(1+im)]) == ([(x = 9.0 + 2.0im,)],)  # worked on 0.6.0, 0.6.20
    @test_broken gradient(x -> abs2(x[].x) + 7 * real(x[].x), Ref(Ref(1+im))) == ((x = 9.0 + 2.0im,),)  # gives nothing, same in 0.6.0

    # Array of mutables:
    @test_broken gradient(x -> sum(getindex.(x).^2), Ref.(1:3))[1] == [(;x=2i) for i in 1:3]
    @test_broken gradient(x -> sum(abs2∘getindex, x), Ref.(1:3))[1] == [(;x=2i) for i in 1:3]

    @test_broken gradient(x -> (getindex.(x).^2)[1], Ref.(1:3))[1][1] == (x=2.0,)  # rest are (x = 0.0,), but nothing would be OK too
    @test_broken gradient(x -> (prod.(getindex.(x)))[1], Ref.(eachcol([1 2; 3 4])))[1][1] == (x = [3.0, 1.0],)

    # Broadcasting over Ref:
    @test_broken gradient(x -> sum(sum, x .* [1,2,3]), Ref([4,5])) == ((x = [6.0, 6.0],),)
    @test_broken gradient(x -> sum(sum, Ref(x) .* [1,2,3]), [4,5]) == ([6.0, 6.0],)

end

@testset "NamedTuples" begin

    @test gradient(x -> x.a, (a=1, b=2)) == (Tangent{NamedTuple{(:a, :b), Tuple{Int,Int}}}(a = 1,),)
    @test_broken gradient(x -> x[1].a, [(a=1, b=2)]) == ([(a = 1, b = nothing)],) # MethodError: no method matching zero(::Type{NamedTuple{(:a, :b), Tuple{Int64, Int64}}})

    @test_broken gradient(x -> x[1].a, [(a=1, b=2), (a=3, b=4)]) == ([(a = 1, b = nothing), nothing],)

    # Mix with Ref
    @test gradient(x -> x[].a, Ref((a=1, b=2))) == (Tangent{Base.RefValue{NamedTuple{(:a, :b), Tuple{Int64, Int64}}}}(x = Tangent{NamedTuple{(:a, :b), Tuple{Int64, Int64}}}(a = 1,),),)
    @test_broken gradient(x -> x[1][].a, [Ref((a=1, b=2)), Ref((a=3, b=4))]) == ([(x = (a = 1, b = nothing),), nothing],)
    @test_broken gradient(x -> x[1].a, [(a=1, b=2), "three"]) == ([(a = 1, b = nothing), nothing],)

end

@testset "Pairs" begin

    @test_broken gradient(x->10*pairs((a=x, b=2))[1], 100) === 10.0  # ArgumentError: Tangent for the primal Base.Pairs{Symbol, Int64, Tuple{Symbol, Symbol}, NamedTuple{(:a, :b), Tuple{Int64, Int64}}} should be backed by a AbstractDict type, not by NamedTuple{(:data,), Tuple{Tuple{Float64, ZeroTangent}}}.
    @test_broken gradient(x->10*pairs((a=x, b=2))[2], 100) === 0

    foo387(; kw...) = 1
    @test_skip gradient(() -> foo387(a=1,b=2.0)) === ()  # Here Diffractor returns a function, by design?

    @test gradient(x->10*(x => 2)[1], 100) === (10.0,)
    @test gradient(x->10*(x => 2)[2], 100) === (ZeroTangent(),)

    @test gradient(x-> (:x => x)[2], 17) == (1,)

    d = Dict(:x=>1.0, :y=>3.0);
    @test_broken gradient(d -> Dict(:x => d[:x])[:x], d) == (Dict(:x => 1),)  # BoundsError: attempt to access 3-element Vector{Core.Compiler.BasicBlock} at index [4]

end

@testset "Iterators" begin

    # enumerate

    @test_broken gradient(1:5) do xs
        sum([x^i for (i,x) in enumerate(xs)])
    end == ([1, 4, 27, 256, 3125],)

    @test_broken gradient([1,10,100]) do xs
        sum([xs[i]^i for (i,x) in enumerate(xs)])
    end == ([1, 2 * 10^1, 3 * 100^2],)

    @test_broken gradient([1,10,100]) do xs
        sum((xs[i]^i for (i,x) in enumerate(xs))) # same without collect
    end == ([1, 2 * 10^1, 3 * 100^2],)

    # zip
    # On Julia 1.4 and earlier, [x/y for (x,y) in zip(10:14, 1:10)] is a DimensionMismatch,
    # while on 1.5 - 1.7 it stops early. 

    @test_broken gradient(10:14, 1:10) do xs, ys
        sum([x/y for (x,y) in zip(xs, ys)])
    end[2] ≈ vcat(.-(10:14) ./ (1:5).^2, zeros(5))

    @test_broken gradient(10:14, 1:10) do xs, ys
        sum(x/y for (x,y) in zip(xs, ys))   # same without collect
    end[2] ≈ vcat(.-(10:14) ./ (1:5).^2, zeros(5))

    @test_skip begin
        bk_z = pullback((xs,ys) -> sum([abs2(x*y) for (x,y) in zip(xs,ys)]), [1,2], [3im,4im])[2]
        @test bk_z(1.0)[1] isa AbstractVector{<:Real}  # projection
    end

    # Iterators.Filter

    @test_broken gradient(2:9) do xs
        sum([x^2 for x in xs if iseven(x)])
    end == ([4, 0, 8, 0, 12, 0, 16, 0],)

    @test_broken gradient(2:9) do xs
        sum(x^2 for x in xs if iseven(x)) # same without collect
    end == ([4, 0, 8, 0, 12, 0, 16, 0],)

    # Iterators.Product

    @test_broken gradient(1:10, 3:7) do xs, ys
        sum([x^2+y for x in xs, y in ys])
    end == (10:10:100, fill(10, 5))

    @test_broken gradient(1:10, 3:7) do xs, ys
        sum(x^2+y for x in xs, y in ys)  # same without collect
    end == (10:10:100, fill(10, 5))

    # Repeat that test without sum(iterator)
    function prod_acc(xs, ys)
        out = 0
        for xy in Iterators.product(xs, ys)
            out += xy[1]^2 + xy[2]
        end
        out
    end
    @test prod_acc(1:10, 3:7) == sum(x^2+y for x in 1:10, y in 3:7)
    @test_broken gradient(prod_acc, 1:10, 3:7) == (10:10:100, fill(10, 5))

    @test_broken gradient(rand(2,3)) do A
        sum([A[i,j] for i in 1:1, j in 1:2])
    end == ([1 1 0; 0 0 0],)

    @test_broken gradient(ones(3,5), 1:7) do xs, ys
        sum([x+y for x in xs, y in ys])
    end == (fill(7, 3,5), fill(15, 7))

    @test_skip begin
        bk_p = pullback((xs,ys) -> sum([x/y for x in xs, y in ys]), Diagonal([3,4,5]), [6,7]')[2]
        @test bk_p(1.0)[1] isa Diagonal  # projection
        @test bk_p(1.0)[2] isa Adjoint
    end

    # Iterators.Product with enumerate

    @test_broken gradient([2 3; 4 5]) do xs
        sum([x^i+y for (i,x) in enumerate(xs), y in xs]) 
    end == ([8 112; 36 2004],)

end

@testset "keyword passing" begin
    # https://github.com/JuliaDiff/ChainRules.jl/issues/257

    struct Type1{VJP}
        x::VJP
    end

    struct Type2{compile}
        Type2(compile=false) = new{compile}()
    end

    function loss_adjoint(θ)
        sum(f623(sensealg=Type1(Type2(true))))
    end

    i = 1

    global x620 = Any[nothing, nothing]

    g622(x, i, sensealg) = Main.x620[i] = sensealg
    ChainRulesCore.@non_differentiable g622(x, i, sensealg)

    function f623(; sensealg=nothing)
        g622(x620, i, sensealg)
        return rand(100)
    end

    loss_adjoint([1.0])
    i = 2

    @test_skip gradient(loss_adjoint, [1.0])

    @test_broken x620[1] == x620[2]

end

@testset "splats" begin

    @test gradient(x -> max(x...), [1,2,3])[1] == [0,0,1]
    @test gradient(x -> min(x...), (1,2,3))[1] === (1.0, 0.0, 0.0)

    @test_broken gradient(x -> max(x...), [1 2; 3 4])[1] == [0 0; 0 1]
    @test_broken gradient(x -> max(x...), [1,2,3]')[1] == [0 0 1]
    @test_broken gradient(x -> max(x...), [1,2,3]')[1] isa Adjoint

    # https://github.com/FluxML/Zygote.jl/issues/599
    @test gradient(w -> sum([w...]), [1,1])[1] isa AbstractVector

    # https://github.com/FluxML/Zygote.jl/issues/866
    f866(x) = reshape(x, fill(2, 2)...)
    @test gradient(x->sum(f866(x)), rand(4))[1] == [1,1,1,1]

    # https://github.com/FluxML/Zygote.jl/issues/731
    f731(x) = sum([x' * x, x...])
    @test gradient(f731, ones(3)) == ([3,3,3],)

end

@testset "accumulation" begin

    # from https://github.com/FluxML/Zygote.jl/issues/905
    function net905(x1)
        x2  = x1
        x3  = x1 + x2
        x4  = x1 + x2 + x3
        x5  = x1 + x2 + x3 + x4
        x6  = x1 + x2 + x3 + x4 + x5
        x7  = x1 + x2 + x3 + x4 + x5 + x6
        x8  = x1 + x2 + x3 + x4 + x5 + x6 + x7
        x9  = x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8
        x10 = x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9
    end
    loss(x) = sum(abs2, net905(x))

    @test gradient(loss, ones(10,10))[1] == fill(131072, 10, 10)
    @test_broken 150_000_000 > @allocated gradient(loss, ones(1000,1000))

end

@testset "tricky broadcasting" begin

    @test gradient(x -> sum(x .+ ones(2,2)), (1,2)) == (Tangent{Tuple{Int, Int}}(2,2),)
    @test gradient(x -> sum(x .+ ones(2,2)), (1,)) == (Tangent{Tuple{Int}}(4),)
    @test gradient(x -> sum(x .+ ones(2,1)), (1,2)) == (Tangent{Tuple{Int, Int}}(1,1),)

    # https://github.com/FluxML/Zygote.jl/issues/975
    gt = gradient((x,p) -> prod(x .^ p), [3,4], (1,2))
    gv = gradient((x,p) -> prod(x .^ p), [3,4], [1,2])
    @test gt[1] == gv[1]
    @test collect(gt[2]) ≈ gv[2]

    # closure captures y -- can't use ForwardDiff
    @test gradient((x,y) -> sum((z->z^2+y[1]).(x)), [1,2,3], [4,5]) == ([2, 4, 6], [3, 0])
    @test_broken gradient((x,y) -> sum((z->z^2+y[1]), x), [1,2,3], [4,5]) == ([2, 4, 6], [3, 0])  # lastindex(::Diffractor.OpticBundle{Int64})
    @test_broken gradient((x,y) -> sum(map((z->z^2+y[1]), x)), [1,2,3], [4,5]) == ([2, 4, 6], [3, 0]) # AssertionError: Base.issingletontype(typeof(f))
    @test_broken gradient((x,y) -> mapreduce((z->z^2+y[1]), +, x), [1,2,3], [4,5]) == ([2, 4, 6], [3, 0])

    # type unstable
    @test gradient(xs -> sum((x -> x<2 ? false : x^2).(xs)), [1,2,3])[1][2:3] == [4, 6]
    @test_broken gradient(xs -> sum((x -> x<2 ? false : x^2), xs), [1,2,3])[1][2:3] == [4, 6]
    @test_broken gradient(xs -> sum(map((x -> x<2 ? false : x^2), xs)), [1,2,3])[1][2:3] == [4, 6]
    @test_broken gradient(xs -> mapreduce((x -> x<2 ? false : x^2), +, xs), [1,2,3])[1][2:3] == [4, 6]

    # with Ref, Val, Symbol
    @test gradient(x -> sum(x .+ Ref(x[1])), [1,2,3]) == ([4,1,1],)
    @test gradient(x -> sum(x .+ (x[1],)), [1,2,3]) == ([4,1,1],)
    @test_broken gradient(x -> sum((first∘tuple).(x, :ignore)), [1,2,3]) == ([1,1,1],)
    @test_broken gradient(x -> sum((first∘tuple).(x, Symbol)), [1,2,3]) == ([1,1,1],)
    _f(x,::Val{y}=Val(2)) where {y} = x/y
    @test gradient(x -> sum(_f.(x, Val(2))), [1,2,3]) == ([0.5, 0.5, 0.5],)
    @test gradient(x -> sum(_f.(x)), [1,2,3]) == ([0.5, 0.5, 0.5],)
    @test_broken gradient(x -> sum(map(_f, x)), [1,2,3]) == ([0.5, 0.5, 0.5],) # InexactError: Int64(0.5)

    # with Bool
    @test_broken gradient(x -> sum(1 .- (x .> 0)), randn(5)) == (nothing,)  # MethodError: no method matching length(::NoTangent)

    @test_broken gradient(x -> sum((y->1-y).(x .> 0)), randn(5)) == (nothing,)
    @test_broken gradient(x -> sum(x .- (x .> 0)), randn(5)) == ([1,1,1,1,1],)

    @test_broken gradient(x -> sum(x ./ [1,2,4]), [1,2,pi]) == ([1.0, 0.5, 0.25],) # DimensionMismatch("arrays could not be broadcast to a common size; got a dimension with lengths 2 and 3")
    @test_broken gradient(x -> sum(map(/, x, [1,2,4])), [1,2,pi]) == ([1.0, 0.5, 0.25],)  # MethodError: no method matching (::Diffractor.∂⃖recurse{1})(::typeof(Core.arrayset), ::Bool, ::Vector{Float64}, ::Float64, ::Int64)

    # negative powers
    @test gradient((x,p) -> sum(x .^ p), [1.0,2.0,4.0], [1,-1,2])[1] ≈ [1.0, -0.25, 8.0]
    @test gradient((x,p) -> sum(x .^ p), [1.0,2.0,4.0], -1)[1] ≈ [-1.0, -0.25, -0.0625]
    @test_broken gradient((x,p) -> sum(z -> z^p, x), [1.0,2.0,4.0], -1)[1] ≈ [-1.0, -0.25, -0.0625]  # MethodError: no method matching lastindex(::Diffractor.OpticBundle{Float64})
    @test_broken gradient((x,p) -> mapreduce(z -> z^p, +, x), [1.0,2.0,4.0], -1)[1] ≈ [-1.0, -0.25, -0.0625] # MethodError: no method matching +(::Tuple{NoTangent}, ::Tuple{NoTangent})

    # second order
    @test_broken gradient(x -> sum(gradient(y -> sum(y.^2), x)[1]), [1, 2])[1] ≈ [2, 2]  # Control flow support not fully implemented yet for higher-order reverse mode (TODO)
    @test_broken gradient(x -> sum(gradient(y -> sum(sin.(y)), x)[1]), [1, 2])[1] ≈ [-0.8414709848078965, -0.9092974268256817]  # MethodError: no method matching Diffractor.Jet(::Int64, ::Float64, ::Tuple{Float64, Float64})
    @test_broken gradient(x -> sum(abs, gradient(y -> sum(log.(2 .* exp.(y)) .^ 2), x)[1]), [1, 2])[1] ≈ [2,2]

    # getproperty, Tangents, etc
    @test gradient(xs -> sum((x->x.im^2).(xs)), [1+2im,3])[1] == [4im, 0]
    @test_broken gradient(xs -> sum((x->x.im^2), xs), [1+2im,3])[1] == [4im, 0]  # MethodError: no method matching lastindex(::Diffractor.OpticBundle{Int64})
    @test_broken gradient(xs -> sum(map(x->x.im^2, xs)), [1+2im,3])[1] == [4im, 0]  # Tried to take the gradient of a complex-valued function
    @test_broken gradient(xs -> mapreduce(x->x.im^2, +, xs), [1+2im,3])[1] == [4im, 0]  # MethodError: Cannot `convert` an object of type Tangent{Complex{Int64}, NamedTuple{(:im,), Tuple{Float64}}} to an object of type Complex{Int64}

end

#####
##### Zygote/test/structures.jl
#####

@testset "async" begin

    function tasks1(x)
      ch = Channel(Inf)
      put!(ch, x^2) + take!(ch)
    end

    @test_broken gradient(tasks1, 5) == (20,)

    function tasks2(x)
      ch = Channel(0)
      t = @async put!(ch, x^2)
      y = take!(ch)
      wait(t)
      return y
    end

    @test_broken gradient(tasks2, 5) == (10,)

    function tasks3(x)
      ch = Channel(0)
      @sync begin
        @async put!(ch, x^2)
        take!(ch)
      end
    end

    @test_broken gradient(tasks3, 5) == (10,)

    tasks4(x) = fetch(@async x^2)
    @test_broken gradient(tasks4, 5) == (10,)

    tasks5(x) = fetch(schedule(Task(() -> x^2)))
    @test_broken gradient(tasks5, 5) == (10,)

end

@testset "issues" begin

    @test pullback(Array, [1f0])[1] == [1f0]

    # issue 594
    
    struct A594 x::Float64 end
  
    f594(a,v) = a.x + v
    g594(A,V) = sum(f594.(A,V))
    X = A594.(randn(2))
    Y = randn(2,2)
    @test_skip begin
        ∇ = gradient(g594,X,Y)  # MethodError: Cannot `convert` an object of type Tangent{A594, NamedTuple{(:x,), Tuple{Float64}}} to an object of type ZeroTangent
        @test ∇[1] == [(x = 2.0,); (x = 2.0,)]
        @test vec(∇[1]) == [(x = 2.0,); (x = 2.0,)]
        @test ∇[2] == [1 1; 1 1]
    end

    # overflow

    struct M{T,B}
        a::T
        b::B
    end
    @test_skip m, b = pullback(nameof, M)  # StackOverflowError
    @test_skip @test b(m) == (nothing, nothing)

end

#####
##### Zygote/test/utils.jl
#####

