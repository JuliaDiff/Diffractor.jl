
# This file contains many examples borrowed from Zygote's tests, 
# all files except its "gradcheck.jl" which is elsewhere.

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

function withgradient(f, x...)
    y, b = Diffractor.∂⃖{1}()(f, x...)
    (val=y, grad=map(unthunk, Base.tail(b(one(y)))))
end

using ChainRulesCore: backing, canonicalize, AbstractTangent

# These pirate methods greatly simplify RHS of tests:
Base.isapprox(x::Tangent, y::NamedTuple) = all(map(isapprox, backing(canonicalize(x)), y))
Base.isapprox(x::Tangent{<:Tuple}, y::Tuple) = all(map(isapprox, backing(x), y))
Base.isapprox(x::Array{<:AbstractTangent}, y::Array) = all(map(isapprox, x, y))
Base.isapprox(::AbstractZero, ::Nothing) = true
Base.isapprox(::AbstractZero, y) = iszero(y)

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

@testset "complex arrays" begin

    # Zygote issue 705
    @test gradient(x -> imag(sum(exp, x)), [1,2,3]) |> only |> isZero
    @test gradient(x -> imag(sum(exp, x)), [1+0im,2,3])[1] ≈ im .* exp.(1:3)

end

#####
##### Zygote/test/features.jl
#####

# This file isn't really organised; here it's broken arbitrarily into testsets each about a page long.
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

    @test gradient(x -> 1, 2) |> only |> isZero

    @test gradient(t -> t[1]*t[2], (2, 3)) |> only |> Tuple == (3, 2)
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
    @test gradient(mul_kw, 2, 3) == (3, 2)  # passes at REPL, not in testset?

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
    @test gradient(x -> sum(f249(; x=x, idx=1:1)), ones(2))[1] == [1, 0]

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

    @test gradient(x -> one(eltype(x)), rand(10)) |> only |> isZero

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

    @test_broken gradient(pow_try, 1) == (2,)  # BoundsError: attempt to access 6-element Vector{Core.Compiler.BasicBlock} at index [0]

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
    @test_broken gradient(x -> (@time x^2), 3) == (6,)  # BoundsError: attempt to access 12-element Vector{Core.Compiler.BasicBlock} at index [0]

    # kwarg splat

    g516(; kwargs...) = kwargs[:x] * kwargs[:z]
    h517(somedata) = g516(; somedata...)
    @test gradient(h517, (; x=3.0, y=4.0, z=2.3)) |> only ≈ (; x=2.3, y=0.0, z=3.0)
    @test_broken gradient(h517, Dict(:x=>3.0, :y=>4.0, :z=>2.3)) == (Tangent{NamedTuple{(:x, :y, :z), Tuple{Float64,Float64,Float64}}}(; x=2.3, y=0.0, z=3.0),)  # ERROR: (1, get(d::IdDict{K, V}, key, default) where {K, V} @ Base iddict.jl:101, :($(Expr(:foreigncall, :(:jl_eqtable_get), Any, svec(Any, Any, Any), 0, :(:ccall), :(%1), Core.Argument(3), Core.Argument(4)))))

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
    @test_broken gradient(baz, (1.0,)) |> only ≈ (1.0,)

    # ChainRules represents these as the same Tangent as immutable structs, but is that ideal?

    @test gradient(x -> x.value^2 + x.value, MyMutable(3)) === (Tangent{MyMutable}(value = 7.0,),)
    @test gradient(x -> x.value^2 + x.value, MyMutable(3)) |> only ≈ (value = 7.0,)  # with new isapprox methods

    @test gradient(x -> x.x^2 + x.x, Ref(3)) == (Tangent{Base.RefValue{Int}}(x = 7.0,),)
    @test gradient(x -> real(x.x^2 + im * x.x), Ref(4)) == (Tangent{Base.RefValue{Int}}(x = 8.0,),)

    # Field access of contents:
    @test_broken gradient(x -> abs2(x.x) + 7 * x.x.re, Ref(1+im)) |> only ≈ (; x = 9.0 + 2.0im)
    @test gradient(x -> abs2(x[1].x) + 7 * x[1].x.re, [Ref(1+im)]) |> only ≈ [(x = 9.0 + 2.0im,)]
    @test gradient(x -> abs2(x[1].x) + 7 * real(x[1].x), [Ref(1+im)]) |> only ≈ [(x = 9.0 + 2.0im,)]  # worked on Zygote 0.6.0, 0.6.20
    @test gradient(x -> abs2(x[].x) + 7 * real(x[].x), Ref(Ref(1+im))) |> only ≈ (x = (x = 9.0 + 2.0im,),)  # Zygote gives nothing, same in 0.6.0

    # Array of mutables:
    @test_broken gradient(x -> sum(getindex.(x).^2), Ref.(1:3)) |> only ≈ [(;x=2i) for i in 1:3]  # MethodError: no method matching one(::Base.RefValue{Int64})
    @test gradient(x -> sum(abs2∘getindex, x), Ref.(1:3)) |> only ≈ [(;x=2i) for i in 1:3]  # Tangent for the primal Base.Pairs{Symbol, Union{}, Tuple{}, NamedTuple{(), Tuple{}}} should be backed by a AbstractDict type, not by NamedTuple

    @test_broken gradient(x -> (getindex.(x).^2)[1], Ref.(1:3))[1][1] ≈ (x=2.0,)  # rest are (x = 0.0,), but nothing would be OK too
    @test_broken gradient(x -> (prod.(getindex.(x)))[1], Ref.(eachcol([1 2; 3 4])))[1][1] ≈ (x = [3.0, 1.0],)  # MethodError: no method matching one(::SubArray{Int64, 1, Matrix{Int64}, Tuple{Base.Slice{Base.OneTo{Int64}}, Int64}, true})

    # Broadcasting over Ref:
    @test gradient(x -> sum(sum, x .* [1,2,3]), Ref([4,5])) |> only ≈ (x = [6.0, 6.0],) 
    @test gradient(x -> sum(sum, Ref(x) .* [1,2,3]), [4,5]) == ([6.0, 6.0],)

end

@testset "NamedTuples" begin

    @test gradient(x -> x.a, (a=1, b=2)) == (Tangent{NamedTuple{(:a, :b), Tuple{Int,Int}}}(a = 1,),)
    @test gradient(x -> x[1].a, [(a=1, b=2)]) |> only ≈ [(a = 1, b = nothing)]

    @test gradient(x -> x[1].a, [(a=1, b=2), (a=3, b=4)]) |> only == [Tangent{NamedTuple{(:a, :b), Tuple{Int64, Int64}}}(a = 1.0,), ZeroTangent()]

    # Mix with Ref
    @test gradient(x -> x[].a, Ref((a=1, b=2))) |> only ≈ (x = (a = 1, b = nothing),)
    @test gradient(x -> x[1][].a, [Ref((a=1, b=2)), Ref((a=3, b=4))]) |> only |> first ≈ (x = (a = 1, b = nothing),)
    @test gradient(x -> x[1].a, [(a=1, b=2), "three"]) |> only |> first ≈ (a = 1, b = nothing)

    @testset "indexing kwargs: PR 1286" begin
        # https://github.com/FluxML/Zygote.jl/pull/1286
        inner_lit_index(; kwargs...) = kwargs[:x]
        outer_lit_index(; kwargs...) = inner_lit_index(; x=kwargs[:x])

        inner_dyn_index(k; kwargs...) = kwargs[k]
        outer_dyn_index(k; kwargs...) = inner_dyn_index(k; x=kwargs[k])

        @test gradient(x -> outer_lit_index(; x), 0.0) == (1.0,)
        @test gradient((x, k) -> outer_dyn_index(k; x), 0.0, :x) == (1.0, NoTangent())
    end
end

@testset "Pairs" begin

    @test gradient(x->10*pairs((a=x, b=2))[1], 100)[1] === 10.0
    @test gradient(x->10*pairs((a=x, b=2))[2], 100) |> only |> isZero

    foo387(; kw...) = 1
    @test_skip gradient(() -> foo387(a=1,b=2.0)) === ()  # Here Diffractor returns a function, by design

    @test gradient(x->10*(x => 2)[1], 100) === (10.0,)
    @test gradient(x->10*(x => 2)[2], 100) |> only |> isZero

    @test gradient(x-> (:x => x)[2], 17) == (1,)

    d = Dict(:x=>1.0, :y=>3.0);
    @test_broken gradient(d -> Dict(:x => d[:x])[:x], d) == (Dict(:x => 1),)  # BoundsError: attempt to access 3-element Vector{Core.Compiler.BasicBlock} at index [4]

    # https://github.com/FluxML/Zygote.jl/pull/1295
    no_kwarg_grad(x; kwargs...) = x[kwargs[:i]]
    @test gradient(x -> no_kwarg_grad(x; i=1), [1]) == ([1],)
end

@testset "Iterators" begin

    # enumerate

    @test_broken gradient(1:5) do xs
        sum([x^i for (i,x) in enumerate(xs)])
    end == ([1, 4, 27, 256, 3125],)

    @test_broken gradient([1,10,100]) do xs
        sum([xs[i]^i for (i,x) in enumerate(xs)])
    end == ([1, 2 * 10^1, 3 * 100^2],)

    @test gradient([1,10,100]) do xs
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

    # Issue 1150

    @test_broken gradient(x -> sum([x[i] for i in 1:3 if i != 100]), [1,2,3])[1] == [1,1,1]
    @test_broken gradient(x -> sum(map(i -> x[i], filter(i -> i != 100, 1:3))), [1,2,3])[1] == [1,1,1]

end

@testset "adjoints of Iterators.product, PR 1170" begin
    # Adapted from Zygote's file test/lib/array.jl

    y, back = pullback(Iterators.product, 1:5, 1:3, 1:2)
    @test_broken back(collect(y)) == (NoTangent(), [6.0, 12.0, 18.0, 24.0, 30.0], [10.0, 20.0, 30.0], [15.0, 30.0])
    @test_broken back([(NoTangent(), j, k) for i in 1:5, j in 1:3, k in 1:2]) == (NoTangent(), [10.0, 20.0, 30.0], [15.0, 30.0])
    @test_broken back([(i, NoTangent(), k) for i in 1:5, j in 1:3, k in 1:2]) == ([6.0, 12.0, 18.0, 24.0, 30.0], NoTangent(), [15.0, 30.0])
    @test_broken back([(i, j, NoTangent()) for i in 1:5, j in 1:3, k in 1:2]) == ([6.0, 12.0, 18.0, 24.0, 30.0], [10.0, 20.0, 30.0], NoTangent())

    # This was wrong before https://github.com/FluxML/Zygote.jl/pull/1170
    @test_broken gradient(x -> sum([y[2] * y[3] for y in Iterators.product(x, x, x, x)]), [1,2,3,4])[1] ≈ [320, 320, 320, 320]  # MethodError: no method matching copy(::Nothing)
    @test_broken gradient(x -> sum(y[2] * y[3] for y in Iterators.product(x, x, x, x)), [1,2,3,4])[1] ≈ [320, 320, 320, 320]  # accum(a::Tuple{NoTangent, NoTangent}, b::Tuple{Tuple{ZeroTangent, NoTangent}}), tail_pullback
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
    @test gradient((x,y) -> sum((z->z^2+y[1]), x), [1,2,3], [4,5]) == ([2, 4, 6], [3, 0])
    @test_broken gradient((x,y) -> sum(map((z->z^2+y[1]), x)), [1,2,3], [4,5]) == ([2, 4, 6], [3, 0]) # AssertionError: Base.issingletontype(typeof(f))
    @test gradient((x,y) -> mapreduce((z->z^2+y[1]), +, x), [1,2,3], [4,5]) == ([2, 4, 6], [3, 0])

    # type unstable
    @test gradient(xs -> sum((x -> x<2 ? false : x^2).(xs)), [1,2,3])[1][2:3] == [4, 6]
    @test gradient(xs -> sum((x -> x<2 ? false : x^2), xs), [1,2,3])[1][2:3] == [4, 6]
    @test_broken gradient(xs -> sum(map((x -> x<2 ? false : x^2), xs)), [1,2,3])[1][2:3] == [4, 6]  # AssertionError: ∂f isa TaylorBundle || ∂f isa TangentBundle{1}
    @test gradient(xs -> mapreduce((x -> x<2 ? false : x^2), +, xs), [1,2,3])[1][2:3] == [4, 6]

    # with Ref, Val, Symbol
    @test gradient(x -> sum(x .+ Ref(x[1])), [1,2,3]) == ([4,1,1],)
    @test gradient(x -> sum(x .+ (x[1],)), [1,2,3]) == ([4,1,1],)
    @test gradient(x -> sum((first∘tuple).(x, :ignore)), [1,2,3]) == ([1,1,1],)
    @test gradient(x -> sum((first∘tuple).(x, Symbol)), [1,2,3]) == ([1,1,1],)
    _f(x,::Val{y}=Val(2)) where {y} = x/y
    @test gradient(x -> sum(_f.(x, Val(2))), [1,2,3]) == ([0.5, 0.5, 0.5],)
    @test gradient(x -> sum(_f.(x)), [1,2,3]) == ([0.5, 0.5, 0.5],)
    @test_broken gradient(x -> sum(map(_f, x)), [1,2,3]) == ([0.5, 0.5, 0.5],)  # InexactError
    @test gradient(x -> sum(map(_f, x)), [1,2,3.0]) == ([0.5, 0.5, 0.5],)

    # with Bool
    @test gradient(x -> sum(1 .- (x .> 0)), randn(5)) == (NoTangent(),)

    @test gradient(x -> sum((y->1-y).(x .> 0)), randn(5)) == (NoTangent(),)
    @test gradient(x -> sum(x .- (x .> 0)), randn(5)) == ([1,1,1,1,1],)

    @test gradient(x -> sum(x ./ [1,2,4]), [1,2,pi]) == ([1.0, 0.5, 0.25],)
    @test_broken gradient(x -> sum(map(/, x, [1,2,4])), [1,2,pi]) == ([1.0, 0.5, 0.25],)  # MethodError: no method matching (::Diffractor.∂⃖recurse{1})(::typeof(Core.arrayset), ::Bool, ::Vector{Float64}, ::Float64, ::Int64)

    # negative powers
    @test gradient((x,p) -> sum(x .^ p), [1.0,2.0,4.0], [1,-1,2])[1] ≈ [1.0, -0.25, 8.0]
    @test gradient((x,p) -> sum(x .^ p), [1.0,2.0,4.0], -1)[1] ≈ [-1.0, -0.25, -0.0625]
    @test gradient((x,p) -> sum(z -> z^p, x), [1.0,2.0,4.0], -1)[1] ≈ [-1.0, -0.25, -0.0625]
    @test gradient((x,p) -> mapreduce(z -> z^p, +, x), [1.0,2.0,4.0], -1)[1] ≈ [-1.0, -0.25, -0.0625]

    # second order
    @test gradient(x -> sum(gradient(y -> sum(y.^2), x)[1]), [1, 2])[1] ≈ [2, 2]
    @test_broken gradient(x -> sum(gradient(y -> sum(sin.(y)), x)[1]), [1, 2])[1] ≈ [-0.8414709848078965, -0.9092974268256817]  # MethodError: no method matching Diffractor.Jet(::Int64, ::Float64, ::Tuple{Float64, Float64}) -> MethodError: no method matching copy(::Nothing)
    @test_broken gradient(x -> sum(abs, gradient(y -> sum(log.(2 .* exp.(y)) .^ 2), x)[1]), [1, 2])[1] ≈ [2,2]

    # getproperty, Tangents, etc
    @test_broken gradient(xs -> sum((x->x.im^2).(xs)), [1+2im,3])[1] == [4im, 0]  # SILENTLY WRONG ANSWER
    @test gradient(xs -> sum((x->x.im^2), xs), [1+2im,3])[1] == [4im, 0]  # MethodError: no method matching lastindex(::Diffractor.OpticBundle{Int64})
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

# This file contains tests of jacobian and hessian functions,
# and of adjoints of ForwardDiff functions.
# To add them, we would need to define various hessian & jacobian functions,
# possibly as in "forwarddiff.jl" in the tests here, possibly as exported functions.


@test_skip @testset "hessian, #hess $hess" for hess in HESSIANS
    @test hess(x -> x[1]*x[2], randn(2)) ≈ [0 1; 1 0]
    @test hess(((x,y),) -> x*y, randn(2)) ≈ [0 1; 1 0] 

    @test hess(x -> sum(x.^3), [1 2; 3 4]) ≈ Diagonal([6, 18, 12, 24])
    @test hess(sin, pi/2) ≈ -1

    @test_throws Exception hess(sin, im*pi)
    @test_throws Exception hess(x -> x+im, pi)
    @test_throws Exception hess(identity, randn(2))
end

@test_skip @testset "diagonal hessian" begin
    @test diaghessian(x -> x[1]*x[2]^2, [1, pi]) == ([0, 2],)

    xs, y = randn(2,3), rand()
    f34(xs, y) = xs[1] * (sum(xs .^ (1:3)') + y^4)  # non-diagonal Hessian, two arguments

    dx, dy = diaghessian(f34, xs, y)
    @test size(dx) == size(xs)
    @test vec(dx) ≈ diag(hessian(x -> f34(x,y), xs))
    @test dy ≈ hessian(y -> f34(xs,y), y)


    zs = randn(7,13)  # test chunk mode
    f713(zs) = sum(vec(zs)' .* exp.(vec(zs)))
    @test vec(diaghessian(f713, zs)[1]) ≈ diag(hessian(f713, zs))

    @test_throws Exception diaghessian(sin, im*pi)
    @test_throws Exception diaghessian(x -> x+im, pi)
    @test_throws Exception diaghessian(identity, randn(2))
end

@test_skip @testset "jacobian(f, args...), $jacobian" for jacobian in JACOBIANS
    @test jacobian(identity, [1,2])[1] == [1 0; 0 1]
    @test withjacobian(identity, [1,2]) == (val = [1,2], grad = ([1 0; 0 1],))

    j1 = jacobian((a,x) -> a.^2 .* x, [1,2,3], 1)
    @test j1[1] ≈ Diagonal([2,4,6])
    @test j1[2] ≈ [1, 4, 9]
    @test j1[2] isa Vector

    j2 = jacobian((a,t) -> sum(a .* t[1]) + t[2], [1,2,3], (4,5))  # scalar output is OK
    @test j2[1] == [4 4 4]
    @test j2[1] isa Matrix
    @test j2[2] === nothing  # input other than Number, Array is ignored

    j3 = jacobian((a,d) -> prod(a, dims=d), [1 2; 3 4], 1)
    @test j3[1] ≈ [3 1 0 0; 0 0 4 2]
    @test j3[2] ≈ [0, 0]  # pullback is always Nothing, but array already allocated

    j4 = jacobian([1,2,-3,4,-5]) do xs
    map(x -> x>0 ? x^3 : 0, xs)  # pullback gives Nothing for some elements x
    end
    @test j4[1] ≈ Diagonal([3,12,0,48,0])

    j5 = jacobian((x,y) -> hcat(x[1], y), fill(pi), exp(1))  # zero-array
    @test j5[1] isa Matrix
    @test vec(j5[1]) == [1, 0]
    @test j5[2] == [0, 1]

    @test_throws ArgumentError jacobian(identity, [1,2,3+im])
    @test_throws ArgumentError jacobian(sum, [1,2,3+im])  # scalar, complex

    f6(x,y) = abs2.(x .* y)
    g6 = gradient(first∘f6, [1+im, 2], 3+4im)
    j6 = jacobian((x,y) -> abs2.(x .* y), [1+im, 2], 3+4im)
    @test j6[1][1,:] ≈ g6[1]
    @test j6[2][1] ≈ g6[2]
end

# using ForwardDiff

@test_skip @testset "adjoints of ForwardDiff functions" begin
    f1(x) = ForwardDiff.gradient(x -> sum(exp.(x.+1)), x)
    x1 = randn(3,7)
    @test jacobian(f1, x1)[1] ≈ ForwardDiff.jacobian(f1, x1)

    f2(x) = ForwardDiff.jacobian(x -> log.(x[1:3] .+ x[2:4]), x)
    x2 = rand(5) .+ 1
    @test jacobian(f2, x2)[1] ≈ ForwardDiff.jacobian(f2, x2)

    f3(x) = sum(ForwardDiff.hessian(x -> sum(x .^2 .* x'), x)[1:4:end])
    x3 = rand(3)
    @test gradient(f3, x3)[1] ≈ ForwardDiff.gradient(f3, x3)

    @test gradient(x -> ForwardDiff.derivative(x -> x^4, x), 7) == (4 * 3 * 7^2,)

    f4(x) = ForwardDiff.derivative(x -> [x,x^2,x^3], x)
    @test jacobian(f4, pi)[1] ≈ ForwardDiff.derivative(f4, pi)

    # Tests from https://github.com/FluxML/Zygote.jl/issues/769
    f(x) = [2x[1]^2 + x[1],x[2]^2 * x[1]]
    g1(x) = sum(ForwardDiff.jacobian(f,x))
    out,back = pullback(g1,[2.0,3.2])
    stakehouse = back(1.0)[1]
    @test typeof(stakehouse) <: Vector
    @test size(stakehouse) == (2,)
    @test stakehouse ≈ ForwardDiff.gradient(g1,[2.0,3.2])

    g2(x) = prod(ForwardDiff.jacobian(f,x))
    out,back = Zygote.pullback(g2,[2.0,3.2])
    @test_skip back(1.0)[1] == ForwardDiff.gradient(g2,[2.0,3.2])  # contains NaN, @adjoint prod isn't careful

    g3(x) = sum(abs2,ForwardDiff.jacobian(f,x))
    out,back = pullback(g3,[2.0,3.2])
    @test back(1.0)[1] == ForwardDiff.gradient(g3,[2.0,3.2])
end

@testset "broadcasted for unary minus" begin
    # TODO add test from https://github.com/FluxML/NNlib.jl/issues/432, needs a hessian function
end

#####
##### Zygote issues
#####

@testset "issue 954: range constructors" begin
    @test_broken gradient(x -> (x:3)[1], 1.2) == (1,)
    @test_broken gradient(x -> (x:1.0:3)[1], 1.2) == (1,)
end

@testset "issue 1071: NamedTuple constructor" begin
    x = [:a=>1, :b=>2]
    @test gradient(x ->  x[1].second, x) |> only ≈ [(first = 0, second = 1,), ZeroTangent()]
    @test_broken gradient(x ->  NamedTuple(x).a, x) |> only ≈ [(first = 0, second = 1,), ZeroTangent()]  # ERROR: (1, get(d::IdDict{K, V}, key, default) where {K, V} @ Base iddict.jl:101, :($(Expr(:foreigncall, :(:jl_eqtable_get), Any, svec(Any, Any, Any), 0, :(:ccall), :(%1), Core.Argument(3), Core.Argument(4)))))
    
    y = (1, 2)
    @test_broken gradient(y -> NamedTuple{(:a,:b)}(y).a, y)[1] isa Tangent{<:Tuple}  # makes a Tangent{NamedTuple}!
    @test_broken gradient(y -> NamedTuple{(:a, :b)}(y).a, y)[1] ≈ (1, 0)
end

@testset "issue 1072: map on NamedTuple" begin
    x = (; a=1, b=2)
    @test map(sqrt, x) == (a = 1.0, b = 1.4142135623730951)
    @test_broken gradient(x ->  map(sqrt, x).a, x) |> only ≈ (a = 0.5, b = nothing)  # MethodError: no method matching unzip_tuple(::Vector{Tuple{NoTangent, Float64}})
end

@testset "issue 1198: permuting values" begin

    function random_hermitian_matrix(N, specrad=1.0)
        σ = 1 / √N
        X = σ * (randn(N, N) + rand(N, N) * im) / √2
        H = specrad * (X + X') / (2 * √2)
    end

    function random_state_vector(N)
        Ψ = rand(N) .* exp.((2π * im) .* rand(N))
        Ψ ./= norm(Ψ)
        return Ψ
    end

    function cheby(Ψ::AbstractVector, H::AbstractMatrix, dt)
        a = [0.9915910021578431, 0.18282201929219635, 0.008403088661031203, 0.000257307553262815]
        Δ = 6.0
        E_min = -3.0
        β = (Δ / 2) + E_min
        c = -2im / Δ

        v0 = Ψ
        ϕ = a[1] * v0
        v1 = c * (H * v0 - β * v0)
        ϕ = ϕ + a[2] * v1

        c *= 2
        for i = 3:length(a)
            v2 = c * (H * v1 - β * v1) + v0
            ϕ = ϕ + a[i] * v2
            
            v0, v1, v2 = v1, v2, v0                                # doesn't work
            # aux = v0; v0 = v1; v1 = v2; v2 = aux                   # doesn't work
            # aux = 1 * v0; v0 = 1 * v1; v1 = 1 * v2; v2 = 1 * aux   # works
        end

        return exp(-1im * β * dt) * ϕ
    end

    N = 2
    dt = 0.1

    Ψ0 = random_state_vector(N)
    Ψ1 = random_state_vector(N)
    H0 = random_hermitian_matrix(N)
    H1 = random_hermitian_matrix(N)

    ϵ = 1.0
    res1 = abs2(Ψ1 ⋅ cheby(Ψ0, H0 + ϵ * H1, dt))
    @test_skip res2, _ = pullback(ϵ -> abs2(Ψ1 ⋅ cheby(Ψ0, H0 + ϵ * H1, dt)), ϵ)  # TypeError: in typeassert, expected Int64, got a value of type Nothing

    @test_broken abs(res1 - res2) < 1e-12

end

@testset "some rules" begin

    # https://github.com/FluxML/Zygote.jl/issues/1190
    g1 = gradient(x -> sum(normalize(x)), [1,2,3,4.0])[1]
    @test_broken g1 ≈ vec(gradient(x -> sum(normalize(x)), [1 2; 3 4.0])[1])  # SILENTLY WRONG ANSWER!

    # https://github.com/FluxML/Zygote.jl/issues/1201
    struct Point1201; x; y; end
    tst1201(p) = [p[1] p[2]; p[3] p[4]]
    pointar = [Point1201(1,2), Point1201(3,4), Point1201(5,6), Point1201(7,8)]

    @test gradient(p -> tst1201(p)[1,2].x, pointar) |> only ≈ [0, (x = 1, y = nothing), 0, 0]

end

const _BC_FIVE = 5
_B_FIVE = 5

@testset "issue 1177, global + $prime^2" for prime in [Diffractor.PrimeDerivativeBack, Diffractor.PrimeDerivativeFwd]
    # https://github.com/FluxML/Zygote.jl/issues/1177
    let var"'" = prime
        # Non-const global:
        f1(x) = 4*x^2 + _B_FIVE*x + 10
        g1 = f1'
        @test g1'(25) ≈ 8.0

        # Without global:
        f2(x, b2=5) = 4*x^2 + b2*x + 10
        g2 = f2'
        @test g2'(25) ≈ 8.0

        # With const global:
        f3(x) = 4*x^2 + _BC_FIVE*x + 10
        g3 = f3'
        @test g3'(25) ≈ 8.0
    end
end

@testset "issue 1127: mutable struct" begin

    mutable struct Particle
        q::Array{Float64,1}
        p::Array{Float64,1} 
        m::Float64
    end

    _dot(arr1,arr2) = sum(.*(arr1,arr2))
    _modsquare(arr1) = _dot(arr1,arr1)
    _norm(arr1) = sqrt(_dot(arr1,arr1))

    function energy(p1::Particle, p2::Particle)
        σ = 0.1
        ϵ = 700.0
        perg = -1.0*100.0 * p1.m * p2.m * (1.0/(_norm(p1.q - p2.q)))
    end

    function hamiltonian(parray)
        s = 0.0
        for i in 1:length(parray)-1
            for j in i+1:(length(parray))
                s = s + energy(parray[i], parray[j])
            end
        end
        return s + sum([0.5*_modsquare(p.p)/p.m for p  in parray])
    end

    p1 = Particle(zeros(3), ones(3), 1.0)
    p2 = Particle(zeros(3) .+ 0.1, ones(3), 1.0)

    @test_skip gradient(hamiltonian, [p1, p2])  # TypeError: in typeassert, expected Int64, got a value of type Nothing

end

@testset "issue 1150: filter, and Iterators.filter" begin
    A = [0.0 1.0 2.0]
    @test_broken gradient(x -> sum([x[i] for i in 1:3 if i != 100]), A) == ([1 1 1],)
    @test_broken gradient(x -> sum(map(i -> x[i], filter(i -> i != 100, 1:3))), A) == ([1 1 1],)  # AssertionError: Base.issingletontype(typeof(f))
end

@testset "issue 1150: other comprehensions" begin
    @test_broken gradient(x -> sum(Float64[x^2 for i in 1:2]), 3.0) == (12.0,)
    @test_broken gradient(xs -> sum([i/j for i in xs for j in xs]), [1,2,3.0])[1] ≈ [-4.1666666, 0.3333333, 1.166666] atol=1e-4
end

@testset "issue 1181: maximum" begin
    foo1181(s) = s / maximum(eachindex([1, 1, 1, 1]))
    bar1181(s) = s / length([1, 1, 1, 1])

    @test gradient(bar1181, 1) == (0.25,)
    @test gradient(foo1181, 1) == (0.25,)
end

@testset "issue 1208: NamedTuple + 2nd order" begin

    NT = (weight = randn(Float32, 2, 2),)
    W = NT.weight
    X = randn(Float32, 2)

    G = gradient(W) do w
        sum(gradient(X) do x
            sum(w * x)^2
        end[1])
    end[1]

    @test G ≈ gradient(NT) do nt
        sum(gradient(X) do x
            sum(nt.weight * x)^2
        end[1])
    end[1].weight
    
end

@testset "issue 1247: iteration on ranges" begin
    @test gradient(r -> sum(x for x in r), 0:1.0) == ([1,1],)
    @test gradient(first, 0:1.0) == ([1,0],)
end

@testset "issue 1290: comprehension" begin
    # Unsolved in Zygote at the time of writing

    function loss_adjoint1(p)
        prediction = p .* ones(2,100)
        prediction = [prediction[:, i] for i in axes(prediction, 2)]
        sum(sum.(prediction))
    end
    function loss_adjoint3(p)  # does not re-use name, 3x faster, same answer
        prediction = p.*ones(2,100)
        prediction3 = [prediction[:, i] for i in axes(prediction, 2)]
        sum(sum.(prediction3))
    end

    @test_broken gradient(loss_adjoint1, ones(2)) |> only == [100, 100]    
    @test_broken gradient(loss_adjoint3, ones(2)) |> only == [100, 100]
    
    function loss_ToucheSir(p)
        prediction = 2p
        boxed_fn(i) = prediction^i
        # Trigger https://github.com/JuliaLang/julia/issues/15276
        prediction = boxed_fn(2)
        return prediction
    end

    @test_broken gradient(loss_ToucheSir, 1.0) == (8.0,)  # MethodError: no method matching copy(::Nothing)
    
    @test gradient(nt -> 2*nt.a.x, (; a=Ref(1.0))) |> only ≈ (a = (x = 2.0,),)
    @test gradient(nt -> 2*nt.a.x, (; a=Ref(1.0))) |> only isa Tangent{<:NamedTuple}
end

@testset "issue 1236: control flow" begin
    # https://github.com/FluxML/Zygote.jl/issues/1236
    # Unsolved in Zygote at the time of writing

    function f1236(x)
        y = [[x]', [x]]
        r = 0.0
        o = 1.0
        for n in 1:2
            o *= y[n]
            if n < 2
                proj_o = o * [1.0]
            else
                # Error
                proj_o = o
                # Fix
                # proj_o = o * 1.0
            end
            r += proj_o
        end
        return r
    end
    
    function f1236_fix(x)
        y = [[x]', [x]]
        r = 0.0
        o = 1.0
        for n in 1:2
            o *= y[n]
            if n < 2
                proj_o = o * [1.0]
            else
                # Error
                # proj_o = o
                # Fix
                proj_o = o * 1.0
            end
            r += proj_o
        end
        return r
    end
    
    @test gradient(f1236, 1.2)[1] ≈ 3.4
    @test gradient(f1236_fix, 1.2)[1] ≈ 3.4
end

@testset "issue 1271: second order & global scope" begin

    # α, β = randn(2, 2), randn(2, 2)
    α, β = ([1.3608105 -0.6387457; -0.3293626 -0.3191105], [1.4995675 -0.28095096; -0.7656779 1.1175071])

    g1271(v) = map(eachcol(v), eachcol(β)) do x, y
               sum(x.*x.*y)
           end |> sum

    # this fails on Zygote:
    @test_broken gradient(α) do k
        sum(gradient(g1271, k)[1])
    end |> only ≈ [2.999135 -0.56190192; -1.5313558 2.2350142]

    # this works on Zygote:
    @test_broken gradient(α) do k
        sum(gradient(k) do v
            map(eachcol(v), eachcol(β)) do x, y
                sum(x.*x.*y)
            end |> sum
        end[1])
    end |> only ≈ [2.999135 -0.56190192; -1.5313558 2.2350142]

end
