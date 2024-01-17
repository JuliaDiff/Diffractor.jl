module tangent
using Diffractor
using Diffractor: AbstractZeroBundle, ZeroBundle, DNEBundle, TaylorBundle, ExplicitTangentBundle
using Diffractor:TaylorTangentIndex, CanonicalTangentIndex
using Diffractor: ExplicitTangent, TaylorTangent, truncate
using ChainRulesCore
using Test

@testset "AbstractZeroBundle" begin
    @testset "Hierachy" begin
        @test ZeroBundle <: AbstractZeroBundle
        @test DNEBundle <: AbstractZeroBundle
        @test ZeroBundle{1} <: AbstractZeroBundle{1}
        @test ZeroBundle{1,typeof(getfield)} <: AbstractZeroBundle{1,typeof(getfield)}
    end

    @testset "Display" begin
        @test repr(ZeroBundle{1}(2.0)) == "ZeroBundle{1}(2.0)"
        #=
        Overloading of Type printing is disabled for now
        @test repr(DNEBundle{1}(getfield)) == "DNEBundle{1}(getfield)"

        @test repr(ZeroBundle{1}) == "ZeroBundle{1}"
        @test repr(ZeroBundle{1, Float64}) == "ZeroBundle{1, Float64}"

        @test repr((ZeroBundle{N, Float64} where N).body) == "ZeroBundle{N, Float64}"

        @test repr(typeof(DNEBundle{1}(getfield))) == "DNEBundle{1, typeof(getfield)}"
        =#
    end
end

@testset "AD through constructor" begin
    # https://github.com/JuliaDiff/Diffractor.jl/issues/152
    # Though we have now removed the underlying cause, we keep this as a regression test just in case
    struct Foo152
        x::Float64
    end

    # Unit Test
    cb = TaylorBundle{1, Foo152}(Foo152(23.5), (Tangent{Foo152}(;x=1.0),))
    tti = TaylorTangentIndex(1,)
    @test cb[tti] == Tangent{Foo152}(; x=1.0)

    # Integration  Test
    let var"'" = Diffractor.PrimeDerivativeFwd
        f(x) = Foo152(x)
        @test f'(23.5) == Tangent{Foo152}(; x=1.0)
    end
end

@testset "getindex" begin
    tt = TaylorBundle{2}(1.5, (1.0, 2.0))
    @test tt[TaylorTangentIndex(1)] == 1.0
    @test tt[TaylorTangentIndex(2)] == 2.0
    @test tt[CanonicalTangentIndex(1)] == 1.0
    @test tt[CanonicalTangentIndex(2)] == 1.0
    @test tt[CanonicalTangentIndex(3)] == 2.0

    et = ExplicitTangentBundle{2}(1.5, (1.0, 2.0, 3.0))
    @test_throws DomainError et[TaylorTangentIndex(1)] == 1.0
    @test et[TaylorTangentIndex(2)] == 3.0
    @test et[CanonicalTangentIndex(1)] == 1.0
    @test et[CanonicalTangentIndex(2)] == 2.0
    @test et[CanonicalTangentIndex(3)] == 3.0

    zb = ZeroBundle{2}(1.5)
    @test zb[TaylorTangentIndex(1)] == ZeroTangent()
    @test zb[TaylorTangentIndex(2)] == ZeroTangent()
    @test zb[CanonicalTangentIndex(1)] == ZeroTangent()
    @test zb[CanonicalTangentIndex(2)] == ZeroTangent()
    @test zb[CanonicalTangentIndex(3)] == ZeroTangent()
end

@testset "promote" begin
    @test promote_type(
        typeof(ExplicitTangentBundle{1}([2.0, 4.0], ([20.0, 200.0],))),
        typeof(TaylorBundle{1}([2.0, 4.0], ([20.0, 200.0],)))
    ) <: ExplicitTangentBundle{1, Vector{Float64}}

    @test promote_type(TaylorBundle{1, Float64, Tuple{Float64}}, ZeroBundle{1, Float64}) <: TaylorBundle{1, Float64, Tuple{Float64}}
    @test promote_type(ExplicitTangentBundle{1, Float64, Tuple{Float64}}, ZeroBundle{1, Float64}) <: ExplicitTangentBundle{1, Float64, Tuple{Float64}}
end
@testset "convert" begin
    @test convert(TaylorBundle{1, Float64, Tuple{Float64}}, ZeroBundle{1}(1.4)) == TaylorBundle{1}(1.4, (0.0,))
    @test convert(ExplicitTangentBundle{1, Float64, Tuple{Float64}}, ZeroBundle{1}(1.4)) == ExplicitTangentBundle{1}(1.4, (0.0,))

    @test convert(
        typeof(ExplicitTangentBundle{1}([2.0, 4.0], ([20.0, 200.0],))),
        TaylorBundle{1}([2.0, 4.0], ([20.0, 200.0],))
    ) == ExplicitTangentBundle{1}([2.0, 4.0], ([20.0, 200.0],))

    @test convert(
        typeof(ExplicitTangentBundle{2}(1.5, (10.0, 10.0, 20.0,))),
        TaylorBundle{2}(1.5, (10.0, 20.0))
    ) === ExplicitTangentBundle{2}(1.5, (10.0, 10.0, 20.0,))
end
@testset "==" begin
    @test TaylorBundle{1}([2.0, 4.0], ([20.0, 200.0],)) == TaylorBundle{1}([2.0, 4.0], ([20.0, 200.0],))
    @test TaylorBundle{1}([2.0, 4.0], ([20.0, 200.0],)) == ExplicitTangentBundle{1}([2.0, 4.0], ([20.0, 200.0],)) 

    @test ZeroBundle{3}(1.5) == ZeroBundle{3}(1.5)
    @test ZeroBundle{3}(1.5) == TaylorBundle{3}(1.5, (0.0, 0.0, 0.0))
    @test ZeroBundle{3}(1.5) == ExplicitTangentBundle{3}(1.5, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
end

@testset "hash" begin
    @test hash(TaylorBundle{1}([2.0, 4.0], ([20.0, 200.0],))) == hash(TaylorBundle{1}([2.0, 4.0], ([20.0, 200.0],)))
    @test hash(TaylorBundle{1}([2.0, 4.0], ([20.0, 200.0],))) == hash(ExplicitTangentBundle{1}([2.0, 4.0], ([20.0, 200.0],)))

    @test hash(ZeroBundle{3}(1.5)) == hash(ZeroBundle{3}(1.5))
    @test hash(ZeroBundle{3}(1.5)) == hash(TaylorBundle{3}(1.5, (0.0, 0.0, 0.0)))
    @test hash(ZeroBundle{3}(1.5)) == hash(ExplicitTangentBundle{3}(1.5, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)))
end


@testset "truncate" begin
    tt = TaylorTangent((1.0,2.0,3.0,4.0,5.0,6.0,7.0))
    @test truncate(tt, Val(2)) == TaylorTangent((1.0,2.0))
    et = ExplicitTangent((1.0,2.0,3.0,4.0,5.0,6.0,7.0))
    @test truncate(et, Val(2)) == ExplicitTangent((1.0,2.0,3.0))
    @test truncate(et, Val(1)) == TaylorTangent((1.0,))
end


@testset "Bad Partial Types" begin
    @test_throws DomainError TaylorBundle{1}(1.5, (ZeroTangent,))   # mistakenly passing a type rather than a value
    @test_throws DomainError TaylorBundle{1}(1.5, (:a,))
    @test_throws DomainError TaylorBundle{1}(1.5, (nothing,))
    @test_throws DomainError TaylorBundle{1}(1.5, ("x",))
end

end  # module
