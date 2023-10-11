module tagent
using Diffractor
using Diffractor: AbstractZeroBundle, ZeroBundle, DNEBundle
using Diffractor: TaylorBundle, TaylorTangentIndex
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
        @test repr(DNEBundle{1}(getfield)) == "DNEBundle{1}(getfield)"

        @test repr(ZeroBundle{1}) == "ZeroBundle{1}"
        @test repr(ZeroBundle{1, Float64}) == "ZeroBundle{1, Float64}"

        @test repr((ZeroBundle{N, Float64} where N).body) == "ZeroBundle{N, Float64}"

        @test repr(typeof(DNEBundle{1}(getfield))) == "DNEBundle{1, typeof(getfield)}"
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

@testset "== and hash" begin
    @test TaylorBundle{1}([2.0, 4.0], ([20.0, 200.0],)) == TaylorBundle{1}([2.0, 4.0], ([20.0, 200.0],))
    @test hash(TaylorBundle{1}(0.0, (0.0,))) == hash(0)
end

@testset "truncate" begin
    tt = TaylorTangent((1.0,2.0,3.0,4.0,5.0,6.0,7.0))
    @test truncate(tt, Val(2)) == TaylorTangent((1.0,2.0))
    et = ExplicitTangent((1.0,2.0,3.0,4.0,5.0,6.0,7.0))
    @test truncate(et, Val(2)) == ExplicitTangent((1.0,2.0,3.0))
    @test truncate(et, Val(1)) == TaylorTangent((1.0,))
end

end  # module
