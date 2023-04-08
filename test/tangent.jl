module tagent
using Diffractor
using Diffractor: AbstractZeroBundle, ZeroBundle, DNEBundle
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

        @test repr(typeof(DNEBundle{1}(getfield))) == "DNEBundle{1, typeof(getfield)}"
    end
end

end  # module
