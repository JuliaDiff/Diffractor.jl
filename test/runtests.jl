using Diffractor
using Test

@testset verbose=true "Diffractor.jl" begin

    @testset verbose=true "Diffractor 0.1's own unit tests" begin
        include("diffractor_01.jl")
    end

    @testset verbose=true "pseudo-Flux higher-order" begin
        # Higher order control flow not yet supported (https://github.com/JuliaDiff/Diffractor.jl/issues/24)
        # include("pinn.jl")
    end

    @testset verbose=true "ChainRules integration" begin
        include("chainrules.jl")
    end

    @testset verbose=true "from ForwardDiff" begin
        include("forwarddiff.jl")
    end

    @testset verbose=true "from Zygote's features.jl" begin
        include("zygote_features.jl")
    end

    @testset verbose=true "from Zygote's gradcheck.jl" begin
        include("zygote_gradcheck.jl")
    end

end
