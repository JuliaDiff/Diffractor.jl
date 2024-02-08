module Diffractor

using StructArrays
using PrecompileTools

export ∂⃖, gradient

const CC = Core.Compiler

const GENERATORS = Expr[]

@static if VERSION ≥ v"1.11.0-DEV.1498"
    import .CC: get_inference_world
    using Base: get_world_counter
else
    import .CC: get_world_counter, get_world_counter as get_inference_world
end

@recompile_invalidations begin
    include("runtime.jl")
    include("interface.jl")
    include("utils.jl")
    include("tangent.jl")
    include("jet.jl")

    include("stage1/generated.jl")
    include("stage1/forward.jl")
    include("stage1/recurse_fwd.jl")
    include("stage1/mixed.jl")
    include("stage1/broadcast.jl")

    include("stage2/interpreter.jl")
    include("stage2/lattice.jl")
    include("stage2/abstractinterpret.jl")
    include("stage2/tfuncs.jl")
    include("stage2/forward.jl")

    include("codegen/forward.jl")
    include("analysis/forward.jl")
    include("codegen/forward_demand.jl")
    include("codegen/reverse.jl")

    include("extra_rules.jl")

    include("higher_fwd_rules.jl")

    include("debugutils.jl")

    include("stage1/termination.jl")
    include("AbstractDifferentiation.jl")
end

end
