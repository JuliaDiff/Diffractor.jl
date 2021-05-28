module Diffractor

export ∂⃖, gradient

include("runtime.jl")
include("interface.jl")
include("extra_rules.jl")
include("utils.jl")
include("tangent.jl")
include("jet.jl")

include("stage1/generated.jl")
include("stage1/termination.jl")
include("stage1/forward.jl")
include("stage1/recurse_fwd.jl")

include("higher_fwd_rules.jl")

include("debugutils.jl")

end
