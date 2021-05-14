module Diffractor

export ∂⃖, gradient

include("runtime.jl")
include("interface.jl")
include("extra_rules.jl")
include("utils.jl")
include("taylor.jl")
include("tangent.jl")

include("stage1/generated.jl")
include("stage1/termination.jl")
include("stage1/forward.jl")
include("stage1/recurse_fwd.jl")

include("debugutils.jl")

end
