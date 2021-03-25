module Diffractor

export ∂⃖, gradient

include("runtime.jl")
include("interface.jl")
include("extra_rules.jl")
include("utils.jl")
include("taylor.jl")

include("stage1/generated.jl")
include("stage1/termination.jl")

include("debugutils.jl")

end
