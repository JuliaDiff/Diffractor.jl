module Diffractor

#macro show(exs...)
#    blk = Expr(:block)
#    push!(blk.args, quote
#        ccall(:jl_safe_printf, Cvoid, (Cstring,), $("$__source__\n"))
#    end)
#    for ex in exs
#        push!(blk.args, quote
#            let s = string($(sprint(Base.show_unquoted,ex)*" = "),
#                                  repr(begin local value = $(esc(ex)) end), "\n")
#                ccall(:jl_safe_printf, Cvoid, (Cstring,), s)
#            end
#        end)
#    end
#    isempty(exs) || push!(blk.args, :value)
#    return blk
#end

using StructArrays

export ∂⃖, gradient

include("runtime.jl")
include("interface.jl")
include("utils.jl")
include("tangent.jl")
include("jet.jl")

include("stage1/generated.jl")
include("stage1/termination.jl")
include("stage1/forward.jl")
include("stage1/recurse_fwd.jl")
include("stage1/mixed.jl")
include("stage1/broadcast.jl")

include("extra_rules.jl")

include("higher_fwd_rules.jl")

include("debugutils.jl")

end
