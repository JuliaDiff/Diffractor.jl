using Diffractor
using Diffractor: var"'"
using ChainRules
using ChainRules: Zero

using Test

# Unit tests
function tup2(f)
    a, b = ∂⃖{2}()(f, 1)
    c, d = b((2,))
    e, f = d(Zero(), 3)
    f((4,))
end

@test tup2(tuple) == (Zero(), 4)

my_tuple(args...) = args
ChainRules.rrule(::typeof(my_tuple), args...) = args, Δ->Core.tuple(NO_FIELDS, Δ...)
@test tup2(my_tuple) == (Zero(), 4)

# Integration tests
@test @inferred(sin'(1.0)) == cos(1.0)
@test @inferred(sin''(1.0)) == -sin(1.0)
@test sin'''(1.0) == -cos(1.0)

f_getfield(x) = getfield((x,), 1)
@test f_getfield'(1) == 1
@test f_getfield''(1) == 0
@test f_getfield'''(1) == 0
