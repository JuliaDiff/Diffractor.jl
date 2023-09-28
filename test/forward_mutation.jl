using Diffractor
using Diffractor: ∂☆, ZeroBundle, TaylorBundle
using Diffractor: bundle

mutable struct MDemo1
    x::Float64
end

∂☆{1}()(ZeroBundle{1}(MDemo1), TaylorBundle{1}(1.5, (1.0,)))

function double!(val::MDemo1)
    val.x *= 2.0
    return val
end
function wrap_and_double(x)
    val = MDemo1(x)
    double!(val)
end
∂☆{1}()(ZeroBundle{1}(wrap_and_double), TaylorBundle{1}(1.5, (1.0,)))
