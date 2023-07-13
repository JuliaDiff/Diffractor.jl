module rules

using Test, Diffractor, ChainRulesCore
using Diffractor: var"'"

# invalidation for rrule
rrule_demo(x) = sin(x)
function rrule_demo_pullback(x)
    return function (Δx)
        return NoTangent(), Δx*cos(x)
    end
end
function ChainRulesCore.rrule(::typeof(rrule_demo), x)
    return rrule_demo(x), rrule_demo_pullback(x)
end
@test cos(42) == rrule_demo'(42)
function rrule_demo_pullback(x)
    return function (Δx)
        return NoTangent(), Δx*sin(x)
    end
end
@test sin(42) == rrule_demo'(42)

# invalidation for frule
frule_demo(x) = sin(x)
function frule_demo_impl(Δx, x)
    sinx, cosx = sincos(x)
    return (sinx, sinx * Δx)
end
function ChainRulesCore.frule((_, Δx), ::typeof(frule_demo), x)
    return frule_demo_impl(Δx, x)
end
@test cos(42) == frule_demo'(42)
function frule_demo_impl(Δx, x)
    sinx, cosx = sincos(x)
    return (sinx, sinx * Δx)
end
@test sin(42) == frule_demo'(42)

end # module rules
