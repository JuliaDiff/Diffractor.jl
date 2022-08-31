using ChainRulesCore
struct DiffractorRuleConfig <: RuleConfig{Union{HasReverseMode,HasForwardsMode}} end

@Base.constprop :aggressive accum(a, b) = a + b
@Base.constprop :aggressive accum(a::Tuple, b::Tuple) = map(accum, a, b)
@Base.constprop :aggressive @generated function accum(x::NamedTuple, y::NamedTuple)
    fnames = union(fieldnames(x), fieldnames(y))
    gradx(f) = f in fieldnames(x) ? :(getfield(x, $(quot(f)))) : :(ZeroTangent())
    grady(f) = f in fieldnames(y) ? :(getfield(y, $(quot(f)))) : :(ZeroTangent())
    Expr(:tuple, [:($f=accum($(gradx(f)), $(grady(f)))) for f in fnames]...)
end
@Base.constprop :aggressive accum(a, b, c, args...) = accum(accum(a, b), c, args...)
@Base.constprop :aggressive accum(a::NoTangent, b) = b
@Base.constprop :aggressive accum(a, b::NoTangent) = a
@Base.constprop :aggressive accum(a::NoTangent, b::NoTangent) = NoTangent()

using ChainRulesCore: Tangent, backing

function accum(x::Tangent{T}, y::NamedTuple) where T
  # @warn "gradient is both a Tangent and a NamedTuple" x y
  z = accum(backing(x), y)
  Tangent{T,typeof(z)}(z)
end
accum(x::NamedTuple, y::Tangent) = accum(y, x)

function accum(x::Tangent{T}, y::Tangent) where T
  z = accum(backing(x), backing(y))
  Tangent{T,typeof(z)}(z)
end
