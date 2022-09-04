using ChainRulesCore
struct DiffractorRuleConfig <: RuleConfig{Union{HasReverseMode,HasForwardsMode}} end

@Base.constprop :aggressive accum(a, b) = a + b
@Base.constprop :aggressive accum(a::Tuple, b::Tuple) = map(accum, a, b)
@Base.constprop :aggressive @generated function accum(x::NamedTuple, y::NamedTuple)
    fnames = union(fieldnames(x), fieldnames(y))
    isempty(fnames) && return :((;))  # code below makes () instead
    gradx(f) = f in fieldnames(x) ? :(getfield(x, $(quot(f)))) : :(ZeroTangent())
    grady(f) = f in fieldnames(y) ? :(getfield(y, $(quot(f)))) : :(ZeroTangent())
    Expr(:tuple, [:($f=accum($(gradx(f)), $(grady(f)))) for f in fnames]...)
end
@Base.constprop :aggressive accum(a, b, c, args...) = accum(accum(a, b), c, args...)
@Base.constprop :aggressive accum(a::AbstractZero, b) = b
@Base.constprop :aggressive accum(a, b::AbstractZero) = a
@Base.constprop :aggressive accum(a::AbstractZero, b::AbstractZero) = NoTangent()

using ChainRulesCore: Tangent, backing

function accum(x::Tangent{T}, y::NamedTuple) where T
  # @warn "gradient is both a Tangent and a NamedTuple" x y
  _tangent(T, accum(backing(x), y))
end
accum(x::NamedTuple, y::Tangent) = accum(y, x)
# This solves an ambiguity, but also avoids Tangent{ZeroTangent}() which + does not:
accum(x::Tangent{T}, y::Tangent) where T = _tangent(T, accum(backing(x), backing(y)))

_tangent(::Type{T}, z) where T = Tangent{T,typeof(z)}(z)
_tangent(::Type, ::NamedTuple{()}) = NoTangent()
