using ChainRulesCore

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
