using ChainRulesCore

@Base.aggressive_constprop accum(a, b) = a + b
@Base.aggressive_constprop accum(a::Tuple, b::Tuple) = map(accum, a, b)
@Base.aggressive_constprop @generated function accum(x::NamedTuple, y::NamedTuple)
    fnames = union(fieldnames(x), fieldnames(y))
    gradx(f) = f in fieldnames(x) ? :(getfield(x, $(quot(f)))) : :(Zero())
    grady(f) = f in fieldnames(y) ? :(getfield(y, $(quot(f)))) : :(Zero())
    Expr(:tuple, [:($f=accum($(gradx(f)), $(grady(f)))) for f in fnames]...)
end
@Base.aggressive_constprop accum(a, b, c, args...) = accum(accum(a, b), c, args...)
@Base.aggressive_constprop accum(a::DoesNotExist, b) = b
@Base.aggressive_constprop accum(a, b::DoesNotExist) = a
@Base.aggressive_constprop accum(a::DoesNotExist, b::DoesNotExist) = DoesNotExist()
