using Base: setindex

include("opticdef.jl")

function (o::AbstractOptic)(f::Function, a)
    m, b = left(o, a)
    b′ = f(b)
    right(o, m, b′)
end

function (o::AbstractOptic)(a)
    m, b = left(o, a)
    b, b′->right(o, m, b′)
end
