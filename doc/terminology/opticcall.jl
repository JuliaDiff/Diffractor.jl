# Block style optics interface
function (o::AbstractOptic)(f::Function, a)
    m, b = left(o, a)
    b′ = f(b)
    right(o, m, b′)
end

# Continuation style optics interface
function (o::AbstractOptic)(a)
    m, b = left(o, a)
    b, b′->right(o, m, b′)
end
