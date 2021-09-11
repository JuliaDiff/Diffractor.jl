abstract type AbstractOptic; end

struct OpticRepr <: AbstractOptic
  l::Function
  r::Function
end
OpticRepr(o::OpticRepr) = o
OpticRepr(o::AbstractOptic) =
  OpticRepr(a->left(o, a),
    (m, b′)->right(o, m, b′))

left(o::OpticRepr, a) = o.l(a)
right(o::OpticRepr, m, b′) = o.r(m, b′)

function ⨟(o₁::AbstractOptic, o₂::AbstractOptic)
  OpticRepr(
    function (a)
      x = left(o₁, a)
      y = left(o₂, x[2])
      (x[1], y[1]), y[2]
    end,
    ((m₁, m₂), c′)->right(o₁, m₁, right(o₂, m₂, c′))
  )
end

