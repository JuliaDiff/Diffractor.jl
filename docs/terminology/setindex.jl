SetIndex(idx) = Optic(
    obj->(obj, getindex(obj, idx)),
    (obj, update)->setindex(obj, update, idx)
)
