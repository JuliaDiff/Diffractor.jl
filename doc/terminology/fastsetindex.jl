struct FastSetIndex <: AbstractOptic
    path::NTuple{N, Int} where N
end
FastSetIndex(i::Int) = FastSetIndex((i,))

path_getindex(obj, ::Tuple{}) = obj
path_getindex(obj, path::Tuple) =
    path_getindex(getindex(obj, first(path)), tail(path))
left(f::FastSetIndex, obj) =
    (obj, path_getindex(obj, f.path))

path_setindex(obj, update, (idx,)::Tuple{Int}) =
    setindex(obj, update, idx)
path_setindex(obj, update, path) =
    setindex(obj,
        path_setindex(getindex(obj, first(path)),
        update, tail(path)),
        first(path))
function right(f::FastSetIndex, obj, update)
    path_setindex(obj, update, f.path)
end

(a::FastSetIndex ⨟ b::FastSetIndex) =
    FastSetIndex((a.path..., b.path...))
