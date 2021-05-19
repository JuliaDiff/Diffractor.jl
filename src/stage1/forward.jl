partial(x::TangentBundle, i) = x.partials[i]
partial(x::ZeroBundle, i) = ZeroTangent()
partial(x::ZeroTangent, i) = ZeroTangent()
primal(x::AbstractTangentBundle) = x.primal
primal(z::ZeroTangent) = ZeroTangent()

# TODO: Which version do we want in ChainRules?
function my_frule(args...)
    frule(map(x->partial(x, 1), args), map(primal, args)...)
end

function (∂⃗p::∂⃗{N})(::ZeroBundle{N, typeof(my_frule)}, args::AbstractTangentBundle{N}...) where {N}
    @show args
    error("TODO")

    ∂⃗p(ZeroBundle{N}(frule), )
end

function (∂⃗p::∂⃗{1})(::ZeroBundle{1, typeof(my_frule)}, args::AbstractTangentBundle{1}...)
    ∂⃗p(ZeroBundle{1}(frule),
        TangentBundle{1}(map(args) do arg
            primal(partial(arg, 1))
        end, (map(args) do arg
            partial(partial(arg, 1), 1)
        end,)),
        map(primal, args)...)
end


shuffle_down(b::ZeroBundle{N, B}) where {N, B} =
    ZeroBundle{minus1(N)}(ZeroBundle{1, B}(b.primal))

function shuffle_down(b::TangentBundle{2, B}) where {B}
    TangentBundle{1}(
        TangentBundle{1}(b.primal, (partial(b, 1),)),
        (TangentBundle{1}(partial(b, 2), (partial(b, 3),)),))
end

function shuffle_up(r)
    @show r
    error()
end

function (::∂⃗{N})(args::AbstractTangentBundle{N}...) where {T, N}
    # N = 1 case manually inlined to avoid ambiguities
    if N === 1
        r = my_frule(args...)
        if r === nothing
            return ∂⃗recurse{1}()(args...)
        else
            return TangentBundle{1}(r[1], (r[2],))
        end
    else
        ∂⃗p = ∂⃗{minus1(N)}()
        r = ∂⃗p(ZeroBundle{minus1(N)}(my_frule), map(shuffle_down, args)...)
        if primal(r) === nothing
            return ∂⃖recurse{N}()(f, args...)
        else
            return TangentBundle{N}(primal(r)[1], (primal(r)[2], r.partials[1]...))
        end
    end
end

# Special case rules for performance
function (::∂⃗{N})(f::ZeroBundle{N, typeof(getfield)}, x::AbstractTangentBundle{N}, s::AbstractTangentBundle{N}) where {N}
    s = primal(s)
    TangentBundle{N}(getfield(primal(x), s),
        map(x->getfield(ChainRulesCore.backing(x), s), x.partials))
end

function (::∂⃗{N})(f::ZeroBundle{N, typeof(tuple)}, args::AbstractTangentBundle{N}...) where {N}
    TangentBundle{N}(map(primal, args),
        ntuple(2^N-1) do i
            map(x->x.partials[i], args)
        end)
end
