partial(x::TangentBundle, i) = x.partials[i]
partial(x::TaylorBundle{1}, i) = x.coeffs[i]
partial(x::ZeroBundle, i) = ZeroTangent()
partial(x::CompositeBundle{N, B}, i) where {N, B} = Tangent{B}(map(x->partial(x, i), x.tup))
partial(x::ZeroTangent, i) = ZeroTangent()

primal(x::AbstractTangentBundle) = x.primal
primal(z::ZeroTangent) = ZeroTangent()

first_partial(x::TangentBundle{1}) = getfield(getfield(x, :partials), 1)
first_partial(x::TaylorBundle{1}) = getfield(getfield(x, :coeffs), 1)
first_partial(x::ZeroBundle) = Zero()
first_partial(x::CompositeBundle) = map(first_partial, x.tup)

# TODO: Which version do we want in ChainRules?
function my_frule(args::ATB{1}...)
    frule(map(first_partial, args), map(primal, args)...)
end

# Fast path for some hot cases
my_frule(::ZeroBundle{1, typeof(frule)}, args::ATB{1}...) = nothing
my_frule(::ZeroBundle{1, typeof(my_frule)}, args::ATB{1}...) = nothing

(::∂☆{N})(::ZeroBundle{N, typeof(my_frule)}, ::ZeroBundle{N, ZeroBundle{1, typeof(frule)}}, args::ATB{N}...) where {N} = ZeroBundle{N}(nothing)
(::∂☆{N})(::ZeroBundle{N, typeof(my_frule)}, ::ZeroBundle{N, ZeroBundle{1, typeof(my_frule)}}, args::ATB{N}...) where {N} = ZeroBundle{N}(nothing)

#=
function (∂☆p::∂☆{1})(::ZeroBundle{1, typeof(my_frule)}, args::AbstractTangentBundle{1}...)
    ∂☆p(ZeroBundle{1}(frule),
        TangentBundle{1}(map(args) do arg
            primal(partial(arg, 1))
        end, (map(args) do arg
            partial(partial(arg, 1), 1)
        end,)),
        map(primal, args)...)
end
=#

shuffle_down(b::ZeroBundle{N, B}) where {N, B} =
    ZeroBundle{minus1(N)}(ZeroBundle{1, B}(b.primal))

function shuffle_down(b::TangentBundle{2, B}) where {B}
    TangentBundle{1}(
        TangentBundle{1}(b.primal, (partial(b, 1),)),
        (TangentBundle{1}(partial(b, 2), (partial(b, 3),)),))
end

function shuffle_down(b::TaylorBundle{2, B}) where {B}
    TangentBundle{1}(
        TaylorBundle{1}(b.primal, (b.coeffs[1],)),
        (TaylorBundle{1}(b.coeffs[1], (b.coeffs[2],)),))
end

function shuffle_down(b::TaylorBundle{3, B}) where {B}
    TaylorBundle{2}(
        TangentBundle{1}(b.primal, (b.coeffs[1],)),
        (TangentBundle{1}(b.coeffs[1], (b.coeffs[2],)),
        TangentBundle{1}(b.coeffs[2], (b.coeffs[3],))
        ))
end

function shuffle_down(b::CompositeBundle{N, B}) where {N, B}
    CompositeBundle{N-1, B}(
        map(shuffle_down, b.tup)
    )
end

function shuffle_up(r::CompositeBundle{1})
    TangentBundle{2}(primal(r.tup[1]), (partial(r.tup[1], 1), primal(r.tup[2]), partial(r.tup[2], 1)))
end

function shuffle_up(r::CompositeBundle{2})
    TangentBundle{3}(r.tup[1].primal,
        (r.tup[1].partials..., r.tup[2].primal, r.tup[2].partials...))
end

function (::∂☆{N})(args::AbstractTangentBundle{N}...) where {N}
    # N = 1 case manually inlined to avoid ambiguities
    if N === 1
        r = my_frule(args...)
        if r === nothing
            return ∂☆recurse{1}()(args...)
        else
            return TangentBundle{1}(r[1], (r[2],))
        end
    else
        ∂☆p = ∂☆{minus1(N)}()
        r = ∂☆p(ZeroBundle{minus1(N)}(my_frule), map(shuffle_down, args)...)
        if primal(r) === nothing
            return ∂☆recurse{N}()(args...)
        else
            return shuffle_up(r)
        end
    end
end

function (::∂☆{N})(args::ZeroBundle{N}...) where {N}
    ZeroBundle{N}(primal(getfield(args, 1))(map(primal, Base.tail(args))...))
end

# Special case rules for performance
@Base.aggressive_constprop function (::∂☆{N})(f::ATB{N, typeof(getfield)}, x::TangentBundle{N}, s::AbstractTangentBundle{N}) where {N}
    s = primal(s)
    TangentBundle{N}(getfield(primal(x), s),
        map(x->lifted_getfield(ChainRulesCore.backing(x), s), x.partials))
end

@Base.aggressive_constprop function (::∂☆{N})(f::ATB{N, typeof(getfield)}, x::TaylorBundle{N}, s::AbstractTangentBundle{N}) where {N}
    s = primal(s)
    TaylorBundle{N}(getfield(primal(x), s),
        map(y->lifted_getfield(y, s), x.coeffs))
end

@Base.aggressive_constprop function (::∂☆{N})(::ATB{N, typeof(getfield)}, x::CompositeBundle{N}, s::AbstractTangentBundle{N, Int}) where {N}
    x.tup[primal(s)]
end

@Base.aggressive_constprop function (::∂☆{N})(::ATB{N, typeof(getfield)}, x::CompositeBundle{N, B}, s::AbstractTangentBundle{N, Symbol}) where {N, B}
    x.tup[fieldindex(B, s)]
end

@Base.aggressive_constprop function (::∂☆{N})(f::ATB{N, typeof(getfield)}, x::ATB{N}, s::ATB{N}, inbounds::ATB{N}) where {N}
    s = primal(s)
    TangentBundle{N}(getfield(primal(x), s, primal(inbounds)),
        map(x->getfield(ChainRulesCore.backing(x), s), x.partials))
end

@Base.aggressive_constprop function (::∂☆{N})(f::ATB{N, typeof(getfield)}, x::ZeroBundle{N}, s::AbstractTangentBundle{N}) where {N}
    ZeroBundle{N}(getfield(primal(x), primal(s)))
end

@Base.aggressive_constprop function (::∂☆{N})(f::ATB{N, typeof(getfield)}, x::ZeroBundle{N}, s::AbstractTangentBundle{N}, inbounds::AbstractTangentBundle{N}) where {N}
    ZeroBundle{N}(getfield(primal(x), primal(s), primal(inbounds)))
end

function (::∂☆{N})(f::ATB{N, typeof(tuple)}, args::AbstractTangentBundle{N}...) where {N}
    ∂vararg{N}()(args...)
end

struct FwdMap{N, T<:AbstractTangentBundle{N}}
    f::T
end
(f::FwdMap{N})(args::AbstractTangentBundle{N}...) where {N} = ∂☆{N}()(f.f, args...)

function (::∂☆{N})(::ZeroBundle{N, typeof(map)}, f::ATB{N}, tup::CompositeBundle{N, <:Tuple}) where {N}
    ∂vararg{N}()(map(FwdMap(f), tup.tup)...)
end

function (::∂☆{N})(f::ZeroBundle{N, typeof(ifelse)}, arg::ATB{N, Bool}, args::ATB{N}...) where {N}
    ifelse(arg.primal, args...)
end

struct FwdIterate{N, T<:AbstractTangentBundle{N}}
    f::T
end
function (f::FwdIterate)(arg::ATB{N}) where {N}
    r = ∂☆{N}()(f.f, arg)
    primal(r) === nothing && return nothing
    (∂☆{N}()(ZeroBundle{N}(getindex), r, ZeroBundle{N}(1)),
     primal(∂☆{N}()(ZeroBundle{N}(getindex), r, ZeroBundle{N}(2))))
end
function (f::FwdIterate)(arg::ATB{N}, st) where {N}
    r = ∂☆{N}()(f.f, arg, ZeroBundle{N}(st))
    primal(r) === nothing && return nothing
    (∂☆{N}()(ZeroBundle{N}(getindex), r, ZeroBundle{N}(1)),
     primal(∂☆{N}()(ZeroBundle{N}(getindex), r, ZeroBundle{N}(2))))
end

function (this::∂☆{N})(::ZeroBundle{N, typeof(Core._apply_iterate)}, iterate::ATB{N}, f::ATB{N}, args::ATB{N}...) where {N}
    Core._apply_iterate(FwdIterate(iterate), this, (f,), args...)
end

function (this::∂☆{N})(::ZeroBundle{N, typeof(iterate)}, t::CompositeBundle{N, <:Tuple}, args::ATB{N}...) where {N}
    r = iterate(t.tup, (args === () ? () : map(primal, args))...)
    r === nothing && return ZeroBundle{N}(nothing)
    ∂vararg{N}()(r[1], ZeroBundle{N}(r[2]))
end

function (this::∂☆{N})(::ZeroBundle{N, typeof(Base.indexed_iterate)}, t::CompositeBundle{N, <:Tuple}, i::ATB{N}, st::ATB{N}...) where {N}
    r = Base.indexed_iterate(t.tup, primal(i), (st === () ? () : map(primal, st))...)
    ∂vararg{N}()(r[1], ZeroBundle{N}(r[2]))
end

function (this::∂☆{N})(::ZeroBundle{N, typeof(Base.indexed_iterate)}, t::TangentBundle{N, <:Tuple}, i::ATB{N}, st::ATB{N}...) where {N}
    ∂vararg{N}()(this(ZeroBundle{N}(getfield), t, i), ZeroBundle{N}(primal(i) + 1))
end


function (this::∂☆{N})(::ZeroBundle{N, typeof(getindex)}, t::CompositeBundle{N, <:Tuple}, i::ZeroBundle) where {N}
    t.tup[primal(i)]
end
