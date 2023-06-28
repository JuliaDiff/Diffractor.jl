partial(x::TangentBundle, i) = partial(getfield(x, :tangent), i)
partial(x::ExplicitTangent, i) = getfield(getfield(x, :partials), i)
partial(x::TaylorTangent, i) = getfield(getfield(x, :coeffs), i)
partial(x::UniformTangent, i) = getfield(x, :val)
partial(x::ProductTangent, i) = ProductTangent(map(x->partial(x, i), getfield(x, :factors)))
partial(x::AbstractZero, i) = x
partial(x::CompositeBundle{N, B}, i) where {N, B<:Tuple} = Tangent{B}(map(x->partial(x, i), getfield(x, :tup))...)
function partial(x::CompositeBundle{N, B}, i) where {N, B}
    # This is tangent for a struct, but fields partials are each stored in a plain tuple
    # so we add the names back using the primal `B`
    # TODO: If required this can be done as a `@generated` function so it is type-stable
    backing = NamedTuple{fieldnames(B)}(map(x->partial(x, i), getfield(x, :tup)))
    return Tangent{B, typeof(backing)}(backing)
end


primal(x::AbstractTangentBundle) = x.primal
primal(z::ZeroTangent) = ZeroTangent()

first_partial(x) = partial(x, 1)

shuffle_down(b::UniformBundle{N, B, U}) where {N, B, U} =
    UniformBundle{N-1, <:Any}(UniformBundle{1, B}(b.primal, b.tangent.val),
                                    UniformBundle{1, U}(b.tangent.val, b.tangent.val))

function shuffle_down(b::ExplicitTangentBundle{N, B}) where {N, B}
    # N.B: This depends on the special properties of the canonical tangent index order
    Base.@constprop :aggressive function _sdown(i::Int64)
        ExplicitTangentBundle{1}(partial(b, 2*i), (partial(b, 2*i+1),))
    end
    ExplicitTangentBundle{N-1}(
        ExplicitTangentBundle{1}(b.primal, (partial(b, 1),)),
        ntuple(_sdown, 1<<(N-1)-1))
end

function shuffle_down(b::TaylorBundle{N, B}) where {N, B}
    Base.@constprop :aggressive function _sdown(i::Int64)
        ExplicitTangentBundle{1}(b.tangent.coeffs[i], (b.tangent.coeffs[i+1],))
    end
    TaylorBundle{N-1}(
        ExplicitTangentBundle{1}(b.primal, (b.tangent.coeffs[1],)),
        ntuple(_sdown, N-1))
end

function shuffle_down(b::CompositeBundle{N, B}) where {N, B}
    z = CompositeBundle{N-1, CompositeBundle{1, B}}(
        (CompositeBundle{N-1, Tuple}(
            map(shuffle_down, b.tup)
        ),)
    )
    z
end

function shuffle_up(r::CompositeBundle{1})
    z₀ = primal(r.tup[1])
    z₁ = partial(r.tup[1], 1)
    z₂ = primal(r.tup[2])
    z₁₂ = partial(r.tup[2], 1)
    if z₁ == z₂
        return TaylorBundle{2}(z₀, (z₁, z₁₂))
    else
        return ExplicitTangentBundle{2}(z₀, (z₁, z₂, z₁₂))
    end
end

function taylor_compatible(a::ATB{N}, b::ATB{N}) where {N}
    primal(b) === a[TaylorTangentIndex(1)] || return false
    return all(1:(N-1)) do i
        b[TaylorTangentIndex(i)] === a[TaylorTangentIndex(i+1)]
    end
end

# Check whether the tangent bundle element is taylor-like
isswifty(::TaylorBundle) = true
isswifty(::UniformBundle) = true
isswifty(b::CompositeBundle) = all(isswifty, b.tup)
isswifty(::Any) = false

function shuffle_up(r::CompositeBundle{N}) where {N}
    a, b = r.tup
    if isswifty(a) && isswifty(b) && taylor_compatible(a, b)
        return TaylorBundle{N+1}(primal(a),
            ntuple(i->i == N+1 ?
                b[TaylorTangentIndex(i-1)] : a[TaylorTangentIndex(i)],
            N+1))
    else
        return TangentBundle{N+1}(r.tup[1].primal,
            (r.tup[1].tangent.partials..., primal(b),
            ntuple(i->partial(b,i), 1<<(N+1)-1)...))
    end
end

function shuffle_up(r::UniformBundle{N, B, U}) where {N, B, U}
    (a, b) = primal(r)
    if r.tangent.val === b
        u = b
    elseif b == NoTangent() && U === ZeroTangent
        u = b
    else
        error()
    end
    UniformBundle{N+1}(a, u)
end
@ChainRulesCore.non_differentiable shuffle_up(r::UniformBundle)

struct ∂☆internal{N}; end
struct ∂☆shuffle{N}; end

function shuffle_base(r)
    (primal, dual) = r
    if isa(dual, Union{NoTangent, ZeroTangent})
        UniformBundle{1}(primal, dual)
    else
        TaylorBundle{1}(primal, (dual,))
    end
end

function (::∂☆internal{1})(args::AbstractTangentBundle{1}...)
    r = frule(#=DiffractorRuleConfig(),=# map(first_partial, args), map(primal, args)...)
    if r === nothing
        return ∂☆recurse{1}()(args...)
    else
        return shuffle_base(r)
    end
end

function ChainRulesCore.frule_via_ad(::DiffractorRuleConfig, partials, args...)
    bundles = map((p,a) -> ExplicitTangentBundle{1}(a, (p,)), partials, args)
    result = ∂☆internal{1}()(bundles...)
    primal(result), first_partial(result)
end

function (::∂☆shuffle{N})(args::AbstractTangentBundle{N}...) where {N}
    ∂☆p = ∂☆{N-1}()
    downargs = map(shuffle_down, args)
    tupargs = ∂vararg{N-1}()(map(first_partial, downargs)...)
    ∂☆p(ZeroBundle{N-1}(frule), #= ZeroBundle{N-1}(DiffractorRuleConfig()), =# tupargs, map(primal, downargs)...)
end

# Special shortcut case if there is no derivative information at all:
function (::∂☆internal{N})(f::AbstractZeroBundle{N}, args::AbstractZeroBundle{N}...) where {N}
    f_v = primal(f)
    args_v = map(primal, args)
    return ZeroBundle{N}(f_v(args_v...))
end
function (::∂☆internal{1})(f::AbstractZeroBundle{1}, args::AbstractZeroBundle{1}...)
    f_v = primal(f)
    args_v = map(primal, args)
    return ZeroBundle{1}(f_v(args_v...))
end

function (::∂☆internal{N})(args::AbstractTangentBundle{N}...) where {N}
    r = ∂☆shuffle{N}()(args...)
    if primal(r) === nothing
        return ∂☆recurse{N}()(args...)
    else
        return shuffle_up(r)
    end
end
(::∂☆{N})(args::AbstractTangentBundle{N}...) where {N} = ∂☆internal{N}()(args...)


# Special case rules for performance
@Base.constprop :aggressive function (::∂☆{N})(f::ATB{N, typeof(getfield)}, x::TangentBundle{N}, s::AbstractTangentBundle{N}) where {N}
    s = primal(s)
    ExplicitTangentBundle{N}(getfield(primal(x), s),
        map(x->lifted_getfield(x, s), x.tangent.partials))
end

@Base.constprop :aggressive function (::∂☆{N})(f::ATB{N, typeof(getfield)}, x::TangentBundle{N}, s::ATB{N}, inbounds::ATB{N}) where {N}
    s = primal(s)
    ExplicitTangentBundle{N}(getfield(primal(x), s, primal(inbounds)),
        map(x->lifted_getfield(x, s), x.tangent.partials))
end

@Base.constprop :aggressive function (::∂☆{N})(f::ATB{N, typeof(getfield)}, x::TaylorBundle{N}, s::AbstractTangentBundle{N}) where {N}
    s = primal(s)
    TaylorBundle{N}(getfield(primal(x), s),
        map(y->lifted_getfield(y, s), x.tangent.coeffs))
end

@Base.constprop :aggressive function (::∂☆{N})(f::ATB{N, typeof(getfield)}, x::TaylorBundle{N}, s::AbstractTangentBundle{N}, inbounds::ATB{N}) where {N}
    s = primal(s)
    TaylorBundle{N}(getfield(primal(x), s, primal(inbounds)),
        map(y->lifted_getfield(y, s), x.tangent.coeffs))
end

@Base.constprop :aggressive function (::∂☆{N})(::ATB{N, typeof(getfield)}, x::CompositeBundle{N}, s::AbstractTangentBundle{N, Int}) where {N}
    x.tup[primal(s)]
end

@Base.constprop :aggressive function (::∂☆{N})(::ATB{N, typeof(getfield)}, x::CompositeBundle{N, B}, s::AbstractTangentBundle{N, Symbol}) where {N, B}
    x.tup[Base.fieldindex(B, primal(s))]
end

@Base.constprop :aggressive function (::∂☆{N})(f::ATB{N, typeof(getfield)}, x::UniformBundle{N, <:Any, U}, s::AbstractTangentBundle{N}) where {N, U}
    UniformBundle{N,<:Any,U}(getfield(primal(x), primal(s)), x.tangent.val)
end

@Base.constprop :aggressive function (::∂☆{N})(f::ATB{N, typeof(getfield)}, x::UniformBundle{N, <:Any, U}, s::AbstractTangentBundle{N}, inbounds::AbstractTangentBundle{N}) where {N, U}
    UniformBundle{N,<:Any,U}(getfield(primal(x), primal(s), primal(inbounds)), x.tangent.val)
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

function (::∂☆{N})(::ZeroBundle{N, typeof(map)}, f::ATB{N}, args::ATB{N, <:AbstractArray}...) where {N}
    # TODO: This could do an inplace map! to avoid the extra rebundling
    rebundle(map(FwdMap(f), map(unbundle, args)...))
end

function (::∂☆{N})(::ZeroBundle{N, typeof(map)}, f::ATB{N}, args::ATB{N}...) where {N}
    ∂☆recurse{N}()(ZeroBundle{N, typeof(map)}(map), f, args...)
end


function (::∂☆{N})(f::ZeroBundle{N, typeof(ifelse)}, arg::ATB{N, Bool}, args::ATB{N}...) where {N}
    ifelse(arg.primal, args...)
end

function (::∂☆{N})(f::ZeroBundle{N, typeof(Core.ifelse)}, arg::ATB{N, Bool}, args::ATB{N}...) where {N}
    Core.ifelse(arg.primal, args...)
end

struct FwdIterate{N, T<:AbstractTangentBundle{N}}
    f::T
end
function (f::FwdIterate)(arg::ATB{N}) where {N}
    r = ∂☆{N}()(f.f, arg)
    # `primal(r) === nothing` would work, but doesn't create `Conditional` in inference
    isa(r, ATB{N, Nothing}) && return nothing
    (∂☆{N}()(ZeroBundle{N}(getindex), r, ZeroBundle{N}(1)),
     primal(∂☆{N}()(ZeroBundle{N}(getindex), r, ZeroBundle{N}(2))))
end
@Base.constprop :aggressive function (f::FwdIterate)(arg::ATB{N}, st) where {N}
    r = ∂☆{N}()(f.f, arg, ZeroBundle{N}(st))
    # `primal(r) === nothing` would work, but doesn't create `Conditional` in inference
    isa(r, ATB{N, Nothing}) && return nothing
    (∂☆{N}()(ZeroBundle{N}(getindex), r, ZeroBundle{N}(1)),
     primal(∂☆{N}()(ZeroBundle{N}(getindex), r, ZeroBundle{N}(2))))
end

function (this::∂☆{N})(::ZeroBundle{N, typeof(Core._apply_iterate)}, iterate::ATB{N}, f::ATB{N}, args::ATB{N}...) where {N}
    Core._apply_iterate(FwdIterate(iterate), this, (f,), args...)
end

function (this::∂☆{N})(::ZeroBundle{N, typeof(iterate)}, t::CompositeBundle{N, <:Tuple}) where {N}
    r = iterate(t.tup)
    r === nothing && return ZeroBundle{N}(nothing)
    ∂vararg{N}()(r[1], ZeroBundle{N}(r[2]))
end

function (this::∂☆{N})(::ZeroBundle{N, typeof(iterate)}, t::CompositeBundle{N, <:Tuple}, a::ATB{N}, args::ATB{N}...) where {N}
    r = iterate(t.tup, primal(a), map(primal, args)...)
    r === nothing && return ZeroBundle{N}(nothing)
    ∂vararg{N}()(r[1], ZeroBundle{N}(r[2]))
end

function (this::∂☆{N})(::ZeroBundle{N, typeof(Base.indexed_iterate)}, t::CompositeBundle{N, <:Tuple}, i::ATB{N}) where {N}
    r = Base.indexed_iterate(t.tup, primal(i))
    ∂vararg{N}()(r[1], ZeroBundle{N}(r[2]))
end

function (this::∂☆{N})(::ZeroBundle{N, typeof(Base.indexed_iterate)}, t::CompositeBundle{N, <:Tuple}, i::ATB{N}, st1::ATB{N}, st::ATB{N}...) where {N}
    r = Base.indexed_iterate(t.tup, primal(i), primal(st1), map(primal, st)...)
    ∂vararg{N}()(r[1], ZeroBundle{N}(r[2]))
end

function (this::∂☆{N})(::ZeroBundle{N, typeof(Base.indexed_iterate)}, t::TangentBundle{N, <:Tuple}, i::ATB{N}, st::ATB{N}...) where {N}
    ∂vararg{N}()(this(ZeroBundle{N}(getfield), t, i), ZeroBundle{N}(primal(i) + 1))
end


function (this::∂☆{N})(::ZeroBundle{N, typeof(getindex)}, t::CompositeBundle{N, <:Tuple}, i::ZeroBundle) where {N}
    t.tup[primal(i)]
end

function (this::∂☆{N})(::ZeroBundle{N, typeof(typeof)}, x::ATB{N}) where {N}
    DNEBundle{N}(typeof(primal(x)))
end

function (this::∂☆{N})(f::ZeroBundle{N, Core.IntrinsicFunction}, args::ATB{N}...) where {N}
    ff = primal(f)
    if ff === Base.not_int
        DNEBundle{N}(ff(map(primal, args)...))
    else
        error("Missing rule for intrinsic function $ff")
    end
end
