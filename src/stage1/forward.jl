partial(x::TangentBundle, i) = partial(getfield(x, :tangent), i)
partial(x::ExplicitTangent, i) = getfield(getfield(x, :partials), i)
partial(x::TaylorTangent, i) = getfield(getfield(x, :coeffs), i)
partial(x::UniformTangent, i) = getfield(x, :val)
partial(x::AbstractZero, i) = x


primal(x::AbstractTangentBundle) = x.primal
primal(z::ZeroTangent) = ZeroTangent()

first_partial(x) = partial(x, 1)

shuffle_down(b::UniformBundle{N, B, U}) where {N, B, U} =
    UniformBundle{N-1}(UniformBundle{1, B}(b.primal, b.tangent.val),
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
        TaylorBundle{1}(b.tangent.coeffs[i], (b.tangent.coeffs[i+1],))
    end
    TaylorBundle{N-1}(
        TaylorBundle{1}(b.primal, (b.tangent.coeffs[1],)),
        ntuple(_sdown, N-1))
end

struct TaylorRequired
    order
    z₁
    z₂
end
function Base.showerror(io::IO, err)
    order_str1 = order_str(err.order)
    print(io, "In Eras mode all higher order derivatives must be taylor, but encountered one where the taylor requirement z₁ == z₂ was not met.")
    println(is, "derivative on $order_str1 path: z₁ = ", err.z₁)
    println(is, "$order_str1 on the derivative path: z₂ = ", err.z₂)
end

function order_str(order::Integer)
    @assert order>=0
    if order == 0
        "primal"
    elseif order == 1
        "derivative"
    elseif order == 2
        "2nd derivative"
    elseif order == 3
        "3rd derivative"
    else
        "$(order)th derivative"
    end
end

"finds the lowerest order derivative that is not taylor compatible, or returns -1 if all compatible"
@noinline function find_taylor_incompatibility(r::TaylorBundle{N, Tuple{B1,B2}}) where {N, B1,B2}
    partial(r, 1)[1] == primal(r)[2] || return 0
    for i in 1:(N-1)
        partial(r, i+1)[1] == partial(r, i)[2] || return i
    end
    return -1  # all compatible
end

function taylor_failure_values(r::TaylorBundle{<:Any, Tuple{Any,Any}}, fail_order)
    fail_order == 0 && return partial(r,1)[1], primal(r)[2]
    return partial(r, i+1)[1], partial(r, i)[2]
end

function shuffle_up(r::TaylorBundle{1, Tuple{B1,B2}}, ::Val{taylor_or_bust}) where {B1,B2, taylor_or_bust}
    z₀ = primal(r)[1]
    z₁ = partial(r, 1)[1]
    z₂ = primal(r)[2]
    z₁₂ = partial(r, 1)[2]

    taylor_fail_order = find_taylor_incompatibility(r)
    if  taylor_fail_order < 0
        return TaylorBundle{2}(z₀, (z₁, z₁₂))
    elseif taylor_or_bust
        @assert taylor_fail_order == 0  # can't be higher
        throw(TaylorRequired(taylor_fail_order, z₁, z₂))
    else
        return ExplicitTangentBundle{2}(z₀, (z₁, z₂, z₁₂))
    end
end

function shuffle_up(r::TaylorBundle{N, Tuple{B1,B2}}, ::Val{taylor_or_bust}) where {N, B1,B2, taylor_or_bust}
    the_primal = primal(r)[1]
    taylor_fail_order = find_taylor_incompatibility(r)
    if taylor_fail_order(r) < 0
        the_partials = ntuple(N+1) do i
            if i <= N
                partial(r, i)[1]  # == `partial(r,i-1)[2]` (except first which is primal(r)[2])
            else  # ii = N+1
                partial(r, i-1)[2]
            end
        end
        return TaylorBundle{N+1}(the_primal, the_partials)
    elseif taylor_or_bust
        @assert taylor_fail_order < N
        throw(TaylorRequired(taylor_fail_order, taylor_failure_values(r, taylor_fail_order)...))
    else
        #XXX: am dubious of the correctness of this
        a_partials = ntuple(i->partial(r, i)[1], N)
        b_partials = ntuple(i->partial(r, i)[2], N)
        the_partials = (a_partials..., primal_b, b_partials...)
        return TangentBundle{N+1}(the_primal, the_partials)
    end
end


function shuffle_up(r::UniformBundle{N, B, U}, _::Val) where {N, B, U}
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
@ChainRulesCore.non_differentiable shuffle_up(r::UniformBundle, ::Val)


function shuffle_up_bundle(r::Diffractor.TangentBundle{1, B}) where {B<:ATB{1}}
    a = primal(r)
    b = partial(r, 1)
    z₀ = primal(a)
    z₁ = partial(a, 1)
    z₂ = b.primal
    z₁₂ = _shuffle_up_partial₁₂(B, b.tangent)

    if z₁ == z₂
        return TaylorBundle{2}(z₀, (z₁, z₁₂))
    else
        return ExplicitTangentBundle{2}(z₀, (z₁, z₂, z₁₂))
    end
end

_shuffle_up_partial₁₂(::Type{<:TaylorBundle}, tangent) = only(tangent.coeffs)
_shuffle_up_partial₁₂(::Type{<:ExplicitTangentBundle}, tangent) = only(tangent.partials)
_shuffle_up_partial₁₂(::Type{<:UniformBundle}, tangent) = tangent.val


function shuffle_up_bundle(r::UniformBundle{1, <:UniformBundle{N, B, U}}) where {N, B, U}
    return UniformBundle{N+1, B, U}(primal(primal(r)))
end
function shuffle_up_bundle(r::UniformBundle{1, <:UniformBundle{1, B, U}}) where {B, U}  # break ambig
    return UniformBundle{2, B, U}(primal(primal(r)))
end


function shuffle_down_bundle(b::TaylorBundle{2, B}) where {B}
    z₀ = primal(b)
    z₁ = b.tangent.coeffs[1]
    z₁₂ = b.tangent.coeffs[2]
    TaylorBundle{1}(TaylorBundle{1}(z₀, (z₁,)), (TaylorBundle{1}(z₁, (z₁₂,)),))
end

#N order, this should be a positive Int
#E eras mode, this controls if we should Error if it isn't Taylor. This should be a Bool
struct ∂☆internal{N, E}; end
struct ∂☆recurse{N, E}; end
∂☆recurse{N}() where N = ∂☆recurse{N,false}
struct ∂☆shuffle{N}; end

function shuffle_base(r)
    (primal, dual) = r
    if dual isa NoTangent
        UniformBundle{1}(primal, dual)
    else
        if dual isa ZeroTangent  # Normalize zero for type-stability reasons
            dual = zero_tangent(primal)
        end
        TaylorBundle{1}(primal, (dual,))
    end
end

function (::∂☆internal{1, E})(args::AbstractTangentBundle{1}...) where E
    r = _frule(Val{E}(), map(first_partial, args), map(primal, args)...)
    if r === nothing
        return ∂☆recurse{1, E}()(args...)
    else
        return shuffle_base(r)
    end
end

# TODO: workout why enabling calling back into AD in Eras mode causes type instability
_frule(::Val{true}, partials, primals...) = frule(partials, primals...)
_frule(::Val{false}, partials, primals...) = frule(DiffractorRuleConfig(), partials, primals...)
function _frule(::Any, ::NTuple{<:Any, AbstractZero}, f, primal_args...)
    # frules are linear in partials, so zero maps to zero, no need to evaluate the frule
    # If all partials are immutable AbstractZero subtyoes we know we don't have to worry about a mutating frule either
    r = f(primal_args...)
    return r, zero_tangent(r)
end

function ChainRulesCore.frule_via_ad(::DiffractorRuleConfig, partials, args...)
    bundles = map(bundle, partials, args)
    result = ∂☆internal{1,false}()(bundles...)
    primal(result), first_partial(result)
end

function (::∂☆shuffle{N})(args::AbstractTangentBundle{N}...) where {N}
    ∂☆p = ∂☆{N-1}()
    downargs = map(shuffle_down, args)
    #@info "∂☆shuffle{N}" args downargs
    tupargs = ∂vararg{N-1}()(map(first_partial, downargs)...)
    ∂☆p(ZeroBundle{N-1}(frule), #= ZeroBundle{N-1}(DiffractorRuleConfig()), =# tupargs, map(primal, downargs)...)
end

# Special shortcut case if there is no derivative information at all:
function (::∂☆internal{N})(f::AbstractZeroBundle{N}, args::AbstractZeroBundle{N}...) where {N}
    f_v = primal(f)
    args_v = map(primal, args)
    return zero_bundle{N}()(f_v(args_v...))
end
function (::∂☆internal{1})(f::AbstractZeroBundle{1}, args::AbstractZeroBundle{1}...)
    f_v = primal(f)
    args_v = map(primal, args)
    return zero_bundle{1}()(f_v(args_v...))
end

function (::∂☆internal{N, E})(args::AbstractTangentBundle{N}...) where {N, E}
    r = ∂☆shuffle{N}()(args...)
    if primal(r) === nothing
        return ∂☆recurse{N, E}()(args...)
    else
        return shuffle_up(r, Val{E}())
    end
end

# TODO: Generalize to N,M
@inline function (::∂☆{1,E})(rec::AbstractZeroBundle{1, ∂☆recurse{1, E}}, args::ATB{1}...) where E
    return shuffle_down_bundle(∂☆recurse{2,E}()(map(shuffle_up_bundle, args)...))
end

(::∂☆{N,E})(args::AbstractTangentBundle{N}...) where {N,E} = ∂☆internal{N,E}()(args...)

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


@Base.constprop :aggressive function (::∂☆{N})(f::ATB{N, typeof(getfield)}, x::UniformBundle{N, <:Any, U}, s::AbstractTangentBundle{N}) where {N, U}
    UniformBundle{N,<:Any,U}(getfield(primal(x), primal(s)), x.tangent.val)
end

@Base.constprop :aggressive function (::∂☆{N})(f::ATB{N, typeof(getfield)}, x::UniformBundle{N, <:Any, U}, s::AbstractTangentBundle{N}, inbounds::AbstractTangentBundle{N}) where {N, U}
    UniformBundle{N,<:Any,U}(getfield(primal(x), primal(s), primal(inbounds)), x.tangent.val)
end

function (::∂☆{N})(f::ATB{N, typeof(tuple)}, args::AbstractTangentBundle{N}...) where {N}
    ∂vararg{N}()(args...)
end

function (::∂☆{N})(f::ATB{N, typeof(tuple)}, args::AbstractZeroBundle{N}...) where {N}
    ZeroBundle{N}(map(primal, args))  # special fast case
end

struct FwdMap{N, E, T<:AbstractTangentBundle{N}}
    f::T
end
FwdMap{E}(f::T) where {N, E, T<:AbstractTangentBundle{N}} = FwdMap{N,E,T}(f)
(f::FwdMap{N,E})(args::AbstractTangentBundle{N}...) where {N,E} = ∂☆{N,E}()(f.f, args...)

function (::∂☆{N,E})(::AbstractZeroBundle{N, typeof(map)}, f::ATB{N}, tup::TaylorBundle{N, <:Tuple}) where {N,E}
    ∂vararg{N}()(map(FwdMap{E}(f), destructure(tup))...)
end

function (::∂☆{N,E})(::AbstractZeroBundle{N, typeof(map)}, f::ATB{N}, args::ATB{N, <:AbstractArray}...) where {N,E}
    # TODO: This could do an inplace map! to avoid the extra rebundling
    rebundle(map(FwdMap{E}(f), map(unbundle, args)...))
end

function (::∂☆{N,E})(::AbstractZeroBundle{N, typeof(map)}, f::ATB{N}, args::ATB{N}...) where {N, E}
    ∂☆recurse{N,E}()(ZeroBundle{N, typeof(map)}(map), f, args...)
end


function (::∂☆{N})(f::AbstractZeroBundle{N, typeof(ifelse)}, arg::ATB{N, Bool}, args::ATB{N}...) where {N}
    ifelse(arg.primal, args...)
end

function (::∂☆{N})(f::AbstractZeroBundle{N, typeof(Core.ifelse)}, arg::ATB{N, Bool}, args::ATB{N}...) where {N}
    Core.ifelse(arg.primal, args...)
end

struct FwdIterate{N, E, T<:AbstractTangentBundle{N}}
    f::T
end
FwdIterate{E}(f::T) where {N, E, T<:AbstractTangentBundle{N}} = FwdIterate{N,E,T}(f)
function (f::FwdIterate{N,E})(arg::ATB{N}) where {N,E}
    r = ∂☆{N,E}()(f.f, arg)
    # `primal(r) === nothing` would work, but doesn't create `Conditional` in inference
    isa(r, ATB{N, Nothing}) && return nothing
    (∂☆{N,E}()(ZeroBundle{N}(getindex), r, ZeroBundle{N}(1)),
     primal(∂☆{N,E}()(ZeroBundle{N}(getindex), r, ZeroBundle{N}(2))))
end
@Base.constprop :aggressive function (f::FwdIterate{N,E})(arg::ATB{N}, st) where {N,E}
    r = ∂☆{N,E}()(f.f, arg, ZeroBundle{N}(st))
    # `primal(r) === nothing` would work, but doesn't create `Conditional` in inference
    isa(r, ATB{N, Nothing}) && return nothing
    (∂☆{N}()(ZeroBundle{N}(getindex), r, ZeroBundle{N}(1)),
     primal(∂☆{N,E}()(ZeroBundle{N}(getindex), r, ZeroBundle{N}(2))))
end

function (this::∂☆{N,E})(::AbstractZeroBundle{N, typeof(Core._apply_iterate)}, iterate::ATB{N}, f::ATB{N}, args::ATB{N}...) where {N,E}
    Core._apply_iterate(FwdIterate{E}(iterate), this, (f,), args...)
end

function (this::∂☆{N})(::AbstractZeroBundle{N, typeof(iterate)}, t::TaylorBundle{N, <:Tuple}) where {N}
    r = iterate(destructure(t))
    r === nothing && return ZeroBundle{N}(nothing)
    ∂vararg{N}()(r[1], ZeroBundle{N}(r[2]))
end

function (this::∂☆{N})(::AbstractZeroBundle{N, typeof(iterate)}, t::TaylorBundle{N, <:Tuple}, a::ATB{N}, args::ATB{N}...) where {N}
    r = iterate(destructure(t), primal(a), map(primal, args)...)
    r === nothing && return ZeroBundle{N}(nothing)
    ∂vararg{N}()(r[1], ZeroBundle{N}(r[2]))
end

function (this::∂☆{N})(::AbstractZeroBundle{N, typeof(Base.indexed_iterate)}, t::TaylorBundle{N, <:Tuple}, i::ATB{N}) where {N}
    r = Base.indexed_iterate(destructure(t), primal(i))
    ∂vararg{N}()(r[1], ZeroBundle{N}(r[2]))
end

function (this::∂☆{N})(::AbstractZeroBundle{N, typeof(Base.indexed_iterate)}, t::TaylorBundle{N, <:Tuple}, i::ATB{N}, st1::ATB{N}, st::ATB{N}...) where {N}
    r = Base.indexed_iterate(destructure(t), primal(i), primal(st1), map(primal, st)...)
    ∂vararg{N}()(r[1], ZeroBundle{N}(r[2]))
end

function (this::∂☆{N})(::AbstractZeroBundle{N, typeof(Base.indexed_iterate)}, t::TangentBundle{N, <:Tuple}, i::ATB{N}, st::ATB{N}...) where {N}
    ∂vararg{N}()(this(ZeroBundle{N}(getfield), t, i), ZeroBundle{N}(primal(i) + 1))
end

function (this::∂☆{N})(::AbstractZeroBundle{N, typeof(getindex)}, t::TaylorBundle{N, <:Tuple}, i::AbstractZeroBundle) where {N}
    field_ind = primal(i)
    the_partials = ntuple(order_ind->partial(t, order_ind)[field_ind], N)
    TaylorBundle{N}(primal(t)[field_ind], the_partials)
end

function (this::∂☆{N})(::AbstractZeroBundle{N, typeof(typeof)}, x::ATB{N}) where {N}
    DNEBundle{N}(typeof(primal(x)))
end


function (this::∂☆{N})(f::AbstractZeroBundle{N, Core.IntrinsicFunction}, args::AbstractZeroBundle{N}...) where {N}
    ff=primal(f)
    return (zero_bundle{N}())(ff(map(primal, args)...))
end

function (this::∂☆{N})(f::AbstractZeroBundle{N, Core.IntrinsicFunction}, args::ATB{N}...) where {N}
    ff = primal(f)
    if ff in (Base.not_int, Base.ne_float)
        DNEBundle{N}(ff(map(primal, args)...))
    else
        error("Missing rule for intrinsic function $ff")
    end
end
