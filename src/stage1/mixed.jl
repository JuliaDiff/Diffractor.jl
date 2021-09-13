# Reverse over forward - the compiler should get this itself, but this saves
# some work in the hot path.

struct ∂⃖composeOdd{C, N}
    a
    b
end

struct ∂⃖composeEven{C, N}
    a
    b
end

function (x::∂⃖composeOdd)(Δ)
    b, ∂b = x.b(Δ)
    a, ∂a = x.a(b[end])
    a, ∂⃖composeEven{N, plus1(N)}(∂a, ∂b)
end

function (x::∂⃖composeEven)(args...)
    a, ∂a = x.a(args...)
    b, ∂b = x.b(a)
    b, ∂⃖composeOdd{N, plus1(N)}(∂a, ∂b)
end

function (x::∂⃖composeOdd{N,N})(Δ) where {N}
    return x.a(x.b(Δ)[end])
end

function (this::∂⃖{N})(::∂☆internal{1}, args::AbstractTangentBundle{1}...) where {N}
    r, ∂r = this(my_frule, args...)
    if r === nothing
        # Forwards directly to the recursion, no need to ever call the
        # backwards for my_frule.
        return this(∂☆recurse{1}(), args...)
    else
        z, ∂z = this(shuffle_base, r)
        return z, ∂⃖composeOdd{1, c_order(N)}(∂r, ∂z)
    end
end

function shuffle_down_frule(∂☆p, my_frule, args...)
    ∂☆p(my_frule, map(shuffle_down, args)...)
end

function (this::∂⃖{N})(::∂☆internal{M}, args::AbstractTangentBundle{1}...) where {N, M}
    r = this(∂☆shuffle{N}(), args...)
    if primal(r) === nothing
        return this(∂☆recurse{N}(), args...)
    else
        z, ∂z = this(shuffle_up, r)
        return z, ∂⃖composeOdd{1, c_order(N)}(∂r, ∂z)
    end
end

#=
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
=#

function (this::∂⃖{N})(that::∂☆{M}, ::ZeroBundle{M, typeof(Core._apply_iterate)},
        iterate, f, args::ATB{M, <:Tuple}...) where {N, M}
    @assert primal(iterate) === Base.iterate
    x, ∂⃖f = Core._apply_iterate(FwdIterate(iterate), this, (that, f), args...)
    return x, ApplyOdd{1, c_order(N)}(UnApply{map(x->length(primal(x)), args)}(), ∂⃖f)
end


function ChainRules.rrule(∂::∂☆{N}, m::ZeroBundle{N, typeof(map)}, p::ZeroBundle{N, typeof(+)}, A::ATB{N}, B::ATB{N}) where {N}
    ∂(m, p, A, B), Δ->(NoTangent(), NoTangent(), NoTangent(), Δ, Δ)
end

mapev_unbundled(_, js, a) = rebundle(mapev(js, unbundle(a)))
function (∂⃖ₙ::∂⃖{N})(∂☆ₘ::∂☆{M}, ::ZeroBundle{M, typeof(map)},
                    f::ZeroBundle{M}, a::ATB{M, <:Array}) where {N, M}
    @assert Base.issingletontype(typeof(primal(f)))
    js = map(primal(a)) do x
        ∂f = ∂☆{N+M}()(ZeroBundle{N+M}(primal(f)),
                     TaylorBundle{N+M}(x,
                       (one(x), (zero(x) for i = 1:(N+M-1))...,)))
        @assert isa(∂f, TaylorBundle) || isa(∂f, ExplicitTangentBundle{1})
        Jet{typeof(x), N+M}(x, ∂f.primal,
            isa(∂f, ExplicitTangentBundle) ? ∂f.tangent.partials : ∂f.tangent.coeffs)
    end
    ∂⃖ₙ(mapev_unbundled, ∂☆ₘ, js, a)
end
