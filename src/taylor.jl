using TaylorSeries

struct ∂Taylor{O, N, T}
    t::T
end
∂Taylor{O, N}(t::T) where {O,N,T} = ∂Taylor{O, N, T}(t)

function which_order(N)
    # OEIS A005811, but offset at 0
    N == 0 && return 0
    N == 1 && return 1
    P = prevpow(2, N)
    which_order(2P - N - 1) + 1
end

function (t::∂Taylor{O, N})(Δ) where {O, N}
    ff = unthunk(Δ) .* let P = which_order(N)
        @show (P, t.t)
        map(t->factorial(P)*t[P], t.t)
    end,
    if isodd(O)
        return (NO_FIELDS, NO_FIELDS, ff), ∂Taylor{O+1, N}(t.t)
    else
        ff, ∂Taylor{O+1, N}(t.t)
    end
end

function (t::∂Taylor{N, N})(Δ) where {N}
    (NO_FIELDS, NO_FIELDS, unthunk(Δ) .* map(t->t[1], t.t))
end

function (::∂⃖{N})(::typeof(map), f, a::Array) where {N}
    @assert Base.issingletontype(typeof(f))
    # TODO: Built-in taylor mode
    t = map(a) do x
        f(x + Taylor1(typeof(x), N))
    end
    tt = map(t->t[0], t)
    map(t->t[0], t), ∂Taylor{1, c_order(N)}(t)
end

