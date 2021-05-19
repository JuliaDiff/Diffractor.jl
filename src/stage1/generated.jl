using ChainRulesCore: NO_FIELDS
using Base.Experimental: @opaque

struct ∂⃖rrule{N}; end
struct ∂⃖recurse{N}; end

include("recurse.jl")

function perform_optic_transform(@nospecialize(ff::Type{∂⃖recurse{N}}), @nospecialize(args)) where {N}
    @assert N >= 1

    # Check if we have an rrule for this function
    mthds = Base._methods_by_ftype(Tuple{args...}, -1, typemax(UInt))
    if length(mthds) != 1
        @show args
        @show mthds
        error()
    end
    match = mthds[1]

    mi = Core.Compiler.specialize_method(match)
    ci = Core.Compiler.retrieve_code_info(mi)

    ci′ = copy(ci)
    ci′.edges = MethodInstance[mi]

    transform!(ci′, mi.def, length(args) - 1, match.sparams, N)

    ci′.ssavaluetypes = length(ci′.code)
    ci′.method_for_inference_limit_heuristics = match.method
    ci′
end

# This relies on PartialStruct to infer well
struct Protected{N}
    a
end
(p::Protected)(args...) = getfield(p, :a)(args...)[1]
@Base.aggressive_constprop (::∂⃖{N})(p::Protected{N}, args...) where {N} = getfield(p, :a)(args...)
@Base.aggressive_constprop (::∂⃖{1})(p::Protected{1}, args...) = getfield(p, :a)(args...)
(::∂⃖{N})(p::Protected, args...) where {N} = error("TODO: Can we support this?")

struct OpticBundle{T}
    x::T
    clos
end
Base.iterate(o::OpticBundle) = (o.x, nothing)
Base.iterate(o::OpticBundle, ::Nothing) = (o.clos, missing)
Base.iterate(o::OpticBundle, ::Missing) = nothing

# Desturucture using `getfield` rather than iterate to make
# inference happier
macro destruct(arg)
    @assert isexpr(arg, :(=))
    lhs = arg.args[1]
    @assert isexpr(lhs, :tuple)
    rhs = arg.args[2]
    s = gensym()
    quote
        $s = $(esc(rhs))
        $([:($(esc(arg)) = getfield($s, $i)) for (i, arg) in enumerate(lhs.args)]...)
        $s
    end
end

@eval function (::∂⃖{1})(::∂⃖{2}, args...)
    @destruct a, b = ∂⃖{3}()(args...)
    (a, $(Expr(:new, Protected{1}, :(Δ->begin
        x1, x2 = b(Δ)
        (x1, $(Expr(:new, Protected{1}, :((Δ...)->begin
            x3, x4 = x2(Δ...)
            (x3, $(Expr(:new, Protected{1}, :(Δ->begin
                x5, x6 = x4(Δ)
                x5, Δ->begin
                    (x7, x8) = x6(Δ...)
                    return (x8, x7)
                end
            end)))), ((x7, x8),)->begin
                (x9, x10) = x8(x7)
                (x10, x9...)
            end
        end)))), ((x9, x10),)->begin
            (x11, x12) = x10(x9...)
            (x12, x11)
        end
    end)))), ((x11, x12),)->begin
        (NO_FIELDS, x12(x11)...)
    end
end

struct ∂⃖weaveInnerOdd{N, O}; b̄; end
@Base.aggressive_constprop function (w::∂⃖weaveInnerOdd{N, N})(Δ) where {N}
    @destruct c, c̄ = w.b̄(Δ...)
    return (c̄, c)
end
@Base.aggressive_constprop function (w::∂⃖weaveInnerOdd{N, O})(Δ) where {N, O}
    @destruct c, c̄ = w.b̄(Δ...)
    return (c̄, c), ∂⃖weaveInnerEven{plus1(N), O}()
end
struct ∂⃖weaveInnerEven{N, O}; end
@Base.aggressive_constprop function (w::∂⃖weaveInnerEven{N, O})(Δ′, x...) where {N, O}
    @destruct y, ȳ  = Δ′(x...)
    return y, ∂⃖weaveInnerOdd{plus1(N), O}(ȳ)
end

struct ∂⃖weaveOuterOdd{N, O}; end
@Base.aggressive_constprop function (w::∂⃖weaveOuterOdd{N, N})((Δ′′, Δ′′′)) where {N}
    return (NO_FIELDS, Δ′′′(Δ′′)...)
end
@Base.aggressive_constprop function (w::∂⃖weaveOuterOdd{N, O})((Δ′′, Δ′′′)) where {N, O}
    @destruct α, ᾱ = Δ′′′(Δ′′)
    return (NO_FIELDS, α...), ∂⃖weaveOuterEven{plus1(N), O}(ᾱ)
end
struct ∂⃖weaveOuterEven{N, O}; ᾱ end
@Base.aggressive_constprop function (w::∂⃖weaveOuterEven{N, O})(Δ⁴...) where {N, O}
    return w.ᾱ(Base.tail(Δ⁴)...), ∂⃖weaveOuterOdd{plus1(N), O}()
end

function (::∂⃖{N})(::∂⃖{1}, args...) where {N}
    @destruct (a, ā) = ∂⃖{plus1(N)}()(args...)
    let O = c_order(N)
        (a, Protected{N}(@opaque Δ->begin
                (b, b̄) = ā(Δ)
                b, ∂⃖weaveInnerOdd{1, O}(b̄)
            end
        )), ∂⃖weaveOuterOdd{1, O}()
    end
end

function (::∂⃖{N})(::∂⃖{M}, args...) where {N, M}
    # TODO
    @destruct (a, b) = ∂⃖{N+M}()(args...)
    error("Not implemented yet ($N, $M)")
end

macro OpticBundle(a, b)
    aa = gensym()
    esc(quote
        $aa = $a
        $(Expr(:new, :(OpticBundle{typeof($aa)}), aa, b))
    end)
end

# ∂⃖rrule has a 4-recurrence - we model this as 4 separate structs that we
# cycle between. N.B.: These names match the names that these variables
# have in Snippet 19 of the terminology guide. They are probably not ideal,
# but if you rename them here, please update the terminology guide also.

struct ∂⃖rruleA{N, O}; ∂; ȳ; ȳ̄ ; end
struct ∂⃖rruleB{N, O}; ᾱ; ȳ̄ ; end
struct ∂⃖rruleC{N, O}; ȳ̄ ; Δ′′′; β̄ ; end
struct ∂⃖rruleD{N, O}; γ̄; β̄ ; end

@Base.aggressive_constprop function (a::∂⃖rruleA{N, O})(Δ) where {N, O}
    @destruct (α, ᾱ) = a.∂(a.ȳ, Δ)
    (α, ∂⃖rruleB{N, O}(ᾱ, a.ȳ̄))
end

@Base.aggressive_constprop function (b::∂⃖rruleB{N, O})(Δ′...) where {N, O}
    @destruct ((Δ′′′, β), β̄) = b.ᾱ(Δ′)
    (β, ∂⃖rruleC{N, O}(b.ȳ̄, Δ′′′, β̄))
end

@Base.aggressive_constprop function (c::∂⃖rruleC{N, O})(Δ′′) where {N, O}
    @destruct (γ, γ̄) = c.ȳ̄((Δ′′, c.Δ′′′))
    (Base.tail(γ), ∂⃖rruleD{N, O}(γ̄, c.β̄))
end

@Base.aggressive_constprop function (d::∂⃖rruleD{N, O})(Δ⁴...) where {N, O}
    (δ₁, δ₂), δ̄  = d.γ̄(ZeroTangent(), Δ⁴...)
    (δ₁, ∂⃖rruleA{N, O+1}(d.β̄ , δ₂, δ̄ ))
end

# Terminal cases
@Base.aggressive_constprop function (c::∂⃖rruleB{N, N})(Δ′...) where {N}
    @destruct (Δ′′′, β) = c.ᾱ(Δ′)
    (β, ∂⃖rruleC{N, N}(c.ȳ̄, Δ′′′, nothing))
end
@Base.aggressive_constprop (c::∂⃖rruleC{N, N})(Δ′′) where {N} =
    Base.tail(c.ȳ̄((Δ′′, c.Δ′′′)))
(::∂⃖rruleD{N, N})(Δ...) where {N} = error("Should not be reached")

# ∂⃖rrule
@Base.pure term_depth(N) = 2^(N-2)
function (::∂⃖rrule{N})(z, z̄) where {N}
    @destruct (y, ȳ) = z
    y, ∂⃖rruleA{term_depth(N), 1}(∂⃖{minus1(N)}(), ȳ, z̄)
end

# The static parameter on `f` disables the compileable_sig heuristic
function (::∂⃖{N})(f::T, args...) where {T, N}
    if N == 1
        # Base case (inlined to avoid ambiguities with manually specified
        # higher order rules)
        z = rrule(f, args...)
        if z === nothing
            return ∂⃖recurse{1}()(f, args...)
        end
        return z
    else
        ∂⃖p = ∂⃖{minus1(N)}()
        @destruct z, z̄ = ∂⃖p(rrule, f, args...)
        if z === nothing
            return ∂⃖recurse{N}()(f, args...)
        else
            return ∂⃖rrule{N}()(z, z̄)
        end
    end
end

@Base.pure function (::∂⃖{1})(::typeof(Core.apply_type), head, args...)
    return rrule(Core.apply_type, head, args...)
end

@Base.aggressive_constprop function ChainRulesCore.rrule(::typeof(Core.getfield), s, field::Symbol)
    getfield(s, field), let P = typeof(s)
        @Base.aggressive_constprop Δ->begin
            nt = NamedTuple{(field,)}((Δ,))
            (NO_FIELDS, Tangent{P, typeof(nt)}(nt), NO_FIELDS)
        end
    end
end

struct ∂⃖getfield{n, f}; end
@Base.aggressive_constprop function (::∂⃖getfield{n, f})(Δ) where {n,f}
    if @generated
        return Expr(:call, tuple, NO_FIELDS,
            Expr(:call, tuple, (i == f ? :(Δ) : NoTangent() for i = 1:n)...),
            NO_FIELDS)
    else
        return (NO_FIELDS, ntuple(i->i == f ? Δ : NoTangent(), n), NO_FIELDS)
    end
end

struct EvenOddEven{O, P, F, G}; f::F; g::G; end
EvenOddEven{O, P}(f::F, g::G) where {O, P, F, G} = EvenOddEven{O, P, F, G}(f, g)
struct EvenOddOdd{O, P, F, G}; f::F; g::G; end
EvenOddOdd{O, P}(f::F, g::G) where {O, P, F, G} = EvenOddOdd{O, P, F, G}(f, g)
@Base.aggressive_constprop (o::EvenOddOdd{O, P, F, G})(Δ) where {O, P, F, G} = (o.f(Δ), EvenOddEven{plus1(O), P, F, G}(o.f, o.g))
@Base.aggressive_constprop (e::EvenOddEven{O, P, F, G})(Δ...) where {O, P, F, G} = (e.g(Δ...), EvenOddOdd{plus1(O), P, F, G}(e.f, e.g))
@Base.aggressive_constprop (o::EvenOddOdd{O, O})(Δ) where {O} = o.f(Δ)


@Base.aggressive_constprop function (::∂⃖{N})(::typeof(Core.getfield), s, field::Int) where {N}
    getfield(s, field), EvenOddOdd{1, c_order(N)}(
        ∂⃖getfield{nfields(s), field}(),
        @Base.aggressive_constprop (_, Δ, _)->getfield(Δ, field))
end

@Base.aggressive_constprop function (::∂⃖{N})(::typeof(Base.getindex), s::Tuple, field::Int) where {N}
    getfield(s, field), EvenOddOdd{1, c_order(N)}(
        ∂⃖getfield{nfields(s), field}(),
        @Base.aggressive_constprop (_, Δ, _)->lifted_getfield(Δ, field))
end

function (::∂⃖{N})(::typeof(Core.getfield), s, field::Symbol) where {N}
    getfield(s, field), let P = typeof(s)
        EvenOddOdd{1, c_order(N)}(
            (@Base.aggressive_constprop Δ->begin
                nt = NamedTuple{(field,)}((Δ,))
                (NO_FIELDS, Tangent{P, typeof(nt)}(nt), NO_FIELDS)
            end),
            (@Base.aggressive_constprop (_, Δs, _)->begin
                isa(Δs, Union{ZeroTangent, NoTangent}) ? Δs : getfield(ChainRulesCore.backing(Δs), field)
            end))
    end
end

# TODO: Temporary - make better
function (::∂⃖{N})(::typeof(Base.getindex), a::Array, inds...) where {N}
    getindex(a, inds...), let
        EvenOddOdd{1, c_order(N)}(
            (@Base.aggressive_constprop Δ->begin
                BB = zero(a)
                BB[inds...] = Δ
                (NO_FIELDS, BB, map(x->NO_FIELDS, inds)...)
            end),
            (@Base.aggressive_constprop (_, Δ, _)->begin
                getindex(Δ, inds...)
            end))
    end
end

function (::∂⃖{N})(::typeof(Core.tuple), args...) where {N}
    Core.tuple(args...), EvenOddOdd{1, c_order(N)}(
        Δ->Core.tuple(NO_FIELDS, Δ...),
        (Δ...)->Core.tuple(Δ[2:end]...)
    )
end

struct UnApply{Spec}; end
@generated function (::UnApply{Spec})(Δ) where Spec
    args = Any[NO_FIELDS, NO_FIELDS, :(Δ[1])]
    start = 2
    for l in Spec
        push!(args, :(Δ[$(start:(start+l-1))]))
        start += l
    end
    :(Core.tuple($(args...)))
end

struct ApplyOdd{O, P}; u; ∂⃖f; end
struct ApplyEven{O, P}; u; ∂⃖∂⃖f; end
@Base.aggressive_constprop function (a::ApplyOdd{O, P})(Δ) where {O, P}
    r, ∂⃖∂⃖f = a.∂⃖f(Δ)
    (a.u(r), ApplyEven{plus1(O), P}(a.u, ∂⃖∂⃖f))
end
@Base.aggressive_constprop function (a::ApplyEven{O, P})(_, _, ff, args...) where {O, P}
    r, ∂⃖∂⃖∂⃖f = Core._apply_iterate(iterate, a.∂⃖∂⃖f, (ff,), args...)
    (r, ApplyOdd{plus1(O), P}(a.u, ∂⃖∂⃖∂⃖f))
end
@Base.aggressive_constprop function (a::ApplyOdd{O, O})(Δ) where {O}
    r = a.∂⃖f(Δ)
    a.u(r)
end

function (this::∂⃖{N})(::typeof(Core._apply_iterate), iterate, f, args::Tuple...) where {N}
    @assert iterate === Base.iterate
    x, ∂⃖f = Core._apply_iterate(iterate, this, (f,), args...)
    return x, ApplyOdd{1, c_order(N)}(UnApply{map(length, args)}(), ∂⃖f)
end


@Base.pure c_order(N::Int) = 2^N - 1

@Base.pure function (::∂⃖{N})(::typeof(Core.apply_type), head, args...) where {N}
    Core.apply_type(head, args...), NonDiffOdd{plus1(plus1(length(args))), 1, c_order(N)}()
end

@Base.aggressive_constprop lifted_getfield(x, s) = getfield(x, s)
lifted_getfield(x::ZeroTangent, s) = ZeroTangent()
lifted_getfield(x::NoTangent, s) = NoTangent()

ChainRulesCore.backing(::ZeroTangent) = ZeroTangent()
ChainRulesCore.backing(::NoTangent) = NoTangent()

function reload()
    Core.eval(Diffractor, quote
        function (ff::∂⃖recurse)(args...)
            $(Expr(:meta, :generated_only))
            $(Expr(:meta,
                    :generated,
                    Expr(:new,
                        Core.GeneratedFunctionStub,
                        :perform_optic_transform,
                        Any[:ff, :args],
                        Any[],
                        @__LINE__,
                        QuoteNode(Symbol(@__FILE__)),
                        true)))
        end
    end)
end
reload()
