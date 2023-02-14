using ChainRulesCore: NoTangent
using Base.Experimental: @opaque

struct ∂⃖rrule{N}; end
struct ∂⃖recurse{N}; end

include("recurse.jl")

function perform_optic_transform(@nospecialize(ff::Type{∂⃖recurse{N}}), @nospecialize(args)) where {N}
    @assert N >= 1

    # Check if we have an rrule for this function
    mthds = Base._methods_by_ftype(Tuple{args...}, -1, typemax(UInt))
    if length(mthds) != 1
        return :(throw(MethodError(ff, args)))
    end
    match = mthds[1]

    mi = Core.Compiler.specialize_method(match)
    ci = Core.Compiler.retrieve_code_info(mi)

    ci′ = copy(ci)
    ci′.edges = MethodInstance[mi]

    r = transform!(ci′, mi.def, length(args) - 1, match.sparams, N)
    if isa(r, Expr)
        return r
    end

    ci′.ssavaluetypes = length(ci′.code)
    ci′.ssaflags = UInt8[0 for i=1:length(ci′.code)]
    ci′.method_for_inference_limit_heuristics = match.method
    ci′
end

# This relies on PartialStruct to infer well
struct Protected{N}
    a
end
(p::Protected)(args...) = getfield(p, :a)(args...)[1]
@Base.constprop :aggressive (::∂⃖{N})(p::Protected{N}, args...) where {N} = getfield(p, :a)(args...)
@Base.constprop :aggressive (::∂⃖{1})(p::Protected{1}, args...) = getfield(p, :a)(args...)
(::∂⃖{N})(p::Protected, args...) where {N} = error("TODO: Can we support this?")

struct OpticBundle{T}
    x::T
    clos
end
Base.getindex(o::OpticBundle, i::Int) = i == 1 ? o.x :
                                        i == 2 ? o.clos :
                                        throw(BoundsError(o, i))
Base.iterate(o::OpticBundle) = (o.x, nothing)
Base.iterate(o::OpticBundle, ::Nothing) = (o.clos, missing)
Base.iterate(o::OpticBundle, ::Missing) = nothing
Base.length(o::OpticBundle) = 2

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
        (NoTangent(), x12(x11)...)
    end
end

struct ∂⃖weaveInnerOdd{N, O}; b̄; end
@Base.constprop :aggressive function (w::∂⃖weaveInnerOdd{N, N})(Δ) where {N}
    @destruct c, c̄ = w.b̄(Δ...)
    return (c̄, c)
end
@Base.constprop :aggressive function (w::∂⃖weaveInnerOdd{N, O})(Δ) where {N, O}
    @destruct c, c̄ = w.b̄(Δ...)
    return (c̄, c), ∂⃖weaveInnerEven{N+1, O}()
end
struct ∂⃖weaveInnerEven{N, O}; end
@Base.constprop :aggressive function (w::∂⃖weaveInnerEven{N, O})(Δ′, x...) where {N, O}
    @destruct y, ȳ  = Δ′(x...)
    return y, ∂⃖weaveInnerOdd{N+1, O}(ȳ)
end

struct ∂⃖weaveOuterOdd{N, O}; end
@Base.constprop :aggressive function (w::∂⃖weaveOuterOdd{N, N})((Δ′′, Δ′′′)) where {N}
    return (NoTangent(), Δ′′′(Δ′′)...)
end
@Base.constprop :aggressive function (w::∂⃖weaveOuterOdd{N, O})((Δ′′, Δ′′′)) where {N, O}
    @destruct α, ᾱ = Δ′′′(Δ′′)
    return (NoTangent(), α...), ∂⃖weaveOuterEven{N+1, O}(ᾱ)
end
struct ∂⃖weaveOuterEven{N, O}; ᾱ end
@Base.constprop :aggressive function (w::∂⃖weaveOuterEven{N, O})(Δ⁴...) where {N, O}
    return w.ᾱ(Base.tail(Δ⁴)...), ∂⃖weaveOuterOdd{N+1, O}()
end

function (::∂⃖{N})(::∂⃖{1}, args...) where {N}
    @destruct (a, ā) = ∂⃖{N+1}()(args...)
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

@Base.constprop :aggressive function (a::∂⃖rruleA{N, O})(Δ) where {N, O}
    # TODO: Is this unthunk in the right place
    @destruct (α, ᾱ) = a.∂(a.ȳ, unthunk(Δ))
    (α, ∂⃖rruleB{N, O}(ᾱ, a.ȳ̄))
end

@Base.constprop :aggressive function (b::∂⃖rruleB{N, O})(Δ′...) where {N, O}
    @destruct ((Δ′′′, β), β̄) = b.ᾱ(Δ′)
    (β, ∂⃖rruleC{N, O}(b.ȳ̄, Δ′′′, β̄))
end

@Base.constprop :aggressive function (c::∂⃖rruleC{N, O})(Δ′′) where {N, O}
    @destruct (γ, γ̄) = c.ȳ̄((Δ′′, c.Δ′′′))
    (Base.tail(γ), ∂⃖rruleD{N, O}(γ̄, c.β̄))
end

@Base.constprop :aggressive function (d::∂⃖rruleD{N, O})(Δ⁴...) where {N, O}
    (δ₁, δ₂), δ̄  = d.γ̄(ZeroTangent(), Δ⁴...)
    (δ₁, ∂⃖rruleA{N, O+1}(d.β̄ , δ₂, δ̄ ))
end

# Terminal cases
@Base.constprop :aggressive function (c::∂⃖rruleB{N, N})(Δ′...) where {N}
    @destruct (Δ′′′, β) = c.ᾱ(Δ′)
    (β, ∂⃖rruleC{N, N}(c.ȳ̄, Δ′′′, nothing))
end
@Base.constprop :aggressive (c::∂⃖rruleC{N, N})(Δ′′) where {N} =
    Base.tail(c.ȳ̄((Δ′′, c.Δ′′′)))
(::∂⃖rruleD{N, N})(Δ...) where {N} = error("Should not be reached")

# ∂⃖rrule
term_depth(N) = 1<<(N-2)
function (::∂⃖rrule{N})(z, z̄) where {N}
    @destruct (y, ȳ) = z
    y, ∂⃖rruleA{term_depth(N), 1}(∂⃖{N-1}(), ȳ, z̄)
end

function (::∂⃖{N})(f::Core.IntrinsicFunction, args...) where {N}
    # A few intrinsic functions are inserted by the compiler, so they need to
    # be handled here. Otherwise, we just throw an appropriate error.
    if f === Core.Intrinsics.not_int && length(args) == 1
        return f(args...), EvenOddOdd{1, c_order(N)}(
            Δ->(NoTangent(), NoTangent()),
            Δ->NoTangent())
    end

    error("Rewrite reached intrinsic function $f. Missing rule?")
end

# The static parameter on `f` disables the compileable_sig heuristic
function (::∂⃖{N})(f::T, args...) where {T, N}
    if N == 1
        # Base case (inlined to avoid ambiguities with manually specified
        # higher order rules)
        z = rrule(DiffractorRuleConfig(), f, args...)
        if z === nothing
            return ∂⃖recurse{1}()(f, args...)
        end
        return z
    else
        ∂⃖p = ∂⃖{N-1}()
        @destruct z, z̄ = ∂⃖p(rrule, f, args...)
        if z === nothing
            return ∂⃖recurse{N}()(f, args...)
        else
            return ∂⃖rrule{N}()(z, z̄)
        end
    end
end

function ChainRulesCore.rrule_via_ad(::DiffractorRuleConfig, f::T, args...) where {T}
    Tuple{Any, Any}(∂⃖{1}()(f, args...))
end

@Base.assume_effects :total function (::∂⃖{1})(::typeof(Core.apply_type), head, args...)
    return rrule(Core.apply_type, head, args...)
end

struct KwFunc{T,S}
    f::T
    kwf::S
    function KwFunc(f)
        kwf = Core.kwfunc(f)
        new{Core.Typeof(f), Core.Typeof(kwf)}(f, kwf)
    end
end
(kw::KwFunc)(args...) = kw.kwf(args...)

function ChainRulesCore.rrule(::typeof(Core.kwfunc), f)
    KwFunc(f), Δ->(NoTangent(), Δ)
end

function ChainRulesCore.rrule(::KwFunc, kwargs, f, args...)
    r = Core.kwfunc(rrule)(kwargs, rrule, f, args...)
    if r === nothing
        return nothing
    end
    x, back = r
    x, Δ->begin
        (NoTangent(), NoTangent(), back(Δ)...)
    end
end

@Base.constprop :aggressive function ChainRulesCore.rrule(::typeof(Core.getfield), s, field::Symbol)
    getfield(s, field), let P = typeof(s)
        @Base.constprop :aggressive Δ->begin
            nt = NamedTuple{(field,)}((Δ,))
            (NoTangent(), Tangent{P, typeof(nt)}(nt), NoTangent())
        end
    end
end

struct ∂⃖getfield{n, f}; end
@Base.constprop :aggressive function (::∂⃖getfield{n, f})(Δ) where {n,f}
    if @generated
        return Expr(:call, tuple, NoTangent(),
            Expr(:call, tuple, (i == f ? :(Δ) : ZeroTangent() for i = 1:n)...),
            NoTangent())
    else
        return (NoTangent(), ntuple(i->i == f ? Δ : ZeroTangent(), n), NoTangent())
    end
end

struct EvenOddEven{O, P, F, G}; f::F; g::G; end
EvenOddEven{O, P}(f::F, g::G) where {O, P, F, G} = EvenOddEven{O, P, F, G}(f, g)
struct EvenOddOdd{O, P, F, G}; f::F; g::G; end
EvenOddOdd{O, P}(f::F, g::G) where {O, P, F, G} = EvenOddOdd{O, P, F, G}(f, g)
@Base.constprop :aggressive (o::EvenOddOdd{O, P, F, G})(Δ) where {O, P, F, G} = (o.f(Δ), EvenOddEven{O+1, P, F, G}(o.f, o.g))
@Base.constprop :aggressive (e::EvenOddEven{O, P, F, G})(Δ...) where {O, P, F, G} = (e.g(Δ...), EvenOddOdd{O+1, P, F, G}(e.f, e.g))
@Base.constprop :aggressive (o::EvenOddOdd{O, O})(Δ) where {O} = o.f(Δ)


@Base.constprop :aggressive function (::∂⃖{N})(::typeof(Core.getfield), s, field::Int) where {N}
    getfield(s, field), EvenOddOdd{1, c_order(N)}(
        ∂⃖getfield{nfields(s), field}(),
        @Base.constprop :aggressive (_, Δ, _)->getfield(Δ, field))
end

@Base.constprop :aggressive function (::∂⃖{N})(::typeof(Base.getindex), s::Tuple, field::Int) where {N}
    getfield(s, field), EvenOddOdd{1, c_order(N)}(
        ∂⃖getfield{nfields(s), field}(),
        @Base.constprop :aggressive (_, Δ, _)->lifted_getfield(Δ, field))
end

function (::∂⃖{N})(::typeof(Core.getfield), s, field::Symbol) where {N}
    getfield(s, field), let P = typeof(s)
        EvenOddOdd{1, c_order(N)}(
            (@Base.constprop :aggressive Δ->begin
                nt = NamedTuple{(field,)}((Δ,))
                (NoTangent(), Tangent{P, typeof(nt)}(nt), NoTangent())
            end),
            (@Base.constprop :aggressive (_, Δs, _)->begin
                isa(Δs, Union{ZeroTangent, NoTangent}) ? Δs : getfield(ChainRulesCore.backing(Δs), field)
            end))
    end
end

# TODO: Temporary - make better
function (::∂⃖{N})(::typeof(Base.getindex), a::Array{<:Number}, inds...) where {N}
    getindex(a, inds...), let
        EvenOddOdd{1, c_order(N)}(
            (@Base.constprop :aggressive Δ->begin
                Δ isa AbstractZero && return (NoTangent(), Δ, map(Returns(Δ), inds)...)
                BB = zero(a)
                BB[inds...] = unthunk(Δ)
                (NoTangent(), BB, map(x->NoTangent(), inds)...)
            end),
            (@Base.constprop :aggressive (_, Δ, _)->begin
                getindex(Δ, inds...)
            end))
    end
end

struct tuple_back{M}; end
(::tuple_back)(Δ::Tuple) = Core.tuple(NoTangent(), Δ...)
(::tuple_back{N})(Δ::AbstractZero) where {N} = Core.tuple(NoTangent(), ntuple(i->Δ, N)...)
(::tuple_back{N})(Δ::Tangent) where {N} = Core.tuple(NoTangent(), ntuple(i->lifted_getfield(Δ, i), N)...)
(t::tuple_back)(Δ::AbstractThunk) = t(unthunk(Δ))

function (::∂⃖{N})(::typeof(Core.tuple), args::Vararg{Any, M}) where {N, M}
    Core.tuple(args...),
        EvenOddOdd{1, c_order(N)}(
            tuple_back{M}(),
            (Δ...)->Core.tuple(Δ[2:end]...)
        )
end

struct UnApply{Spec, Types}; end
@generated function (::UnApply{Spec, Types})(Δ) where {Spec, Types}
    args = Any[NoTangent(), NoTangent(), :(Δ[1])]
    start = 2
    for (l, T) in zip(Spec, Types.parameters)
        if T <: Array
            push!(args, :([Δ[$(start:(start+l-1))]...]))
        else
            push!(args, :(Δ[$(start:(start+l-1))]))
        end
        start += l
    end
    :(Core.tuple($(args...)))
end

struct ApplyOdd{O, P}; u; ∂⃖f; end
struct ApplyEven{O, P}; u; ∂⃖∂⃖f; end
@Base.constprop :aggressive function (a::ApplyOdd{O, P})(Δ) where {O, P}
    r, ∂⃖∂⃖f = a.∂⃖f(Δ)
    (a.u(r), ApplyEven{O+1, P}(a.u, ∂⃖∂⃖f))
end
@Base.constprop :aggressive function (a::ApplyEven{O, P})(_, _, ff, args...) where {O, P}
    r, ∂⃖∂⃖∂⃖f = Core._apply_iterate(iterate, a.∂⃖∂⃖f, (ff,), args...)
    (r, ApplyOdd{O+1, P}(a.u, ∂⃖∂⃖∂⃖f))
end
@Base.constprop :aggressive function (a::ApplyOdd{O, O})(Δ) where {O}
    r = a.∂⃖f(Δ)
    a.u(r)
end

function (this::∂⃖{N})(::typeof(Core._apply_iterate), iterate, f, args::Union{Tuple, Vector, NamedTuple}...) where {N}
    @assert iterate === Base.iterate
    x, ∂⃖f = Core._apply_iterate(iterate, this, (f,), args...)
    return x, ApplyOdd{1, c_order(N)}(UnApply{map(length, args), typeof(args)}(), ∂⃖f)
end


c_order(N::Int) = 1<<N - 1

@Base.assume_effects :total function (::∂⃖{N})(::typeof(Core.apply_type), head, args...) where {N}
    Core.apply_type(head, args...), NonDiffOdd{length(args)+2, 1, c_order(N)}()
end

@Base.constprop :aggressive lifted_getfield(x, s) = getfield(x, s)
lifted_getfield(x::ZeroTangent, s) = ZeroTangent()
lifted_getfield(x::NoTangent, s) = NoTangent()

lifted_getfield(x::Tangent, s) = getproperty(x, s)

function lifted_getfield(x::Tangent{<:Tangent{T}}, s) where T
    bb = getfield(x.backing, 1)
    z = lifted_getfield(bb, s)
    z
end

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
