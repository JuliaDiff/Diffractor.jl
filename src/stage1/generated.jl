using ChainRulesCore: NO_FIELDS
using Base.Experimental: @opaque

struct ∂⃖rrule{N}; end
struct ∂⃖recurse{N}; end

include("recurse.jl")

function perform_optic_transform(@nospecialize(ff::Type{∂⃖recurse{N}}), @nospecialize(args)) where {N}
    @assert N >= 1

    # Check if we have an rrule for this function
    mthds = Base._methods_by_ftype(Tuple{args...}, -1, typemax(UInt))
    @assert length(mthds) == 1
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

# This is a sort of inverse of ∂⃖rrule
@eval function (::∂⃖{1})(::∂⃖{1}, args...)
    @destruct a, b = ∂⃖{2}()(args...)
    (a, $(Expr(:new, Protected{1}, :(@opaque Δ->begin
        @destruct (x, y) = b(Δ)
        x, @opaque Δ′′->begin
            @destruct α, β = y(Δ′′...)
            (β, α)
        end
    end)))), @opaque ((Δ′′′,Δ⁴),)->begin
        # Add trivial gradient w.r.t ∂⃖{1}
        (NO_FIELDS, Δ⁴(Δ′′′)...)
    end
end

@eval function (::∂⃖{2})(::∂⃖{1}, args...)
    @destruct a, b = ∂⃖{3}()(args...)
    (a, $(Expr(:new, Protected{2}, :(@opaque Δ->begin
        @destruct (x, y) = b(Δ)
        x, @opaque Δ′′->begin
            @destruct α, β = y(Δ′′...)
            (β, α), @opaque ((Δ⁵, xx))->begin
                yy, zz = Δ⁵(xx)
                yy, Δ⁶->begin
                    aaa, bbb = zz(Δ⁶...)
                    (bbb, aaa)
                end
            end
        end
    end)))), @opaque ((Δ′′′,Δ⁴),)->begin
        (γ, δ) = Δ⁴(Δ′′′)
        # Add trivial gradient w.r.t ∂⃖{1}
        (NO_FIELDS, γ...), (Δ⁷...)->begin
            δ(Base.tail(Δ⁷)...), ((ccc, ddd),)->begin
                (NO_FIELDS, ddd(ccc)...)
            end
        end
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

function (::∂⃖{N})(::∂⃖{M}, args...) where {N, M}
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

#=
function (::∂⃖rrule{2})(z, z̄)
    @destruct (y, ȳ) = z
    @OpticBundle(y, @opaque Δ->begin
        @destruct (α, ᾱ) = ∂⃖{1}()(ȳ, Δ)
        @OpticBundle(α, @opaque (Δ′...)->begin
            @destruct (Δ′′′, β) = ᾱ(Δ′)
            @OpticBundle(β, @opaque Δ′′->begin
                # Drop gradient w.r.t. `rrule`
                return Base.tail(z̄((Δ′′, Δ′′′)))
            end)
        end)
    end)
end

function (::∂⃖rrule{3})(z, z̄)
    @destruct (y, ȳ) = z
    @OpticBundle(y, @opaque Δ->begin # A
        @destruct (α, ᾱ) = ∂⃖{2}()(ȳ, Δ)
        @OpticBundle(α, @opaque (Δ′...)->begin # B
            (Δ′′′, β), β̄  = ᾱ(Δ′)
            @OpticBundle(β, @opaque Δ′′->begin # C
                # Drop gradient w.r.t. `rrule`
                (γ, γ̄) = z̄((Δ′′, Δ′′′))
                @OpticBundle(Base.tail(γ), @opaque (Δ⁴...)->begin # D
                    (δ₁, δ₂), δ̄ = γ̄(Zero(), Δ⁴...)
                    δ₁, Δ⁵->begin #A
                        ϵ, ϵ̄ = β̄(δ₂, Δ⁵)
                        ϵ, (Δ⁶...)->begin
                            (ζ₁, ζ₂) = ϵ̄(Δ⁶)
                            ζ₂, Δ⁷->begin
                                return Base.tail(δ̄((Δ⁷, ζ₁)))
                            end
                        end
                    end
                end)
            end)
        end)
    end)
end
=#

# ∂⃖rrule has a 4-recurrence - we model this as 4 separate structs that we
# cycle between. N.B.: These names match the names that these variables
# have in Snippet 19 of the terminology guide. They are probably not ideal,
# but if you rename them here, please update the terminology guide also.

struct ∂⃖rruleA{N, O}; ∂; ȳ; ȳ̄ ; end
struct ∂⃖rruleB{N, O}; ᾱ; ȳ̄ ; end
struct ∂⃖rruleC{N, O}; ȳ̄ ; Δ′′′; β̄ ; end
struct ∂⃖rruleD{N, O}; γ̄; β̄ ; end

function (a::∂⃖rruleA{N, O})(Δ) where {N, O}
    @destruct (α, ᾱ) = a.∂(a.ȳ, Δ)
    (α, ∂⃖rruleB{N, O}(ᾱ, a.ȳ̄))
end

function (b::∂⃖rruleB{N, O})(Δ′...) where {N, O}
    @destruct ((Δ′′′, β), β̄) = b.ᾱ(Δ′)
    (β, ∂⃖rruleC{N, O}(b.ȳ̄, Δ′′′, β̄))
end

function (c::∂⃖rruleC{N, O})(Δ′′) where {N, O}
    @destruct (γ, γ̄) = c.ȳ̄((Δ′′, c.Δ′′′))
    (Base.tail(γ), ∂⃖rruleD{N, O}(γ̄, c.β̄))
end

function (d::∂⃖rruleD{N, O})(Δ⁴...) where {N, O}
    (δ₁, δ₂), δ̄  = d.γ̄(Zero(), Δ⁴...)
    (δ₁, ∂⃖rruleA{N, O+1}(d.β̄ , δ₂, δ̄ ))
end

# Terminal cases
function (c::∂⃖rruleB{N, N})(Δ′...) where {N}
    @destruct (Δ′′′, β) = c.ᾱ(Δ′)
    (β, ∂⃖rruleC{N, N}(c.ȳ̄, Δ′′′, nothing))
end
(c::∂⃖rruleC{N, N})(Δ′′) where {N} = Base.tail(c.ȳ̄((Δ′′, c.Δ′′′)))
(::∂⃖rruleD{N, N})(Δ...) where {N} = error("Should not be reached")

# ∂⃖rrule
@Base.pure term_depth(N) = 2^(N-2)
function (::∂⃖rrule{N})(z, z̄) where {N}
    @destruct (y, ȳ) = z
    y, ∂⃖rruleA{term_depth(N), 1}(∂⃖{minus1(N)}(), ȳ, z̄)
end

# The static parameter on `f` disables the compileable_sig heuristic
function (::∂⃖{N})(f::T, args...) where {T, N}
    ∂⃖p = ∂⃖{minus1(N)}()
    @destruct z, z̄ = ∂⃖p(rrule, f, args...)
    if z === nothing
        return ∂⃖recurse{N}()(f, args...)
    else
        return ∂⃖rrule{N}()(z, z̄)
    end
end

function (::∂⃖{1})(args...)
    z = rrule(args...)
    if z === nothing
        return ∂⃖recurse{1}()(args...)
    end
    return z
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

function (this::∂⃖{1})(::typeof(Core._apply_iterate), iterate, f, args::Tuple...)
    x, ∂⃖f = Core._apply_iterate(iterate, this, (f,), args...)
    return x, let u=UnApply{map(length, args)}()
        Δ->u(∂⃖f(Δ))
    end
end

@Base.pure function (::∂⃖{1})(::typeof(Core.apply_type), head, args...)
    return rrule(Core.apply_type, head, args...)
end

function (::∂⃖{1})(::typeof(Core.tuple), args...)
    return rrule(Core.tuple, args...)
end

function (::∂⃖{2})(::typeof(Core.tuple), args...)
    return Core.tuple(args...), Δ->begin
        Core.tuple(NO_FIELDS, Δ...), (Δ...)->begin
            Core.tuple(Δ[2:end]...), Δ->begin
                Core.tuple(NO_FIELDS, Δ...)
            end
        end
    end
end

function (::∂⃖{3})(::typeof(Core.tuple), args...)
    return Core.tuple(args...), Δ->begin
        Core.tuple(NO_FIELDS, Δ...), (Δ...)->begin
            Core.tuple(Δ[2:end]...), Δ->begin
                Core.tuple(NO_FIELDS, Δ...), (Δ...)->begin
                    Core.tuple(Δ[2:end]...), Δ->begin
                        Core.tuple(NO_FIELDS, Δ...), (Δ...)->begin
                            Core.tuple(Δ[2:end]...), Δ->begin
                                Core.tuple(NO_FIELDS, Δ...)
                            end
                        end
                    end
                end
            end
        end
    end
end


@Base.aggressive_constprop function ChainRulesCore.rrule(::typeof(Core.getfield), s, field::Symbol)
    getfield(s, field), let P = typeof(s)
        @Base.aggressive_constprop Δ->begin
            nt = NamedTuple{(field,)}((Δ,))
            (NO_FIELDS, Composite{P, typeof(nt)}(nt), NO_FIELDS)
        end
    end
end

@Base.aggressive_constprop function (::∂⃖{1})(::typeof(Core.getfield), s, field::Symbol)
    getfield(s, field), let P = typeof(s)
        @Base.aggressive_constprop Δ->begin
            nt = NamedTuple{(field,)}((Δ,))
            (NO_FIELDS, Composite{P, typeof(nt)}(nt), NO_FIELDS)
        end
    end
end

@Base.aggressive_constprop function (::∂⃖{2})(::typeof(Core.getfield), s, field::Symbol)
    getfield(s, field), let P = typeof(s)
        @Base.aggressive_constprop Δ->begin
            nt = NamedTuple{(field,)}((Δ,))
            (NO_FIELDS, Composite{P, typeof(nt)}(nt), NO_FIELDS), (Δ...)->begin
                getfield(s, field), Δ->begin
                    nt = NamedTuple{(field,)}((Δ,))
                    (NO_FIELDS, Composite{P, typeof(nt)}(nt), NO_FIELDS)
                end
            end
        end
    end
end

@Base.aggressive_constprop function (::∂⃖{N})(::typeof(Core.getfield), s, field::Symbol) where {N}
    error("Implement me")
end



@Base.aggressive_constprop function (::∂⃖{1})(::typeof(Core.getfield), args...)
    return rrule(Core.getfield, args...)
end



function (::∂⃖{N})(::typeof(Core.tuple), args...) where {N}
    error("TODO: Implement me")
end

@Base.pure c_order(N::Int) = 2^N - 1

@Base.pure function (::∂⃖{N})(::typeof(Core.apply_type), head, args...) where {N}
    Core.apply_type(head, args...), NonDiffOdd{plus1(plus1(length(args))), 1, c_order(N)}()
end

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
