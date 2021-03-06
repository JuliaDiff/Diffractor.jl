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
struct Protected
    a
end
(p::Protected)(args...) = getfield(p, :a)(args...)[1]
@Base.aggressive_constprop (::∂⃖{1})(p::Protected, args...) = getfield(p, :a)(args...)

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

#=
# This is inlined directly below for inference performance
function (::∂⃖rrule{2})(z, z̄)
    (y, ȳ) = z
    OpticBundle(y, @opaque Δ->begin
        (α, ᾱ) = ∂⃖(ȳ, Δ)
        α, @opaque Δ′->begin
            (Δ′′′, β) = ᾱ(Δ′)
            β, @opaque Δ′′->begin
                # Drop gradient w.r.t. `rrule`
                (_, a′, x′) = z̄((Δ′′, Δ′′′))
                return (a′, x′)
            end
        end
    end)
end
=#

# This is a sort of inverse of ∂⃖rrule
@eval function (::∂⃖{1})(::∂⃖{1}, args...)
    @destruct a, b = ∂⃖{2}()(args...)
    (a, $(Expr(:new, Protected, :(@opaque Δ->begin
        @destruct x, y = b(Δ)
        x, @opaque Δ′′->begin
            @destruct α, β = y(Δ′′)
            (β, α)
        end
    end)))), @opaque ((Δ′′′,Δ⁴),)->begin
        # Add trivial gradient w.r.t ∂⃖{1}
        (NO_FIELDS, Δ⁴(Δ′′′)...)
    end
end


function (::∂⃖rrule{N})(z, z̄) where N
    error("Not implemented yet")
end

macro OpticBundle(a, b)
    aa = gensym()
    esc(quote
        $aa = $a
        $(Expr(:new, :(OpticBundle{typeof($aa)}), aa, b))
    end)
end



# The static parameter on `f` disables the compileable_sig heuristic
function (::∂⃖{N})(f::T, args...) where {T, N}
    @assert N === 2 # TODO: generalize
    @destruct z, z̄ = ∂⃖{minus1(N)}()(rrule, f, args...)
    if z === nothing
        return ∂⃖recurse{N}()(f, args...)
    else
        # return ∂⃖rrule{N}()(z, z̄)
        # ∂⃖rrule{2}, but inlined manually
        @destruct (y, ȳ) = z
        @OpticBundle(y, @opaque Δ->begin
            @destruct (α, ᾱ) = ∂⃖(ȳ, Δ)
            @OpticBundle(α, @opaque Δ′->begin
                @destruct (Δ′′′, β) = ᾱ(Δ′)
                @OpticBundle(β, @opaque Δ′′->begin
                    # Drop gradient w.r.t. `rrule`
                    @destruct (_, a′, x′) = z̄((Δ′′, Δ′′′))
                    return (a′, x′)
                end)
            end)
        end)
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
    args = Any[]
    start = 1
    for l in spec
        push!(args, :(Δ[$(start:(start+l-1))]))
        start += l
    end
    :(Core.tuple(args...))
end

function (this::∂⃖{1})(::typeof(Core._apply_iterate), iterate, f, args...) where N
    x, ∂⃖f = Core._apply_iterate(iterate, this, (f,), args...)
    return x, UnApply{map(length, args)}()
end

@Base.pure function (::∂⃖{1})(::typeof(Core.apply_type), head, args...)
    return rrule(Core.apply_type, head, args...)
end

@Base.aggressive_constprop function (::∂⃖{1})(::typeof(Core.getfield), args...)
    return rrule(Core.getfield, args...)
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
