struct ∂☆recurse{N}; end

struct ∂vararg{N}; end

(::∂vararg{N})() where {N} = ZeroBundle{N}(())
function (::∂vararg{N})(a::AbstractTangentBundle{N}...) where N
    CompositeBundle{N, Tuple{map(x->basespace(typeof(x)), a)...}}(a)
end

struct ∂☆new{N}; end

(::∂☆new{N})(B::Type, a::AbstractTangentBundle{N}...) where {N} =
    CompositeBundle{N, B}(a)

@generated (::∂☆new{N})(B::Type) where {N} = return :(ZeroBundle{$N}($(Expr(:new, :B))))

# Sometimes we don't know whether or not we need to the ZeroBundle when doing
# the transform, so this can happen - allow it for now.
(this::∂☆new{N})(B::ATB{N, <:Type}, args::ATB{N}...) where {N} = this(primal(B), args...)


π(::Type{<:AbstractTangentBundle{N, B}} where N) where {B} = B

∂☆passthrough(args::Tuple{Vararg{ATB{N}}}) where {N} =
    ZeroBundle{N}(primal(getfield(args, 1))(map(primal, Base.tail(args))...))

function ∂☆nomethd(@nospecialize(args))
    throw(MethodError(primal(args[1]), map(primal, Base.tail(args))))
end
function ∂☆builtin((f_bundle, args...))
    f = primal(f_bundle)
    throw(DomainError(f, "No `ChainRulesCore.frule` found for the built-in function `$f`"))
end

function perform_fwd_transform(world::UInt, source::LineNumberNode,
                               @nospecialize(ff::Type{∂☆recurse{N}}), @nospecialize(args)) where {N}
    if all(x->x <: ZeroBundle, args)
        return generate_lambda_ex(world, source,
            Core.svec(:ff, :args), Core.svec(), :(∂☆passthrough(args)))
    end

    sig = Tuple{map(π, args)...}
    if sig.parameters[1] <: Core.Builtin
        return generate_lambda_ex(world, source,
            Core.svec(:ff, :args), Core.svec(), :(∂☆builtin(args)))
    end
    
    mthds = Base._methods_by_ftype(sig, -1, world)
    if mthds === nothing || length(mthds) != 1
        # Core.println("[perform_fwd_transform] ", sig, " => ", mthds)
        return generate_lambda_ex(world, source,
            Core.svec(:ff, :args), Core.svec(), :(∂☆nomethd(args)))
    end
    match = only(mthds)::Core.MethodMatch

    mi = Core.Compiler.specialize_method(match)
    ci = Core.Compiler.retrieve_code_info(mi, world)

    return fwd_transform(ci, mi, length(args)-1, N)
end

@eval function (ff::∂☆recurse)(args...)
    $(Expr(:meta, :generated_only))
    $(Expr(:meta, :generated, perform_fwd_transform))
end
