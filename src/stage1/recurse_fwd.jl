struct ∂☆recurse{N}; end

struct ∂vararg{N}; end

(::∂vararg{N})() where {N} = ZeroBundle{N}(())
function (::∂vararg{N})(a::AbstractTangentBundle{N}...) where N
    B = Tuple{map(x->basespace(Core.Typeof(x)), a)...}
    return (∂☆new{N}())(B, a...)
end

struct ∂☆new{N}; end

# we split out the 1st order derivative as a special case for performance
# but the nth order case does also work for this
function (::∂☆new{1})(B::Type, xs::AbstractTangentBundle{1}...)
    @info "∂☆new{1}" B supertype(B) typeof(xs)
    primal_args = map(primal, xs)
    the_primal = _construct(B, primal_args)
    tangent_tup = map(first_partial, xs)
    the_partial = if B<:Tuple
        Tangent{B, typeof(tangent_tup)}(tangent_tup)
    else
        names = fieldnames(B)
        tangent_nt = NamedTuple{names}(tangent_tup)
        StructuralTangent{B}(tangent_nt)
    end
    the_final_partial = maybe_construct_natural_tangent(B, the_partial)
    B2 = typeof(the_primal)  # HACK: if the_primal actually has types in it then we want to make sure we get DataType not Type(...)
    return TaylorBundle{1, B2}(the_primal, (the_final_partial,))
end

function (::∂☆new{N})(B::Type, xs::AbstractTangentBundle{N}...) where {N}
    primal_args = map(primal, xs)
    the_primal = _construct(B, primal_args)
    the_partials = ntuple(Val{N}()) do ii
        tangent_tup = map(x->partial(x, ii), xs)
        tangent = if B<:Tuple
            Tangent{B, typeof(tangent_tup)}(tangent_tup)
        else
            # No matter the order we use `StructuralTangent{B}` for the partial
            # It follows all required properties of the tangent to the n-1th order tangent
            names = fieldnames(B)
            tangent_nt = NamedTuple{names}(tangent_tup)
            StructuralTangent{B}(tangent_nt)
        end

        return maybe_construct_natural_tangent(B, tangent)
    end
    return TaylorBundle{N, B}(the_primal, the_partials)
end

_construct(::Type{B}, args) where B<:Tuple = B(args)
# Hack for making things that do not have public constructors constructable:
@generated _construct(B::Type, args) = Expr(:splatnew, :B, :args)


maybe_construct_natural_tangent(::Type, structural_tangent) = structural_tangent
for BaseSpaceType in (Number, AbstractArray{<:Number})
    @eval function maybe_construct_natural_tangent(::Type{B}, structural_tangent) where B<:$BaseSpaceType
        try
            # TODO: should this use `_construct` ?
            # TODO: is this right?
            unwrap_tup(x::Tangent{<:Tuple}) = ChainRulesCore.backing(x)
            unwrap_tup(x) = x
            field_tangents = map(unwrap_tup, structural_tangent)
            B(field_tangents...)
        catch       
            error(
                "`struct` types that subtype `$($BaseSpaceType)` are generally expected to provide default constructors (one arg per field), and to be usable as their own tangent type.\n" *
                "If they are not please overload the frule for the constructor: `ChainRulesCore.frule((_, dargs...), ::Type{$B}, args...)` " * 
                "and make it return whatever tangent type you want. But be warned this is off the beaten track. Here be dragons 🐉"
            )
        end
    end
end


@generated (::∂☆new{N})(B::Type) where {N} = return :(zero_bundle{$N}()($(Expr(:new, :B))))

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
    argtypes = Any[Core.Typeof(primal(arg)) for arg in args]
    tt = Base.signature_type(f, argtypes)
    sig = Base.sprint(Base.show_tuple_as_call, Symbol(""), tt)
    throw(DomainError(f, "No `ChainRulesCore.frule` found for the built-in function `$sig`"))
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
