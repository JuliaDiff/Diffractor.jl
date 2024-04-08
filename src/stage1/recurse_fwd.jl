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

function fwd_transform(ci::CodeInfo, args...)
    newci = copy(ci)
    fwd_transform!(newci, args...)
    return newci
end

function fwd_transform!(ci::CodeInfo, mi::MethodInstance, nargs::Int, N::Int)
    new_code = Any[]
    @static if VERSION ≥ v"1.12.0-DEV.173"
        debuginfo = Core.Compiler.DebugInfoStream(mi, ci.debuginfo, length(ci.code))
        new_codelocs = Int32[]
    else
        new_codelocs = Any[]
    end
    ssa_mapping = Int[]
    loc_mapping = Int[]

    emit!(@nospecialize stmt) = stmt
    function emit!(stmt::Expr)
        stmt.head ∈ (:call, :(=), :new, :isdefined) || return stmt
        push!(new_code, stmt)
        @static if VERSION ≥ v"1.12.0-DEV.173"
            if isempty(new_codelocs)
                push!(new_codelocs, 0, 0, 0)
            else
                append!(new_codelocs, new_codelocs[end-2:end])
            end
        else
            push!(new_codelocs, isempty(new_codelocs) ? 0 : new_codelocs[end])
        end
        return SSAValue(length(new_code))
    end

    function mapstmt!(@nospecialize stmt)
        if isexpr(stmt, :(=))
            return Expr(stmt.head, emit!(mapstmt!(stmt.args[1])), emit!(mapstmt!(stmt.args[2])))
        elseif isexpr(stmt, :call)
            args = map(stmt.args) do stmt
                emit!(mapstmt!(stmt))
            end
            return Expr(:call, ∂☆{N}(), args...)
        elseif isexpr(stmt, :new)
            args = map(stmt.args) do stmt
                emit!(mapstmt!(stmt))
            end
            return Expr(:call, ∂☆new{N}(), args...)
        elseif isexpr(stmt, :splatnew)
            args = map(stmt.args) do stmt
                emit!(mapstmt!(stmt))
            end
            return Expr(:call, Core._apply_iterate, FwdIterate(DNEBundle{N}(iterate)), ∂☆new{N}(), emit!(Expr(:call, tuple, args[1])), args[2:end]...)
        elseif isa(stmt, SSAValue)
            return SSAValue(ssa_mapping[stmt.id])
        elseif isa(stmt, Core.SlotNumber)
            return SlotNumber(2 + stmt.id)
        elseif isa(stmt, Argument)
            return SlotNumber(2 + stmt.n)
        elseif isa(stmt, NewvarNode)
            return NewvarNode(SlotNumber(2 + stmt.slot.id))
        elseif isa(stmt, ReturnNode)
            return ReturnNode(emit!(mapstmt!(stmt.val)))
        elseif isa(stmt, GotoNode)
            return stmt
        elseif isa(stmt, GotoIfNot)
            return GotoIfNot(emit!(Expr(:call, primal, emit!(mapstmt!(stmt.cond)))), stmt.dest)
        elseif isexpr(stmt, :static_parameter)
            return ZeroBundle{N}(mi.sparam_vals[stmt.args[1]::Int])
        elseif isexpr(stmt, :foreigncall)
            return Expr(:call, error, "Attempted to AD a foreigncall. Missing rule?")
        elseif isexpr(stmt, :meta) || isexpr(stmt, :inbounds)  || isexpr(stmt, :loopinfo) ||
               isexpr(stmt, :code_coverage_effect)
            # Can't trust that meta annotations are still valid in the AD'd
            # version.
            return nothing
        elseif isexpr(stmt, :isdefined)
            return Expr(:call, zero_bundle{N}(), emit!(stmt))
        # Always disable `@inbounds`, as we don't actually know if the AD'd
        # code is truly `@inbounds` or not.
        elseif isexpr(stmt, :boundscheck)
            return DNEBundle{N}(true)
        else
            # Fallback case, for literals.
            # If it is an Expr, then it is not a literal
            if isa(stmt, Expr)
                error("Unexprected statement encountered. This is a bug in Diffractor. stmt=$stmt")
            end
            return Expr(:call, zero_bundle{N}(), stmt)
        end
    end

    meth = mi.def::Method
    for i = 1:meth.nargs
        if meth.isva && i == meth.nargs
            args = map(i:(nargs+1)) do j::Int
                emit!(Expr(:call, getfield, SlotNumber(2), j))
            end
            emit!(Expr(:(=), SlotNumber(2 + i), Expr(:call, ∂vararg{N}(), args...)))
        else
            emit!(Expr(:(=), SlotNumber(2 + i), Expr(:call, getfield, SlotNumber(2), i)))
        end
    end

    for (i, stmt) = enumerate(ci.code)
        push!(loc_mapping, length(new_code)+1)
        @static if VERSION ≥ v"1.12.0-DEV.173"
            append!(new_codelocs, debuginfo.codelocs[3i-2:3i])
        else
            push!(new_codelocs, ci.codelocs[i])
        end
        push!(new_code, mapstmt!(stmt))
        push!(ssa_mapping, length(new_code))
    end

    # Rewrite control flow
    for (i, stmt) in enumerate(new_code)
        if isa(stmt, GotoNode)
            new_code[i] = GotoNode(loc_mapping[stmt.label])
        elseif isa(stmt, GotoIfNot)
            new_code[i] = GotoIfNot(stmt.cond, loc_mapping[stmt.dest])
        end
    end

    ci.slotnames = Symbol[Symbol("#self#"), :args, ci.slotnames...]
    ci.slotflags = UInt8[0x00, 0x00, ci.slotflags...]
    ci.slottypes = ci.slottypes === nothing ? nothing : Any[Any, Any, ci.slottypes...]
    ci.code = new_code
    @static if VERSION ≥ v"1.12.0-DEV.173"
        empty!(debuginfo.codelocs)
        append!(debuginfo.codelocs, new_codelocs)
        ci.debuginfo = Core.DebugInfo(debuginfo, length(new_code))
    else
        ci.codelocs = new_codelocs
    end
    ci.ssavaluetypes = length(new_code)
    ci.ssaflags = UInt8[0 for i=1:length(new_code)]
    ci.method_for_inference_limit_heuristics = meth
    ci.edges = MethodInstance[mi]

    return ci
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
