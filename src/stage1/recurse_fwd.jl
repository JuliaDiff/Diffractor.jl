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

function perform_fwd_transform(world::UInt, source::LineNumberNode,
                               @nospecialize(ff::Type{∂☆recurse{N}}), @nospecialize(args)) where {N}
    if all(x->x <: ZeroBundle, args)
        return :(∂☆passthrough(args))
    end

    # Check if we have an rrule for this function
    sig = Tuple{map(π, args)...}
    mthds = Base._methods_by_ftype(sig, -1, world)
    if mthds === nothing || length(mthds) != 1
        # Core.println("[perform_fwd_transform] ", sig, " => ", mthds)
        stub = Core.GeneratedFunctionStub(identity, Core.svec(:ff, :args), Core.svec())
        return stub(world, source, :(∂☆nomethd(args)))
    end
    match = only(mthds)::Core.MethodMatch

    mi = Core.Compiler.specialize_method(match)
    ci = Core.Compiler.retrieve_code_info(mi, world)

    ci′ = copy(ci)
    ci′.edges = MethodInstance[mi]

    transform_fwd!(ci′, mi.def, length(args) - 1, match.sparams, N)

    ci′.ssavaluetypes = length(ci′.code)
    ci′.ssaflags = UInt8[0 for i=1:length(ci′.code)]
    ci′.method_for_inference_limit_heuristics = match.method
    slotnames = Symbol[Symbol("#self#"), :args, ci.slotnames...]
    slotflags = UInt8[(0x00 for i = 1:2)..., ci.slotflags...]
    slottypes = ci.slottypes === nothing ? nothing : Any[(Any for i = 1:2)..., ci.slottypes...]
    ci′.slotnames = slotnames
    ci′.slotflags = slotflags
    ci′.slottypes = slottypes

    return ci′
end

let ex = :(function (ff::∂☆recurse)(args...)
               $(Expr(:meta, :generated_only))
               $(Expr(:meta, :generated, perform_fwd_transform))
           end)
    push!(GENERATORS, ex)
    Core.eval(@__MODULE__, ex)
end
