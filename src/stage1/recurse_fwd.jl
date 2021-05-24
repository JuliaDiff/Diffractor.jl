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

function transform_fwd!(ci, meth, nargs, sparams, N)
    new_code = Any[]
    new_codelocs = Any[]
    ssa_mapping = Int[]

    function emit!(stmt)
        (isexpr(stmt, :call) || isexpr(stmt, :(=)) || isexpr(stmt, :new)) || return stmt
        push!(new_code, stmt)
        push!(new_codelocs, isempty(new_codelocs) ? 0 : new_codelocs[end])
        SSAValue(length(new_code))
    end

    function mapstmt!(stmt)
        if isexpr(stmt, :(=))
            return Expr(stmt.head, emit!(mapstmt!(stmt.args[1])), emit!(mapstmt!(stmt.args[2])))
        elseif isexpr(stmt, :call)
            args = map(stmt.args) do stmt
                emit!(mapstmt!(stmt))
            end
            return Expr(:call, ∂☆{N}(), args...)
        elseif isexpr(stmt, :new)
            args = map(stmt.args[2:end]) do stmt
                emit!(mapstmt!(stmt))
            end
            return Expr(:call, ∂☆new{N}(), stmt.args[1], args...)
        elseif isa(stmt, SSAValue)
            return SSAValue(ssa_mapping[stmt.id])
        elseif isa(stmt, Core.SlotNumber)
            return SlotNumber(2 + stmt.id)
        elseif isa(stmt, Argument)
            return SlotNumber(2 + stmt.n)
        elseif isa(stmt, ReturnNode)
            return ReturnNode(emit!(mapstmt!(stmt.val)))
        elseif isa(stmt, GotoNode)
            return stmt
        elseif isa(stmt, GotoIfNot)
            return GotoIfNot(emit!(Expr(:call, primal, stmt.cond)), stmt.dest)
        elseif isexpr(stmt, :static_parameter)
            return ZeroBundle{N}(sparams[stmt.args[1]])
        else
            return Expr(:call, ZeroBundle{N}, stmt)
        end
    end

    for i = 1:meth.nargs
        if meth.isva && i == meth.nargs
            args = map(i:(nargs+1)) do j
                emit!(Expr(:call, getfield, SlotNumber(2), j))
            end
            emit!(Expr(:(=), SlotNumber(2 + i), Expr(:call, ∂vararg{N}(), args...)))
        else
            emit!(Expr(:(=), SlotNumber(2 + i), Expr(:call, getfield, SlotNumber(2), i)))
        end
    end

    for (stmt, codeloc) in zip(ci.code, ci.codelocs)
        push!(new_codelocs, codeloc)
        push!(new_code, mapstmt!(stmt))
        push!(ssa_mapping, length(new_code))
    end

    ci.code = new_code
    ci.codelocs = new_codelocs
    ci
end

π(::Type{<:AbstractTangentBundle{N, B}} where N) where {B} = B

function perform_fwd_transform(@nospecialize(ff::Type{∂☆recurse{N}}), @nospecialize(args)) where {N}
    # Check if we have an rrule for this function
    sig = Tuple{map(π, args)...}
    mthds = Base._methods_by_ftype(sig, -1, typemax(UInt))
    if length(mthds) != 1
        error()
    end
    match = mthds[1]

    mi = Core.Compiler.specialize_method(match)
    ci = Core.Compiler.retrieve_code_info(mi)

    ci′ = copy(ci)
    ci′.edges = MethodInstance[mi]

    transform_fwd!(ci′, mi.def, length(args) - 1, match.sparams, N)

    ci′.ssavaluetypes = length(ci′.code)
    ci′.method_for_inference_limit_heuristics = match.method
    slotnames = Symbol[Symbol("#self#"), :args, ci.slotnames...]
    slotflags = UInt8[(0x00 for i = 1:2)..., ci.slotflags...]
    slottypes = Any[(Any for i = 1:2)..., ci.slotflags...]
    ci′.slotnames = slotnames
    ci′.slotflags = slotflags
    ci′.slottypes = slottypes
    ci′
end

@eval function (ff::∂☆recurse)(args...)
    $(Expr(:meta, :generated_only))
    $(Expr(:meta,
            :generated,
            Expr(:new,
                Core.GeneratedFunctionStub,
                :perform_fwd_transform,
                Any[:ff, :args],
                Any[],
                @__LINE__,
                QuoteNode(Symbol(@__FILE__)),
                true)))
end
