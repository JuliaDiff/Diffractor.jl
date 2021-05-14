struct ∂⃗recurse{N}; end

function transform_fwd!(ci, meth, nargs, sparams, N)
    new_code = Any[]
    new_codelocs = Any[]
    ssa_mapping = Int[]

    function emit!(stmt)
        isexpr(stmt, :call) || return stmt
        push!(new_code, stmt)
        push!(new_codelocs, new_codelocs[end])
        SSAValue(length(new_code))
    end

    function mapstmt!(stmt)
        if isexpr(stmt, :(=))
            return Expr(stmt.head, mapstmt!(stmt.args[1]), emit!(mapstmt!(stmt.args[2])))
        elseif isexpr(stmt, :call)
            args = map(stmt.args) do stmt
                emit!(mapstmt!(stmt))
            end
            return Expr(:call, ∂⃗{N}(), args...)
        elseif isa(stmt, SSAValue)
            return SSAValue(ssa_mapping[stmt.id])
        elseif isa(stmt, Core.SlotNumber)
            if stmt.id <= meth.nargs
                return Expr(:call, getfield, Argument(2), stmt.id)
            else
                return SlotNumber(stmt.id - (meth.nargs - 2))
            end
        elseif isa(stmt, Argument)
            return Expr(:call, getfield, Argument(2), stmt.n)
        elseif isa(stmt, ReturnNode)
            return ReturnNode(mapstmt!(stmt.val))
        else
            return Expr(:call, ZeroBundle{N}, stmt)
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

function perform_fwd_transform(@nospecialize(ff::Type{∂⃗recurse{N}}), @nospecialize(args)) where {N}
    # Check if we have an rrule for this function
    mthds = Base._methods_by_ftype(Tuple{map(π, args)...}, -1, typemax(UInt))
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

    transform_fwd!(ci′, mi.def, length(args) - 1, match.sparams, N)

    ci′.ssavaluetypes = length(ci′.code)
    ci′.method_for_inference_limit_heuristics = match.method
    slotnames = Symbol[Symbol("#self#"), :args, ci.slotnames[(match.method.nargs+1):end]...]
    slotflags = UInt8[(0x00 for i = 1:2)..., ci.slotflags[(match.method.nargs+1):end]...]
    slottypes = Any[(Any for i = 1:2)..., ci.slotflags[(match.method.nargs+1):end]...]
    ci′.slotnames = slotnames
    ci′.slotflags = slotflags
    ci′.slottypes = slottypes
    ci′
end

@eval function (ff::∂⃗recurse)(args...)
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
