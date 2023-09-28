function fwd_transform(ci, args...)
    newci = copy(ci)
    fwd_transform!(newci, args...)
    return newci
end

function fwd_transform!(ci, mi, nargs, N)
    new_code = Any[]
    new_codelocs = Any[]
    ssa_mapping = Int[]
    loc_mapping = Int[]

    emit!(@nospecialize stmt) = stmt
    function emit!(stmt::Expr)
        stmt.head ∈ (:call, :(=), :new, :isdefined) || return stmt
        push!(new_code, stmt)
        push!(new_codelocs, isempty(new_codelocs) ? 0 : new_codelocs[end])
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
            return Expr(:call, Core._apply_iterate, FwdIterate(ZeroBundle{N}(iterate)), ∂☆new{N}(), emit!(Expr(:call, tuple, args[1])), args[2:end]...)
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
            return Expr(:call, ZeroBundle{N}, emit!(stmt))
        # Always disable `@inbounds`, as we don't actually know if the AD'd
        # code is truly `@inbounds` or not.
        elseif isexpr(stmt, :boundscheck)
            return ZeroBundle{N}(true)
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

    for (stmt, codeloc) in zip(ci.code, ci.codelocs)
        push!(loc_mapping, length(new_code)+1)
        push!(new_codelocs, codeloc)
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
    ci.codelocs = new_codelocs
    ci.ssavaluetypes = length(new_code)
    ci.ssaflags = UInt8[0 for i=1:length(new_code)]
    ci.method_for_inference_limit_heuristics = meth
    ci.edges = MethodInstance[mi]

    return ci
end
