function transform_fwd!(ci, meth, nargs, sparams, N)
    new_code = Any[]
    new_codelocs = Any[]
    ssa_mapping = Int[]
    loc_mapping = Int[]

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
            return ZeroBundle{N}(sparams[stmt.args[1]])
        elseif isexpr(stmt, :foreigncall)
            return Expr(:call, error, "Attempted to AD a foreigncall. Missing rule?")
        elseif isexpr(stmt, :meta) || isexpr(stmt, :inbounds)
            # Can't trust that meta annotations are still valid in the AD'd
            # version.
            return nothing
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

    ci.code = new_code
    ci.codelocs = new_codelocs
    ci
end
