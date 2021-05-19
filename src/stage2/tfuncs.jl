function backwards_tfunc(@nospecialize(f), primal::IRCode, inst::Expr, @nospecialize(Δ))
    if f === Core.tuple
        return tuple_tfunc(Any[NoTangent, tuple_type_fields(Δ)...])
    elseif f == Core.getfield
        tt = argextype(inst.args[2], primal, primal.sptypes)
        ot = widenconst(tt)
        if ot <: Tuple
            if length(ot.parameters) == 1
                # Only one choice
                rt = tuple_tfunc(Any[Δ])
            else
                idxt = argextype(inst.args[3], primal, primal.sptypes)
                isa(idxt, Const) || error()
                args = Any[ZeroTangent for i = 1:length(ot.parameters)]
                args[idxt.val] = Δ
                rt = tuple_tfunc(args)
            end
        else
            idxt = argextype(inst.args[3], primal, primal.sptypes)
            if !isa(idxt, Const)
                @show idxt
                error()
            end
            if isa(idxt.val, Symbol)
                symn = idxt.val
            elseif isa(idxt.val, Int)
                @show tt
                symn = fieldname(tt, idxt.val)
            else
                error()
            end
            rt = NamedTuple{(symn,), Tuple{widenconst(Δ)}}
        end
        return tuple_tfunc(Any[NoTangent, rt, NoTangent, NoTangent])
    elseif f == Core.apply_type
        return tuple_tfunc(Any[NoTangent for i = 1:length(inst.args)])
    elseif f == Core.typeof
        return tuple_tfunc(Any[ZeroTangent for i = 1:length(inst.args)])
    elseif f == (===)
        return tuple_tfunc(Any[ZeroTangent for i = 1:length(inst.args)])
    elseif f == nfields
        return tuple_tfunc(Any[ZeroTangent for i = 1:length(inst.args)])
    elseif f == fieldtype
        return tuple_tfunc(Any[ZeroTangent for i = 1:length(inst.args)])
    elseif f === Core._apply_iterate
        ft = argextype(inst.args[3], primal, primal.sptypes)
        f = singleton_type(ft)
        @assert f === Core.tuple
        return backwards_tfunc(f, primal, inst, Δ)
    end
    @show f
    error()
end

function forward_tfunc(@nospecialize(f), primal::IRCode, inst::Expr, @nospecialize(Δ))
    if f === Core.tuple
        return tuple_tfunc(Any[tuple_type_fields(Δ)[2:end]...])
    elseif f == Core.getfield
        idxt = argextype(inst.args[3], primal, primal.sptypes)
        isa(idxt, Const) || error()
        val = getfield_tfunc(Δ, Const(2))
        fieldval = (isa(val, Const) && (isa(val.val, ZeroTangent) || isa(val.val, NoTangent))) ?
            Const(ZeroTangent()) : getfield_tfunc(val, idxt)
        return fieldval
    elseif f === Core.apply_type || f === Core.typeof
        return NoTangent
    end
    @show f
    error()
end

function getfield_prop_zero_tfunc(@nospecialize(s), @nospecialize(x))
    if isa(s, Const)
        if isa(s.val, NoTangent) || isa(s.val, ZeroTangent)
            return s
        end
    end
    if s === ZeroTangent || s === NoTangent
        return s
    end
    return getfield_tfunc(s, x)
end
