if :recursion_relation in fieldnames(Method)

first(methods(Diffractor.∂⃖recurse{1}())).recursion_relation = function(method1, method2, parent_sig, new_sig)
    # Recursion from a higher to a lower order is always allowed
    parent_order = parent_sig.parameters[1].parameters[1]
    child_order = new_sig.parameters[1].parameters[1]
    #@Core.Main.Base.show (parent_order, child_order)
    if parent_order > child_order
        return true
    end
    wrapped_parent_sig = Tuple{parent_sig.parameters[2:end]...}
    wrapped_new_sig = Tuple{parent_sig.parameters[2:end]...}
    if method2 !== nothing && isdefined(method2, :recursion_relation)
        # TODO: What if method2 is itself a generated function.
        return method2.recursion_relation(method2, nothing, wrapped_parent_sig, wrapped_new_sig)
    end
    return Core.Compiler.type_more_complex(new_sig, parent_sig, Core.svec(parent_sig), 1, 3, length(method1.sig.parameters)+1)
end

first(methods(PrimeDerivativeBack(sin))).recursion_relation = function(method1, method2, parent_sig, new_sig)
    # Recursion from a higher to a lower order is always allowed
    parent_order = parent_sig.parameters[1].parameters[1]
    child_order = new_sig.parameters[1].parameters[1]
    #@Core.Main.Base.show (parent_order, child_order)
    if parent_order > child_order
        return true
    end
    wrapped_parent_sig = Tuple{parent_sig.parameters[2:end]...}
    wrapped_new_sig = Tuple{parent_sig.parameters[2:end]...}
    return Core.Compiler.type_more_complex(new_sig, parent_sig, Core.svec(parent_sig), 1, 3, length(method1.sig.parameters)+1)
end

which(Tuple{∂⃖{N}, T, Vararg{Any}} where {T,N}).recursion_relation = function(_, _, parent_sig, new_sig)
    # Any actual recursion will always be caught be one of the functions we're
    # recursing into.
    return isa(Base.unwrap_unionall(parent_sig.parameters[1].parameters[1]), Int) &&
           isa(Base.unwrap_unionall(new_sig.parameters[1].parameters[1]), Int)
end

which(Tuple{∂⃖{N}, ∂⃖{1}, Vararg{Any}} where {N}).recursion_relation = function(_, _, parent_sig, new_sig)
    # Allowed as long as both parent and new sig have concrete integers. In that
    # case, actual recursion will be caught elsewhere.
    return isa(Base.unwrap_unionall(parent_sig.parameters[1].parameters[1]), Int) &&
           isa(Base.unwrap_unionall(new_sig.parameters[1].parameters[1]), Int)
end

for (;method) in Base._methods_by_ftype(Tuple{Diffractor.∂☆recurse{N}, Vararg{Any}} where {N}, nothing, -1, Base.get_world_counter())
    method.recursion_relation = function (method1, method2, parent_sig, new_sig)
        # Recursion from a higher to a lower order is always allowed
        parent_order = parent_sig.parameters[1].parameters[1]
        child_order = new_sig.parameters[1].parameters[1]
        if parent_order > child_order
            return true
        end
        #Core.Compiler.@show (parent_sig, new_sig)
        return false
    end
end

for (;method) in Base._methods_by_ftype(Tuple{Diffractor.∂☆internal{N}, Vararg{Any}} where {N}, nothing, -1, Base.get_world_counter())
    method.recursion_relation = function (method1, method2, parent_sig, new_sig)
        return true
    end
end

for (;method) in Base._methods_by_ftype(Tuple{Diffractor.∂☆{N}, Vararg{Any}} where {N}, nothing, -1, Base.get_world_counter())
    method.recursion_relation = function (method1, method2, parent_sig, new_sig)
        return true
    end
end

end
