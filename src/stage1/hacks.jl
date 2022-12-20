# Updated copy of the same code in Base, but with bugs fixed
using Core.Compiler: count_added_node!, NewSSAValue, add_pending!,
    StmtRange, BasicBlock

# Re-named in https://github.com/JuliaLang/julia/pull/47051
const add! = VERSION < v"1.9-" ? Core.Compiler.add! : Core.Compiler.add_inst!

Base.length(c::Core.Compiler.NewNodeStream) = Core.Compiler.length(c)
Base.setindex!(i::Instruction, args...) = Core.Compiler.setindex!(i, args...)
Core.Compiler.BasicBlock(x::UnitRange) =
    BasicBlock(StmtRange(first(x), last(x)))
Core.Compiler.BasicBlock(x::UnitRange, preds::Vector{Int}, succs::Vector{Int}) =
    BasicBlock(StmtRange(first(x), last(x)), preds, succs)
Base.size(x::Core.Compiler.UnitRange) = Core.Compiler.size(x)

function my_insert_node!(compact::IncrementalCompact, before, inst::NewInstruction, attach_after::Bool=false)
    @assert inst.effect_free_computed
    if isa(before, SSAValue)
        if before.id < compact.result_idx
            count_added_node!(compact, inst.stmt)
            line = something(inst.line, compact.result[before.id][:line])
            node = add!(compact.new_new_nodes, before.id, attach_after)
            node[:inst], node[:type], node[:line], node[:flag] = inst.stmt, inst.type, line, inst.flag
            return NewSSAValue(node.idx)
        else
            line = something(inst.line, compact.ir.stmts[before.id][:line])
            node = add_pending!(compact, before.id, attach_after)
            node[:inst], node[:type], node[:line], node[:flag] = inst.stmt, inst.type, line, inst.flag
            os = OldSSAValue(length(compact.ir.stmts) + length(compact.ir.new_nodes) + length(compact.pending_nodes))
            push!(compact.ssa_rename, os)
            push!(compact.used_ssas, 0)
            return os
        end
    elseif isa(before, OldSSAValue)
        pos = before.id
        if pos < compact.idx
            renamed = compact.ssa_rename[pos]
            count_added_node!(compact, inst.stmt)
            line = something(inst.line, compact.result[renamed.id][:line])
            node = add!(compact.new_new_nodes, renamed.id, attach_after)
            node[:inst], node[:type], node[:line], node[:flag] = inst.stmt, inst.type, line, inst.flag
            return NewSSAValue(node.idx)
        else
            if pos > length(compact.ir.stmts)
                #@assert attach_after
                info = compact.pending_nodes.info[pos - length(compact.ir.stmts) - length(compact.ir.new_nodes)]
                pos, attach_after = info.pos, info.attach_after
            end
            line = something(inst.line, compact.ir.stmts[pos][:line])
            node = add_pending!(compact, pos, attach_after)
            node[:inst], node[:type], node[:line], node[:flag] = inst.stmt, inst.type, line, inst.flag
            os = OldSSAValue(length(compact.ir.stmts) + length(compact.ir.new_nodes) + length(compact.pending_nodes))
            push!(compact.ssa_rename, os)
            push!(compact.used_ssas, 0)
            return os
        end
    elseif isa(before, NewSSAValue)
        before_entry = compact.new_new_nodes.info[before.id]
        line = something(inst.line, compact.new_new_nodes.stmts[before.id][:line])
        new_entry = add!(compact.new_new_nodes, before_entry.pos, attach_after)
        new_entry[:inst], new_entry[:type], new_entry[:line], new_entry[:flag] = inst.stmt, inst.type, line, inst.flag
        return NewSSAValue(new_entry.idx)
    else
        error("Unsupported")
    end
end
