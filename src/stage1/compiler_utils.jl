# Utilities that should probably go into Core.Compiler
using Core.Compiler: CFG, BasicBlock, BBIdxIter

function Base.push!(cfg::CFG, bb::BasicBlock)
    @assert cfg.blocks[end].stmts.stop+1 == bb.stmts.start
    push!(cfg.blocks, bb)
    push!(cfg.index, bb.stmts.start)
end

function Core.Compiler.NewInstruction(node)
    Core.Compiler.NewInstruction(node, Any)
end

function Base.setproperty!(x::Core.Compiler.Instruction, f::Symbol, v)
    Core.Compiler.setindex!(x, v, f)
end

function Base.getproperty(x::Core.Compiler.Instruction, f::Symbol)
    Core.Compiler.getindex(x, f)
end

function Base.setindex!(ir::Core.Compiler.IRCode, ni::NewInstruction, i::Int)
    stmt = ir.stmts[i]
    stmt.inst = ni.stmt
    stmt.type = ni.type
    stmt.flag = ni.flag
    stmt.line = something(ni.line, 0)
    ni
end

function Base.push!(ir::IRCode, ni::NewInstruction)
    # TODO: This should be a check in insert_node!
    @assert length(ir.new_nodes.stmts) == 0
    ir[Core.Compiler.add!(ir.stmts)] = ni
    ir
end

function Base.iterate(it::Iterators.Reverse{BBIdxIter},
        (bb, idx)::Tuple{Int, Int}=(length(it.itr.ir.cfg.blocks), length(it.itr.ir.stmts)+1))
    idx == 1 && return nothing
    active_bb = it.itr.ir.cfg.blocks[bb]
    if idx == first(active_bb.stmts)
        bb -= 1
    end
    return (bb, idx - 1), (bb, idx - 1)
end

Base.lastindex(x::Core.Compiler.InstructionStream) =
    Core.Compiler.length(x)
