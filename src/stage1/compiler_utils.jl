# Utilities that should probably go into CC
using .Compiler: IRCode, CFG, BasicBlock, BBIdxIter

function Base.push!(cfg::CFG, bb::BasicBlock)
    @assert cfg.blocks[end].stmts.stop+1 == bb.stmts.start
    push!(cfg.blocks, bb)
    push!(cfg.index, bb.stmts.start)
end

if VERSION < v"1.11.0-DEV.258"
    Base.getindex(ir::IRCode, ssa::SSAValue) = CC.getindex(ir, ssa)
end

if VERSION < v"1.12.0-DEV.1268"
    if isdefined(CC, :Future)
        Base.isready(future::CC.Future) = CC.isready(future)
        Base.getindex(future::CC.Future) = CC.getindex(future)
        Base.setindex!(future::CC.Future, value) = CC.setindex!(future, value)
    end

    Base.iterate(c::IncrementalCompact, args...) = CC.iterate(c, args...)
    Base.iterate(p::CC.Pair, args...) = CC.iterate(p, args...)
    Base.iterate(urs::CC.UseRefIterator, args...) = CC.iterate(urs, args...)
    Base.iterate(x::CC.BBIdxIter, args...) = CC.iterate(x, args...)
    Base.getindex(urs::CC.UseRefIterator, args...) = CC.getindex(urs, args...)
    Base.getindex(urs::CC.UseRef, args...) = CC.getindex(urs, args...)
    Base.getindex(c::CC.IncrementalCompact, args...) = CC.getindex(c, args...)
    Base.setindex!(c::CC.IncrementalCompact, args...) = CC.setindex!(c, args...)
    Base.setindex!(urs::CC.UseRef, args...) = CC.setindex!(urs, args...)

    Base.copy(ir::IRCode) = CC.copy(ir)

    Base.length(c::CC.NewNodeStream) = CC.length(c)
    Base.setindex!(i::Instruction, args...) = CC.setindex!(i, args...)
    Base.size(x::CC.UnitRange) = CC.size(x)

    CC.get(a::Dict, b, c) = Base.get(a,b,c)
    CC.haskey(a::Dict, b) = Base.haskey(a, b)
    CC.setindex!(a::Dict, b, c) = setindex!(a, b, c)
end

CC.NewInstruction(@nospecialize node) =
    NewInstruction(node, Any, CC.NoCallInfo(), nothing, CC.IR_FLAG_REFINED)

Base.setproperty!(x::CC.Instruction, f::Symbol, v) = CC.setindex!(x, v, f)

Base.getproperty(x::CC.Instruction, f::Symbol) = CC.getindex(x, f)

function Base.setindex!(ir::IRCode, ni::NewInstruction, i::Int)
    stmt = ir.stmts[i]
    stmt.inst = ni.stmt
    stmt.type = ni.type
    stmt.flag = something(ni.flag, 0)  # fixes 1.9?
    @static if VERSION ≥ v"1.12.0-DEV.173"
        stmt.line = something(ni.line, CC.NoLineUpdate)
    else
        stmt.line = something(ni.line, 0)
    end
    return ni
end

function Base.push!(ir::IRCode, ni::NewInstruction)
    # TODO: This should be a check in insert_node!
    @assert length(ir.new_nodes.stmts) == 0
    @static if isdefined(CC, :add!)
        # Julia 1.7 & 1.8
        ir[CC.add!(ir.stmts)] = ni
    else
        # Re-named in https://github.com/JuliaLang/julia/pull/47051
        ir[CC.add_new_idx!(ir.stmts)] = ni
    end
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

Base.lastindex(x::CC.InstructionStream) =
    CC.length(x)

"""
    find_end_of_phi_block(ir::IRCode, start_search_idx::Int)

Finds the last index within the same basic block, on or after the `start_search_idx` which is not within a phi block.
A phi-block is a run on PhiNodes or nothings that must be the first statements within the basic block.

If `start_search_idx` is not within a phi block to begin with, then just returns `start_search_idx`
"""
function find_end_of_phi_block(ir::IRCode, start_search_idx::Int)
    # Short-cut for early exit:
    stmt = ir.stmts[start_search_idx][:inst]
    stmt !== nothing && !isa(stmt, PhiNode) && return start_search_idx

    # Actually going to have to go digging throught the IR to out if were are in a phi block
    bb=CC.block_for_inst(ir.cfg, start_search_idx)
    end_search_idx=ir.cfg.blocks[bb].stmts[end]
    for idx in (start_search_idx):(end_search_idx-1)
        stmt = ir.stmts[idx+1][:inst]
        # next statment is no longer in a phi block, so safe to insert
        stmt !== nothing && !isa(stmt, PhiNode) && return idx
    end
    return end_search_idx
end

function replace_call!(ir::IRCode, idx::SSAValue, new_call::Expr)
    ir[idx][:inst] = new_call
    ir[idx][:type] = Any
    ir[idx][:info] = CC.NoCallInfo()
    ir[idx][:flag] = CC.IR_FLAG_REFINED
end
