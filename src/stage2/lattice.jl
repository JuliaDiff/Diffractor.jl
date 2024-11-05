using Core.Compiler: CallInfo, CallMeta
import Core.Compiler: widenconst

struct CompClosure; opaque; end # TODO: Is this a YAKC?
(::CompClosure)(x) = error("Hello")

# TODO: Are these lattice elements or info components?
struct AbstractCompClosure
    order::Int
    seq::Int
    primal_info::Any
    prev_seq_infos::Vector
end

function Base.show(io::IO, cc::AbstractCompClosure)
    print(io, "∂⃖", superscript(cc.order), subscript(cc.seq))
    if isa(cc.primal_info, MethodMatchInfo) &&
       length(cc.primal_info.results.matches) == 1
        match = cc.primal_info.results.matches[1]
        print(io, match.method.name)
    else
        printstyled(io, "!!!", typeof(cc.primal_info), color=:red)
    end
end

widenconst(::AbstractCompClosure) = Core.OpaqueClosure

struct RRuleInfo
    rrule_rt
    info
end

struct CompClosInfo
    clos::AbstractCompClosure
    infos::Vector
end

struct PrimClosInfo
    next::Any
end

struct PrimClosure
    name::Symbol
    order::Int
    seq::Int
    dual::Any
    info_below::Any
    info_carried::Any
end
widenconst(::PrimClosure) = CompClosure


function Base.show(io::IO, pc::PrimClosure)
    printstyled(IOContext(io, :color=>true), "∂⃖", superscript(pc.order), subscript(pc.seq), pc.name; color=:green)
end

struct RecurseInfo
    info
end

struct ReifyInfo
    info
end

# Forward mode info
struct FRuleCallInfo <: CallInfo
    info::CallInfo
    frule_call::CallMeta
    function FRuleCallInfo(@nospecialize(info::CallInfo), frule_call::CallMeta)
        new(info, frule_call)
    end
end
CC.nsplit_impl(info::FRuleCallInfo) = CC.nsplit(info.info)
CC.getsplit_impl(info::FRuleCallInfo, idx::Int) = CC.getsplit(info.info, idx)
CC.getresult_impl(info::FRuleCallInfo, idx::Int) = CC.getresult(info.info, idx)
if isdefined(CC, :add_uncovered_edges_impl)
    CC.add_uncovered_edges_impl(edges::Vector{Any}, info::FRuleCallInfo, @nospecialize(atype)) = CC.add_uncovered_edges!(edges, info.info, atype)
end
if isdefined(CC, :add_edges_impl)
    function CC.add_edges_impl(edges::Vector{Any}, info::FRuleCallInfo)
        CC.add_edges!(edges, info.info)
        CC.add_edges!(edges, info.frule_call.info)
    end
end

function Base.show(io::IO, info::FRuleCallInfo)
    print(io, "FRuleCallInfo(", typeof(info.info), ", ", typeof(info.frule_call.info), ")")
end

function Cthulhu.process_info(interp::AbstractInterpreter, info::FRuleCallInfo, argtypes::Cthulhu.ArgTypes, @nospecialize(rt), optimize::Bool, @nospecialize(exct))
    return Cthulhu.process_info(interp, info.info, argtypes, rt, optimize, exct)
end


# Helpers
tuple_type_fields(rt) = isa(rt, PartialStruct) ? rt.fields : widenconst(rt).parameters
