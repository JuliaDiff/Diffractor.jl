using Core.Compiler: CodeInfo
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

# Helpers
tuple_type_fields(rt) = isa(rt, PartialStruct) ? rt.fields : widenconst(rt).parameters
