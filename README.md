# Diffractor - Next Generation AD

[![Build Status](https://github.com/JuliaDiff/Diffractor.jl/workflows/CI/badge.svg)](https://github.com/JuliaDiff/Diffractor.jl/actions?query=workflow:CI)
[![Coverage](https://codecov.io/gh/JuliaDiff/Diffractor.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaDiff/Diffractor.jl)

**Docs:**
[![](https://img.shields.io/badge/docs-master-blue.svg)](https://juliadiff.org/Diffractor.jl/dev)

# General Overview

Diffractor is an experimental next-generation, compiler-based AD system for Julia.

Design goals:
- Ultra high performance for both scalar and array code
- Efficient higher order derivatives through nested AD
- Reasonable compile times
- High flexibility (like Zygote)
- Support for forward/reverse/mixed modes
- Fast Jacobians

This is achieved through a combination of innovations:
- A new lowest level interface (∂⃖ the "AD optic functor" or "diffractor"), more suited to higher order AD
- New capabilities in Base Julia (Opaque closures, inference plugins)
- Better integration with ChainRules.jl
- Demand-driven forward-mode AD (Applying transforms to only those IR statements that contribute to relevant outputs of the function being differentiated)

# Current status

Diffractor is currently supported on Julia v1.10+.
While the best performance is generally achieved by running on Julia nightly due to constant compiler improvements, the current release of Diffractor is guaranteed to work on Julia v1.10.

## Current Status: Forward-Mode
Currently, forward-mode is the only fully-functional mode and is now shipping in some closed source products.
It is in a position to compete with [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl), and with [TaylorDiff.jl](https://github.com/JuliaDiff/TaylorDiff.jl).
It is not as battle-tested as ForwardDiff.jl, but it has several advantages:
Primarily, as it is not an operator overloading AD, it frees one from the need to relax type-constants and worry about the types of containers.
Furthermore, Like TaylorDiff.jl, it supports Taylor series based computation of higher order derviatives.
It directly and efficiently uses ChainRules.jl's `frules`, no need for a wrapper macro to import them etc.

One limitation over ForwardDiff.jl is a lack of chunking support, to pushforward multiple bases at once.

## Current Status: Reverse-Mode
Improved reverse mode support is planned for a future release.
While reverse mode was originally implemented and working, it has been stripped out until such a time as it can be properly implemented on top of new Julia compiler changes.<br>
⚠️ **Reverse Mode support should be considered experimental, and may break without warning, and may not be fixed rapidly.** ⚠️ <br>

With that said, issues and PRs for reverse mode continue to be appreciated.
### Status as of last time reverse mode was worked on:

The plan is to implement this in two stages:
1. Generated function based transforms, using the ChainRules, the new low level interface and Opaque closures
2. Adding inference plugins

Currently the implementation of Phase 1 is essentially complete, though mostly untested.
Experimentation is welcome, though it is probably not ready yet to be a production
AD system. The compiler parts of phase 1 are a bit "quick and dirty" as the main
point of phase 1 is to prove out that the overall scheme works. As a result, it
has known suboptimalities. I do not intend to do much work on these, since they
will be obsoleted by phase 2 anyway.

A few features are still missing, e.g. chunking and I intend to do some more work
on user friendly interfaces, but it should overall be useable as an AD system.
