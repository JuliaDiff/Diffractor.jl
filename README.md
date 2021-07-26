# Diffractor - Next Generation AD

# General Overview

Diffractor is an experimental next-generation, compiler-based AD system for Julia.
Its public interface should be familiar to users, essentially matching Zygote.

Design goals:
- Ultra high performance for both scalar and array code
- Efficient higher order derivatives
- Reasonable compile times
- High flexibility (like Zygote)
- Support for forward/reverse/mixed modes

This is achieved through a combination of innovations:
- A new lowest level interface (∂⃖ the "AD optic functor" or "diffractor"), more suited to higher order AD
- New capabilities in Base Julia (Opaque closures, inference plugins)
- Better integration with ChainRules.jl

# Current Status

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
