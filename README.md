# Diffractor - Next Generation AD

Diffractor is an experimental next-generation, compiler-based AD system for Julia.
Its public interface should be fimilar to users, essentially matching Zygote.

Design goals:
- Ultra high performance for both scalar and array code
- Efficient higher order derivatives
- Reasonable compile times
- High flexibility (like Zygote)

This is achieved through a combination of innovations:
- A new lowest level interface more suited to higher order AD (∂⃖ the "AD optic functor" or "diffractor"),
more suited to higher order AD
- New capabilities in Base Julia (Opaque closures, inference plugins)
- Better integration with ChainRules.jl

The plan is to implement this in two stages:
1. Generated function based transforms, using the ChainRules, the new low level interface and Opaque closures
2. Adding inference plugins

After stage 1, Diffractor should hopefully match Zyogte in terms us usability,
while having slightly better performance. Stage 2 is required for full performance,
but should not require any interface changes.

Work on both stages is proceeding simultaneously, but stage 2 will depend on
additional features in Julia base that may take additional time to implement.
