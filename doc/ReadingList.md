This is a list of references I found useful while thinking about Diffractor.
If you are new to Julia, AD or Diffractor and are primarily intersted in
Diffractor, how it works, how to use it, or even there general Diffractor
theory, this is probably not the list for you. As always in the literature,
some of these references use terms differently from how they are used in
Diffractor (as well as being inconsistent with each other). Additionally,
many of these references are quite dense and though I've found small nuggets
of insight in each, excavating those took many hours. Also, these are not
introductory texts. If you've not taken an introductory differential
geometry course, I would recommend looking for that first. Don't feel bad if
some of these references read like like nonsense. It often reads that way to me to.

# Reading on Optics

- "Categories of Optics" - Mitchell Riley - https://arxiv.org/abs/1809.00738

The original paper on optics. Good background for understanding the optics terminology.

# Readings on First Order Differential Geometry

- "Introduction to Smooth Manifolds" John M. Lee

Chapter 11 "The Cotangent Bundle" is useful for a reference on the theory of cotangent bundles,
which corresponds to the structure of reverse mode AD through the optical equivalence. Also a
useful general reference for Differential Geomtry.

- "Natural Operations in Differential Geometry" - Ivan Kolář

I recommend Chapter IV. "Jets and Natural bundles"

# Readings on Higher Order Differential Geometry

- "Second Order Tangent Vectors in Riemannian Geometry", Fisher and Laquer, J. Korean Math Soc 36 (1999)

This one is quite good. I recommend reading the first half at least and tracing through the definitions.
This corresponds fairly closely to notion of iterated tangent spaces as implemented in Diffractor.

- "The Geometry of Jet Bundles" D. J. Saunders

I recommend reading Chapter 5 "Second order Jet bundles", though of course some earlier chapters
may be useful to understand this chapter. I'm not 100% happy with the notation, but it gives good
intuition.

