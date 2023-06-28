var documenterSearchIndex = {"docs":
[{"location":"api.html","page":"API","title":"API","text":"CurrentModule = Diffractor","category":"page"},{"location":"api.html#Diffractor","page":"API","title":"Diffractor","text":"","category":"section"},{"location":"api.html","page":"API","title":"API","text":"","category":"page"},{"location":"api.html","page":"API","title":"API","text":"Modules = [Diffractor]","category":"page"},{"location":"api.html#Diffractor.AbstractTangentBundle","page":"API","title":"Diffractor.AbstractTangentBundle","text":"abstract type TangentBundle{N, B}; end\n\nThis type represents the N-th order (iterated) tangent bundle [1] TⁿB over some base (Riemannian) manifold B. Note that TⁿB is itself another manifold and thus in particular a vector space (over ℝ). As such, subtypes of this abstract type are expected to support the usual vector space operations.\n\nHowever, beyond that, this abstract type makes no guarantee about the representation. That said, to gain intution for what this object is, it makes sense to pick some explicit bases and write down examples.\n\nTo that end, suppose that B=ℝ. Then T¹B=T¹ℝ is just our usual notion of a dual number, i.e. for some element η ∈ T¹ℝ, we may consider η = a + bϵ for real numbers (a, b) and ϵ an infinitessimal differential such that ϵ^2 = 0.\n\nEquivalently, we may think of η as being identified with the vector (a, b) ∈ ℝ² with some additional structure. The same form essentially holds for general B, where we may write (as sets):\n\nT¹B = {(a, b) | a ∈ B, b ∈ Tₐ B }\n\nNote that these vectors are orthogonal to those in the underlying base space. For example, if B=ℝ², then we have:\n\nT¹ℝ² = {([aₓ, a_y], [bₓ, b_y]) | [aₓ, a_y] ∈ ℝ², [bₓ, b_y] ∈ Tₐ ℝ² }\n\nFor convenience, we will sometimes writes these in one as:\n\nη ∈ T ℝ²  = aₓ x̂ + a_y ŷ + bₓ ∂/∂x|_aₓ x + b_y ∂/∂y|_{a_y}\n         := aₓ x̂ + a_y ŷ + bₓ ∂/∂¹x + b_y ∂/∂¹y\n         := [aₓ, a_y] + [bₓ, b_y] ∂/∂¹\n         := a + b ∂/∂¹\n         := a + b ∂₁\n\nThese are all definitional equivalences and we will mostly work with the final form. An important thing to keep in mind though is that the subscript on ∂₁ does not refer to a dimension of the underlying base manifold (for which we will rarely pick an explicit basis here), but rather tags the basis of the tangent bundle.\n\nLet us iterate this construction to second order. We have:\n\nT²B = T¹(T¹B) = { (α, β) | α ∈ T¹B, β ∈ T_α T¹B }\n              = { ((a, b), (c, d)) | a ∈ B, b ∈ Tₐ B, c ∈ Tₐ B, d ∈ T²ₐ B}\n\n(where in the last equality we used the linearity of the tangent vector).\n\nFollowing our above notation, we will canonically write such an element as:\n\n  a + b ∂₁ + c ∂₂ + d ∂₂ ∂₁\n= a + b ∂₁ + c ∂₂ + d ∂₁ ∂₂\n\nIt is worth noting that there still only is one base point a of the underlying manifold and thus TⁿB is a vector bundle over B for all N.\n\nFurther Reading\n\n[1] https://en.wikipedia.org/wiki/Tangent_bundle\n\n\n\n\n\n","category":"type"},{"location":"api.html#Diffractor.CompositeBundle","page":"API","title":"Diffractor.CompositeBundle","text":"TupleTangentBundle{N, B <: Tuple}\n\nRepresents the tagent bundle where the base space is some tuple type. Mathematically, this tangent bundle is the product bundle of the individual element bundles.\n\n\n\n\n\n","category":"type"},{"location":"api.html#Diffractor.ExplicitTangent","page":"API","title":"Diffractor.ExplicitTangent","text":"struct ExplicitTangent{P}\n\nA fully explicit coordinate representation of the tangent space, represented by a vector of 2^(N-1) partials.\n\n\n\n\n\n","category":"type"},{"location":"api.html#Diffractor.Jet","page":"API","title":"Diffractor.Jet","text":"struct Jet{T, N}\n\nRepresents the truncated (N-1)-th order Taylor series\n\nf(a) + (x-a)f'(a) + 1/2(x-a)^2f''(a) + ...\n\nCoefficients are stored in unscaled form. For a jet j, several operations are supported:\n\nIndexing j[i] returns fᵢ\nEvaluation j(x) semantically evaluates the truncated taylor series at x. However, evaluation is restricted to be precisely at a - the additional information in the taylor series is only available through derivatives. Mathematically this corresponds to an infinitessimal ball around a.\n\n\n\n\n\n","category":"type"},{"location":"api.html#Diffractor.ProductTangent","page":"API","title":"Diffractor.ProductTangent","text":"struct ProductTangent{T <: Tuple{Vararg{AbstractTangentSpace}}}\n\nRepresents the product space of the given representations of the tangent space.\n\n\n\n\n\n","category":"type"},{"location":"api.html#Diffractor.TangentBundle","page":"API","title":"Diffractor.TangentBundle","text":"struct TangentBundle{N, B, P}\n\nRepresents a tangent bundle as an explicit primal together with some representation of (potentially a product of) the tangent space.\n\n\n\n\n\n","category":"type"},{"location":"api.html#Diffractor.TaylorTangent","page":"API","title":"Diffractor.TaylorTangent","text":"struct TaylorTangent{C}\n\nThe taylor bundle construction mods out the full N-th order tangent bundle by the equivalence relation that coefficients of like-order basis elements be equal, i.e. rather than a generic element\n\na + b ∂₁ + c ∂₂ + d ∂₃ + e ∂₂ ∂₁ + f ∂₃ ∂₁ + g ∂₃ ∂₂ + h ∂₃ ∂₂ ∂₁\n\nwe have a tuple (c₀, c₁, c₂, c₃) corresponding to the full element\n\nc₀ + c₁ ∂₁ + c₁ ∂₂ + c₁ ∂₃ + c₂ ∂₂ ∂₁ + c₂ ∂₃ ∂₁ + c₂ ∂₃ ∂₂ + c₃ ∂₃ ∂₂ ∂₁\n\ni.e.\n\nc₀ + c₁ (∂₁ + ∂₂ + ∂₃) + c₂ (∂₂ ∂₁ + ∂₃ ∂₁ + ∂₃ ∂₂) + c₃ ∂₃ ∂₂ ∂₁\n\nThis restriction forms a submanifold of the original manifold. The naming is by analogy with the (truncated) Taylor series\n\nc₀ + c₁ x + 1/2 c₂ x² + 1/3! c₃ x³ + O(x⁴)\n\n\n\n\n\n","category":"type"},{"location":"api.html#Diffractor.UniformTangent","page":"API","title":"Diffractor.UniformTangent","text":"struct UniformTangent\n\nRepresents an N-th order tangent bundle with all unform partials. Particularly useful for representing singleton values.\n\n\n\n\n\n","category":"type"},{"location":"api.html#Diffractor.∂⃖","page":"API","title":"Diffractor.∂⃖","text":"∂⃖{N}\n\n∂⃖{N} is the reverse-mode AD optic functor of order N. A call (::∂⃖{N})(f, args...) corresponds to  ∂⃖ⁿ f(args...) in the linear encoding of an N-optic (see the terminology guide for definitions of these terms).\n\nIn general (::∂⃖{N})(f, args...) will return a tuple of the original primal value f(args...) (in rare cases primitives may modify the primal value - in general we will ignore this rare complication for the purposes of clear documentation) and an optic continuation λ. The interpretation of this continuation depends on the order of the functor:\n\nFor example, ∂⃖{1} computes first derivatives. In particular, for a function f, ∂⃖{1}(f, args...) will return the tuple (f(args...), f⋆) (read \"f upper-star\").\n\n\n\n\n\n","category":"type"},{"location":"api.html#Diffractor.∂☆","page":"API","title":"Diffractor.∂☆","text":"∂☆{N}\n\n∂☆{N} is the forward-mode AD functor of order N. A call (::∂☆{N})(f, args...) evaluating a function f: A -> B is lifted to its pushforward on the N-th order tangent bundle f⋆: Tⁿ A -> Tⁿ B.\n\n\n\n\n\n","category":"type"},{"location":"api.html#Diffractor.∇","page":"API","title":"Diffractor.∇","text":"∇(f, args...)\n\nComputes the gradient ∇f(x, y, z...) (at (x, y, z...)). In particular, the return value will be a tuple of partial derivatives (∂f/∂x, ∂f/∂y, ∂f/∂z...).\n\nCurried version\n\nAlternatively, ∇ may be curried, essentially giving the gradient as a function:\n\nExamples\n\njulia> using Diffractor: ∇\n\njulia> map(∇(*), (1,2,3), (4,5,6))\n((4.0, 1.0), (5.0, 2.0), (6.0, 3.0))\n\nThe derivative ∂f/∂f\n\nNote that since in Julia, there is no distinction between functions and values, there is in principle a partial derivative with respect to the function itself. However, said partial derivative is dropped by this interface. It is however available using the lower level ∂⃖ if desired. This interaction can also be used to obtain gradients with respect to only some of the arguments by using a closure:\n\n∇((x,z)->f(x,y,z))(x, z) # returns (∂f/∂x, ∂f/∂z)\n\nThough of course the same can be obtained by simply indexing the resulting tuple (in well-inferred code there should not be a performance difference between these two options).\n\n\n\n\n\n","category":"type"},{"location":"api.html#Diffractor.:'-Tuple{Any}","page":"API","title":"Diffractor.:'","text":"f'\n\nThis is a convenience syntax for taking the derivative of a function f: ℝ -> ℝ. In particular, for such a function f'(x) will be the first derivative of f at x (and similar for f''(x) and second derivatives and so on.)\n\nNote that the syntax conflicts with the Base definition for the adjoint of a matrix and thus is not enabled by default. To use it, add the following to the top of your module:\n\nusing Diffractor: var\"'\"\n\nIt is also available using the @∂ macro:\n\n@∂ f'(x)\n\n\n\n\n\n","category":"method"},{"location":"api.html#Diffractor.dx-Tuple{Real}","page":"API","title":"Diffractor.dx","text":"dx(x)\n\ndx represents the trival differential one-form of a one dimensional Riemannian manifold M. In particular, it is a section of the cotangent bundle of M, meaning it may be evaluted at a point x of M to obtain an element of the cotangent space T*ₓ M to M at x. We impose no restrictions on the representations of either the manifold itself or the cotangent space.\n\nBy default, the only implementation provided identifies T*ₓ ℝ ≃ ℝ, keeping watever type is used to represent ℝ. i.e.\n\ndx(x::Real) = one(x)\n\nHowever, users may provide additional overloads for custom representations of one dimensional Riemannian manifolds.\n\n\n\n\n\n","category":"method"},{"location":"api.html#Diffractor.forward_diff_no_inf!-Tuple{Core.Compiler.IRCode, Vector{Pair{Core.SSAValue, Int64}}}","page":"API","title":"Diffractor.forward_diff_no_inf!","text":"forward_diff_no_inf!(ir, to_diff; visit_custom!, transform)\n\nInternal method which generates the code for forward mode diffentiation\n\nir the IR being differnetation\nto_diff: collection of all SSA values for which the derivative is to be taken,             paired with the order (first deriviative, second derivative etc)\nvisit_custom!(ir, stmt, order::Int, recurse::Bool): \n\n\tdecides if the custom `transform!` should be applied to a `stmt` or not\n\tDefault: `false` for all statements\n\ntransform!(ir, ssa::SSAValue, order::Int) mutates ir to do a custom tranformation.\n\n\n\n\n\n","category":"method"},{"location":"api.html#Diffractor.jet_taylor_ev-Union{Tuple{N}, Tuple{Val{N}, Any, Any}} where N","page":"API","title":"Diffractor.jet_taylor_ev","text":"jet_taylor_ev(::Val{}, jet, taylor)\n\nGenerates a closed form arithmetic expression for the N-th component of the action of a 1d jet (of order at least N) on a maximally symmetric (i.e. taylor) tangent bundle element. In particular, if we represent both the jet and the taylor tangent bundle element by their associated canonical taylor series:\n\nj = j₀ + j₁ (x - a) + j₂ 1/2 (x - a)^2 + ... + jₙ 1/n! (x - a)^n\nt = t₀ + t₁ (x - t₀) + t₂ 1/2 (x - t₀)^2 + ... + tₙ 1/n! (x - t₀)^n\n\nthen the action of evaluating j on t, is some other taylor series\n\nt′ = a + t′₁ (x - a) + t′₂ 1/2 (x - a)^2 + ... + t′ₙ 1/n! (x - a)^n\n\nThe t′ᵢ can be found by explicitly plugging in t for every x and expanding out, dropping terms of orders that are higher. This computes closed form expressions for the t′ᵢ that are hopefully easier on the compiler.\n\n\n\n\n\n","category":"method"},{"location":"api.html#Diffractor.∂x-Tuple{Real}","page":"API","title":"Diffractor.∂x","text":"∂x(x)\n\nFor x in a one dimensional manifold, map x to the trivial, unital, 1st order tangent bundle. It should hold that ∀x ⟨∂x(x), dx(x)⟩ = 1\n\n\n\n\n\n","category":"method"},{"location":"api.html#Diffractor.@∂-Tuple{Any}","page":"API","title":"Diffractor.@∂","text":"@∂\n\nConvenice macro for writing partial derivatives. E.g. The expression:\n\n@∂ f(∂x, ∂y)\n\nWill compute the partial derivative ∂^2 f/∂x∂y at (x, y)`. And similarly\n\n@∂ f(∂²x, ∂y)\n\nwill compute the derivative ∂^3 f/∂x^2 ∂y at (x,y).\n\n\n\n\n\n","category":"macro"},{"location":"reading_list.html","page":"Reading List","title":"Reading List","text":"This is a list of references I found useful while thinking about Diffractor. If you are new to Julia, AD or Diffractor and are primarily intersted in Diffractor, how it works, how to use it, or even the general Diffractor theory, this is probably not the list for you. As always in the literature, some of these references use terms differently from how they are used in Diffractor (as well as being inconsistent with each other). Additionally, many of these references are quite dense and though I've found small nuggets of insight in each, excavating those took many hours. Also, these are not introductory texts. If you've not taken an introductory differential geometry course, I would recommend looking for that first. Don't feel bad if some of these references read like like nonsense. It often reads that way to me too.","category":"page"},{"location":"reading_list.html#Reading-on-Optics","page":"Reading List","title":"Reading on Optics","text":"","category":"section"},{"location":"reading_list.html","page":"Reading List","title":"Reading List","text":"\"Categories of Optics\" - Mitchell Riley - https://arxiv.org/abs/1809.00738","category":"page"},{"location":"reading_list.html","page":"Reading List","title":"Reading List","text":"The original paper on optics. Good background for understanding the optics terminology.","category":"page"},{"location":"reading_list.html#Readings-on-First-Order-Differential-Geometry","page":"Reading List","title":"Readings on First Order Differential Geometry","text":"","category":"section"},{"location":"reading_list.html","page":"Reading List","title":"Reading List","text":"\"Introduction to Smooth Manifolds\" John M. Lee","category":"page"},{"location":"reading_list.html","page":"Reading List","title":"Reading List","text":"Chapter 11 \"The Cotangent Bundle\" is useful for a reference on the theory of cotangent bundles, which corresponds to the structure of reverse mode AD through the optical equivalence. Also a useful general reference for Differential Geomtry.","category":"page"},{"location":"reading_list.html","page":"Reading List","title":"Reading List","text":"\"Natural Operations in Differential Geometry\" - Ivan Kolář","category":"page"},{"location":"reading_list.html","page":"Reading List","title":"Reading List","text":"I recommend Chapter IV. \"Jets and Natural bundles\"","category":"page"},{"location":"reading_list.html#Readings-on-Higher-Order-Differential-Geometry","page":"Reading List","title":"Readings on Higher Order Differential Geometry","text":"","category":"section"},{"location":"reading_list.html","page":"Reading List","title":"Reading List","text":"\"Second Order Tangent Vectors in Riemannian Geometry\", Fisher and Laquer, J. Korean Math Soc 36 (1999)","category":"page"},{"location":"reading_list.html","page":"Reading List","title":"Reading List","text":"This one is quite good. I recommend reading the first half at least and tracing through the definitions. This corresponds fairly closely to notion of iterated tangent spaces as implemented in Diffractor.","category":"page"},{"location":"reading_list.html","page":"Reading List","title":"Reading List","text":"\"The Geometry of Jet Bundles\" D. J. Saunders","category":"page"},{"location":"reading_list.html","page":"Reading List","title":"Reading List","text":"I recommend reading Chapter 5 \"Second order Jet bundles\", though of course some earlier chapters may be useful to understand this chapter. I'm not 100% happy with the notation, but it gives good intuition.","category":"page"},{"location":"index.html#Diffractor","page":"Introduction","title":"Diffractor","text":"","category":"section"},{"location":"index.html","page":"Introduction","title":"Introduction","text":"Next-generation AD","category":"page"},{"location":"index.html","page":"Introduction","title":"Introduction","text":"PDF containing the terminology","category":"page"}]
}