using Diffractor
using Diffractor: var"'"

using Test

@test sin'(1.0) == cos(1.0)
