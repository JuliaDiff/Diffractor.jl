# Diffractor

Next-generation AD

[PDF containing the terminology](terminology.pdf)

## Getting Started

⚠️This certainly has bugs and issues. Please open issues on [Diffractor.jl](https://github.com/JuliaDiff/Diffractor.jl/), or [AbstractDifferentiation.jl](https://github.com/JuliaDiff/AbstractDifferentiation.jl/issues) as appropriate.⚠️

Diffractor's public API is via [AbstractDifferentiation.jl](https://github.com/JuliaDiff/AbstractDifferentiation.jl/).
Please see the AbstractDifferentiation.jl docs for detailed usage.

```@jldoctest
julia> using Diffractor: DiffractorForwardBackend

julia> using AbstractDifferentiation: derivative

julia> derivative(DiffractorForwardBackend(), +, 1.5, 10.0)
(1.0, 1.0)

julia> derivative(DiffractorForwardBackend(), *, 1.5, 10.0)
(10.0, 1.5)

julia> jacobian(DiffractorForwardBackend(), prod, [1.5, 2.5, 10.0]) |> only
1×3 Matrix{Float64}:
 25.0  15.0  3.75

julia> jacobian(DiffractorForwardBackend(), identity, [1.5, 2.5, 10.0]) |> only
3×3 Matrix{Float64}:
 1.0  0.0  0.0
 0.0  1.0  0.0
 0.0  0.0  1.0
```