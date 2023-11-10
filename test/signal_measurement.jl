"""
This is a demo of using Diffractor for Very basic signal processing.
Where you have a signal that is represented as a series of sampled readings.
Not even something as sophistricated as FFT, just time domain edge detection stuff.
Its an important use case so we test it directly.
"""
module SignalMeasurement
using Test
using ChainRulesCore
using Diffractor
using Diffractor: ∂☆, ZeroBundle, TaylorBundle
using Diffractor: bundle, first_partial, TaylorTangentIndex, primal


function make_soft_square_pulse(width, hardness=100)
    function soft_square(t)
        1/(1+exp(-hardness*(t-0.25))) - 1/(1+exp(-hardness*(t-0.25-width)))
    end
end
#soft_square = make_soft_square_pulse(0.5, 8)
#signal = soft_square.(0:0.001:1)]st
#scatter(signal)

@testset "pulse width" begin
    function determine_width(xs, ts)
        # vs real signal processing functions this is not very robust, but for sake of demonstration it is fine.
        start_ind = findfirst(>(0.5), xs)
        end_ind = findnext(<(0.5), xs, start_ind)
        return ts[end_ind] - ts[start_ind]
    end

    function signal_problem(width)
        func = make_soft_square_pulse(width, 8)
        ts = 0.0:0.001:1.0
        signal = map(func, ts)
        return determine_width(signal, ts)
    end

    function ChainRulesCore.frule((_, ẋs, ṫs), ::typeof(determine_width), xs, ts)
        iszero(ṫs) || throw(ArgumentError("not supporting nonzero tangents to `ts` (time steps)"))
        # If we needed to support this we could do something with interpolation and resampling
        
        # Apply finite difference
        y = determine_width(xs, ts)
        y⃑ = determine_width(xs .+ ẋs, ts)
        ẏ = y⃑ - y
        return y, ẏ
    end


    # δ (pertubation) Can't be too big or it will blow the perturbed signal out of the window
    for δ in (0.001, 0.003, 0.0045, 0.1, 0.04)
        🐰 = ∂☆{1}()(ZeroBundle{1}(signal_problem), TaylorBundle{1}(0.5, (δ,)))
        @test primal(🐰) ≈ signal_problem(0.5)
        @test convert(Float64, first_partial(🐰)) ≈ δ rtol=0.2
    end
end

@testset "risetime" begin
    function determine_risetime(xs, ts)
        start_ind = findfirst(>(0.2), xs)
        end_ind = findnext(>(0.8), xs, start_ind)
        return ts[end_ind] - ts[start_ind]
    end

    function ChainRulesCore.frule((_, ẋs, ṫs), ::typeof(determine_risetime), xs, ts)
        iszero(ṫs) || throw(ArgumentError("not supporting nonzero tangents to `ts` (time steps)"))
        # If we needed to support this we could do something with interpolation and resampling
        
        # Apply finite difference
        y = determine_risetime(xs, ts)
        y⃑ = determine_risetime(xs .+ ẋs, ts)
        ẏ = y⃑ - y
        return y, ẏ
    end

    function signal_risetime_problem(hardness)
        func = make_soft_square_pulse(0.5, hardness)
        ts = 0.0:0.001:1.0
        signal = map(func, ts)
        return determine_risetime(signal, ts)
    end



    🐇 = ∂☆{1}()(ZeroBundle{1}(signal_risetime_problem), TaylorBundle{1}(12, (1.0,)))
    @test primal(🐇) ≈ signal_risetime_problem(12)
    @test convert(Float64, first_partial(🐇)) < 0 # As you increase the hardness the risetime decreases
end


end  # module