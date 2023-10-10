"""
This is a demo of using Diffractor for Very basic signal processing.
Where you have a signal that is represented as a series of sampled readings.
Not even something as sophistricated as FFT, just time domain edge detection stuff.
Its an important use case so we test it directly.
"""
module SignalMeasurement
using Test
using Diffractor
using Diffractor: ∂☆, ZeroBundle, TaylorBundle
using Diffractor: bundle, first_partial, TaylorTangentIndex, primal


function make_soft_square_pulse(width, hardness=100)
    function soft_square(t)
        1/(1+exp(-hardness*(t-0.25))) - 1/(1+exp(-hardness*(t-0.25-width)))
    end
end
#soft_square = make_soft_square_pulse(0.5, 8)
#signal = soft_square.(0:0.001:1)
#scatter(signal)

function determine_width(xs, ts)
    # vs real signal processing functions this is not very robust, but for sake of demonstration it is fine.
    @assert eachindex(xs) == eachindex(ts)

    start_idx = nothing
    end_idx = nothing
    for ii in eachindex(xs)
        x = xs[ii]
        if isnothing(start_idx)
            if x > 0.5
                start_idx = ii
            end
        else
            if x < 0.5
                end_idx = ii
                break
            end
        end
    end

    (isnothing(start_idx) || isnothing(end_idx)) && throw(DomainError("no pulse found"))
    return ts[end_idx] - ts[start_idx]
end

#ts = 0:0.001:1
#determine_width(make_soft_square_pulse(0.5, 5).(ts), ts)


function signal_problem(width)
    func = make_soft_square_pulse(width, 8)
    ts = 0.0:0.001:1.0
    signal = map(func, ts)
    return determine_width(signal, ts)
end



🐰 = ∂☆{1}()(ZeroBundle{1}(signal_problem), TaylorBundle{1}(0.5, (1.0,)))
@test primal(🐰) ≈ signal_problem(0.5)
@test first_partial(🐰) ≈ 1.0

end  # module