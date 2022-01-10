using Gen

@gen function model()
    x ~ uniform_continuous(-1, 1)
    y ~ uniform_continuous(-1, 1)
    return sqrt(x^2 + y^2)
end

function do_chunk(num_samples_per_thread)
    total = 0
    for i in 1:num_samples_per_thread
        trace = simulate(model, ())
        radius = get_retval(trace)
        if radius < 1.0
            total += 1
        end
    end
    return total / num_samples_per_thread
end

function estimate(num_threads::Integer, num_samples_per_thread::Integer)
    result = Vector{Float64}(undef, num_threads)
    Threads.@threads for thread_idx in 1:num_threads
        result[thread_idx] = do_chunk(num_samples_per_thread)
    end 
    return 4.0 * sum(result) / length(result)
end

result = estimate(parse(Int, ARGS[1]), parse(Int, ARGS[2]))
println("result: $result")
