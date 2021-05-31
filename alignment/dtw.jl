module DTW
include("step_pattern.jl")
using Distances: PreMetric

"""
Given a Rule-pattern, computes its cost
"""
@inline function step_pattern_cost(step_pattern::StepPattern, matrix::Matrix{T}, i::Int64,
    j::Int64)::Tuple{T, Int64} where {T<:Number}

    step_costs = Vector{T}(undef, length(step_pattern.rules) + 1)
    for (k, rule) in enumerate(step_pattern.rules)
        rule_cost::T = 0.0
        stop = false
        for comp::Component in rule.components
            if i - comp.row > 0 && j - comp.col > 0
                val = matrix[i - comp.row, j - comp.col]
                if isinf(val)
                    # a value outside the window
                    step_costs[k] = Inf
                    stop = true
                    break
                end
                rule_cost += comp.weight * val
            else
                # this rule would connect an element outside the matrix
                step_costs[k] = Inf
                stop = true
                break
            end
        end
        if !stop
            step_costs[k] = rule.weight * rule_cost + rule.bias
        end
    end
    # if no rule is allowed, use Inf
    step_costs[end] = Inf
    return findmin(step_costs)
end

"""
DTW functions working with Vectors of Vectors, that is: each point can
have a variable numer of features.

Returns a matrix containing at each entry the dtw cost to reach that entry.
The total dtw_cost can be found in [end, end].
"""
function dtw_matrix(a::T, b::T, dist_fn::PreMetric; window_fn::Function=(i, j) -> true,
                    step_pattern::StepPattern=symmetric2) where {T}

    matrix = fill(Inf, length(a), length(b))
    matrix[1, 1] = dist_fn(a[1], b[1])
    for j in 1:size(matrix, 2), i in 1:size(matrix, 1)
        # only if in window and not (1, 1)
        if window_fn(i, j) && i + j > 2
            matrix[i, j] = dist_fn(a[i], b[j])  # Calculate distance of current signal indices
            matrix[i, j], _ = step_pattern_cost(step_pattern, matrix, i, j)
        end
    end
    return matrix
end

function dtw_matrix(dist_matrix::Matrix{Float64}; window_fn::Function=(i, j) -> true,
                    step_pattern::StepPattern=symmetric2)

    matrix = fill(Inf::Float64, size(dist_matrix)...)
    matrix[1, 1] = dist_matrix[1, 1]
    for j in 1:size(matrix, 2), i in 1:size(matrix, 1)
        # only if in window and not (1, 1)
        if window_fn(i, j) && i + j > 2
            matrix[i, j] = dist_matrix[i, j]
            matrix[i, j], _ = step_pattern_cost(step_pattern, matrix, i, j)
        end
    end

    return matrix
end

"""
Modifies dist_matrix in place to save ram
"""
function dtw_matrix!(dist_matrix::Matrix{Float64}; window_fn::Function=(i, j) -> true,
                     step_pattern::StepPattern=symmetric2)

    for j in 1:size(dist_matrix, 2), i in 1:size(dist_matrix, 1)
        # only if in window and not (1, 1)
        if window_fn(i, j) && i + j > 2
            dist_matrix[i, j], _ = step_pattern_cost(step_pattern, dist_matrix, i, j)
        end
    end

    return dist_matrix
end

"""
DTW functions working with Vectors of Vectors, that is: each point can
have a variable numer of features.

Returns a matrix containing at each entry the rule used for the best path.
"""
function dtw_matrix_directions(a::T, b::T, dist_fn::PreMetric;
                               window_fn::Function=(i, j) -> true,
                               step_pattern::StepPattern=symmetric2)::Union{Matrix{Int64},
                                                                             Nothing} where {T}

    directions = zeros(Int64, length(a), length(b))
    matrix = fill(Inf, length(a), length(b))
    matrix[1, 1] = dist_fn(a[1], b[1])
    for j in 1:size(matrix, 2), i in 1:size(matrix, 1)
        # only if in window and not (1, 1)
        if window_fn(i, j) && i + j > 2
            matrix[i, j] = dist_fn(a[i], b[j])  # Calculate distance of current signal indices
            matrix[i, j], directions[i, j] = step_pattern_cost(step_pattern, matrix, i, j)
        end
    end
    if isinf(matrix[end, end])
        # no valid path
        return nothing
    end

    return directions
end

function dtw_matrix_directions(dist_matrix::Matrix{Float64};
                               window_fn::Function=(i, j) -> true,
                               step_pattern::StepPattern=symmetric2)::Union{Matrix{Int64},
                                                                             Nothing}

    directions = zeros(Int64, size(dist_matrix)...)
    matrix = fill(Inf, size(dist_matrix)...)
    matrix[1, 1] = dist_matrix[1, 1]
    for j in 1:size(matrix, 2), i in 1:size(matrix, 1)
        # only if in window and not (1, 1)
        if window_fn(i, j) && i + j > 2
            matrix[i, j] = dist_matrix[i, j]
            matrix[i, j], directions[i, j] = step_pattern_cost(step_pattern, matrix, i, j)
        end
    end

    if isinf(matrix[end, end])
        # no valid path
        return nothing
    end

    return directions
end

"""
Modifies dist_matrix in place to save ram
"""
function dtw_matrix_directions!(dist_matrix::Matrix{Float64};
                                window_fn::Function=(i, j) -> true,
                                step_pattern::StepPattern=symmetric2)::Union{Matrix{Int64},
                                                                              Nothing}

    directions = zeros(Int64, size(dist_matrix)...)
    for j in 1:size(dist_matrix, 2), i in 1:size(dist_matrix, 1)
        # only if in window and not (1, 1)
        if window_fn(i, j) && i + j > 2
            dist_matrix[i, j], directions[i, j] = step_pattern_cost(step_pattern,
                                                                    dist_matrix, i, j)
        end
    end
    if isinf(dist_matrix[end, end])
        # no valid path
        return nothing
    end

    return directions
end

"""
Performs the trackback algorithm and returns the best path found starting
from the first index of the original input sequences (i.e. the first row
and column are discarded).
"""
@views function trackback(matrix::Matrix{Int64},
                          step_pattern::StepPattern=symmetric2)::Matrix{Int64}
    pos = collect(size(matrix))
    at_start_index = false
    shortest_path::Vector{Vector{Int64}} = []
    push!(shortest_path, copy(pos))  # We start at the end and work back
    L = length(step_pattern.rules)

    while !at_start_index

        # with direction matrix
        dir = matrix[pos...]

        best_rule = step_pattern.rules[dir]

        # computing the movements according to this rule
        for comp in best_rule.components[(end - 1):-1:1]
            end_comp = pos - [comp.row, comp.col]
            pushfirst!(shortest_path, end_comp)
        end

        # update pos
        pos -= [best_rule.components[1].row, best_rule.components[1].col]

        # Stop once you're at the beginning
        if pos == [1, 1]
            at_start_index = true
        end
    end
    return vcat(transpose(shortest_path)...)
end

"""
Perform DTW and path trackback; returns `Nothing` if no path is found.
"""
function dtw_path(a::T, b::T, dist_fn::PreMetric; window_fn::Function=(i, j) -> true,
                  step_pattern::StepPattern=symmetric2)::Union{Matrix{Int64},
                                                                Nothing} where {T}
    matrix = dtw_matrix_directions(a, b, dist_fn; window_fn, step_pattern)
    if matrix === nothing
        return nothing
    end
    return trackback(matrix, step_pattern)
end
function dtw_path(dist_matrix::Matrix{Float64}; window_fn::Function=(i, j) -> true,
                  step_pattern::StepPattern=symmetric2)::Union{Matrix{Int64},
                                                                Nothing}
    matrix = dtw_matrix_directions(dist_matrix; window_fn, step_pattern)
    if matrix === nothing
        return nothing
    end
    return trackback(matrix, step_pattern)
end
function dtw_path!(dist_matrix::Matrix{Float64}; window_fn::Function=(i, j) -> true,
                   step_pattern::StepPattern=symmetric2)::Union{Matrix{Int64},
                                                                 Nothing}
    matrix = dtw_matrix_directions!(dist_matrix; window_fn, step_pattern)
    if matrix === nothing
        return nothing
    end
    return trackback(matrix, step_pattern)
end
end # module
