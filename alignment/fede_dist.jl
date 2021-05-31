module FedeDist
using Distances: PreMetric

function simcarn_diff(diff, th, use_simcarn)
    if !use_simcarn
        return diff < th
    end
    diff = diff % 12
    if diff == 0
        return 0 < th
    elseif diff == 1
        # minor second
        return 0.8 < th
    elseif diff == 2
        # major second
        return 0.8 < th
    elseif diff == 3
        # minor third
        return 0.25 < th
    elseif diff == 4
        # major third
        return 0.25 < th
    elseif diff == 5
        # perfect fourth
        return 0.7 < th
    elseif diff == 6
        # diminished fifth
        return 0.5 < th
    elseif diff == 7
        # perfect fifth
        return 0.25 < th
    elseif diff == 8
        # minor sixth
        return 0.6 < th
    elseif diff == 9
        # major sixth
        return 0.6 < th
    elseif diff == 10
        # minor seventh
        return 0.5 < th
    elseif diff == 11
        # major seventh
        return 0.8 < th
    else
        return 1 < th
    end
end

"""
Counts how many elements in `x` and `y` are common; each element is counted at
most once, so that the maximum values returned are `length(x)` and `length(y)`
"""
function my_intersect_count(x, y, th, simcarn)

    count_x::Int64 = 0
    count_y::Int64 = 0
    last_x = -1
    free_y = trues(length(y))
    for (i, x_val) in enumerate(x)
        for (j, y_val) in enumerate(y)
            if simcarn_diff(abs(x_val - y_val), th, simcarn)
                if last_x < i
                    count_x += 1
                    last_x = i
                end
                if free_y[j]
                    count_y += 1
                    free_y[j] = false
                end
            end
        end
    end

    return count_x, count_y
end

"""
Returns the indices referred to `x` and `y` that match common elements
"""
function my_intersect_idx(x, y, th, simcarn)

    common_x = Int64[]
    common_y = Int64[]
    for (i, x_val) in enumerate(x), (j, y_val) in enumerate(y)
        if simcarn_diff(abs(x_val - y_val), th, simcarn)
            push!(common_x, i)
            push!(common_y, j)
        end
    end

    return common_x, common_y
end

"""
match pitches if their difference lies under `th`
"""
function _jaccard_dist(sample1::Vector{Int64}, sample2::Vector{Int64},
th::Float64,
                       simcarn::Bool)::Float64

    _intersect = my_intersect_count(sample1, sample2, th, simcarn)
    intersect = min(_intersect[1], _intersect[2])
    @fastmath return 1 - intersect / (length(sample1) + length(sample2) - intersect)
end

struct JaccardDist <: PreMetric
    th::Float64
    simcarn::Bool
end

function (self::JaccardDist)(x::Vector{Int64}, y::Vector{Int64})
    return _jaccard_dist(x, y, self.th, self.simcarn)
end
end # module
