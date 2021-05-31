module FedeDTW
using Distances: PreMetric
using Statistics: median, quantile
using PyCall: pyimport

"""
Computes the 1-1 average difference at various shift and penalizes each difference by
elevating to `1/shift`.

Returns the meainum
"""
function win_dist(
    arr1::Vector{Vector{Int64}},
    arr2::Vector{Vector{Int64}},
    dist_fn::PreMetric,
)
    # compute distance matrix
    # for j in 1:length(arr2)
    #     @simd for i in 1:length(arr1)
    #         diff[i, j] = dist_fn(arr1[i], arr2[j])
    #     end
    # end

    # # apply hungarian algorithm and find all indices
    # indices = findall(==(Hungarian.STAR), Hungarian.munkres(diff))

    # # compute cost of the matching
    # return median(diff[indices])
    # return quantile(diff[indices], 0.25)
    #
    out = Inf
    K = length(arr1)
    K1 = K + 1
    @simd for shift = 1:K
        v = mean(dist_fn.(arr1[shift:K], arr2[1:K1-shift]))^(1 / shift)
        if v < out
            out = v
        end
    end
    return out
end

"""
given an idx, a radius and a maximum length, returns starting and ending
indices of a a window centered at that idx and having that radius, without
indices > length nor < 0
"""
function idx_range(idx, radius, length)
    return max(1, idx - radius), min(length, idx + radius)
end

struct FedeWindowParam
    dist::PreMetric
    min_radius::Int64
    α::Int64
    β::Int64
end

"""
A windowing function which computes a different slanted-band at each point
based on the local difference of the main slanted diagonal; the local
radius is computed as:

`max(
   min_radius,
   floor(
       α * avg_dist_fn(
               x[i - β : i + β],
               y[j - β : j + β]
           )
   )
)`

where:
* N is the length of x
* M is the length of y
* avg_dist_fn is the average of dist_fn on each corresponding sample
* j = floor(i * M / N)

By words, `β` is half the length of a sliding window used to compute
distances between excerpts of `x` and `y` taken along the slanted diagonal.
The distance is multiplied by `α` to get the local radius length.

`x` and `y` are sequences with shape ([M, N], features)
"""
struct FedeWindowFunc <: Function
    mask::Matrix{Bool}
end

# adding the callable action
(self::FedeWindowFunc)(i::Int64, j::Int64)::Bool = self.mask[i, j]

function get_fede_win(
    params::FedeWindowParam,
    x::Vector{Vector{T}},
    y::Vector{Vector{T}},
) where {T<:Number}
    # take the distance function
    N = length(x)
    M = length(y)
    _transpose = false
    if M > N
        # x should always be longer than y
        x, y = y, x
        N, M = M, N
        # if we swap x and y, we need to swap the mask too
        _transpose = true
    end

    # a matrix where distances between windows are computed
    # R = params.β * 2 + 1
    # diff = fill(Inf, R, R)

    # a mask to remember points
    mask = falses(N, M)

    # for each point in x
    for i = 1:N
        # compute the point in y along the diagonal
        j = floor(Int64, (i - 1) * M / N) + 1

        # compute the sliding windows
        start_x, end_x = idx_range(i, params.β, N)
        start_y, end_y = idx_range(j, params.β, M)
        _x = x[start_x:end_x]
        _y = y[start_y:end_y]

        # pad the windows
        if start_x == 1
            pushfirst!(_x, [[0] for k = 1:(params.β-i+1)]...)
        end
        if end_x == N
            push!(_x, [[0] for k = 1:(i+params.β-N)]...)
        end
        if start_y == 1
            pushfirst!(_y, [[0] for k = 1:(params.β-j+1)]...)
        end
        if end_y == M
            push!(_y, [[0] for k = 1:(j+params.β-M)]...)
        end

        # compute the local radius
        lr = max(
            params.min_radius,
            floor(Int64, params.α * quantile(params.dist.(_x, _y), 0.25)),
        )
        # floor(Int64, params.α * win_dist(_x, _y, params.dist)))

        # set the points inside the local radius to True
        idx_i = idx_range(i, lr, N)
        idx_j = idx_range(j, lr, M)
        mask[idx_i[1]:idx_i[2], idx_j[1]:idx_j[2]] .= true
    end

    if _transpose
        mask = transpose(mask)
    end

    return FedeWindowFunc(mask)
end

"""
Returns a Vector of Tuples. Each tuple contains a value from `x` and a Vector
with the indices where that value is in `x`. Only non-unique values are
returned.
"""
function my_nonunique(x::AbstractArray{T}) where {T}
    # counting each value and remembering indices
    indices = Dict{T,Vector{Int64}}()
    for (idx, val) in enumerate(x)
        if haskey(indices, val)
            # this is a duplicated value, push its index
            push!(indices[val], idx)
        else
            # this is the first time we meet this value
            indices[val] = [idx]
        end
    end
    # now filter out unique values
    out = Tuple{T,Vector{Int64}}[]
    for (key, val) in indices
        if length(val) > 1
            push!(out, (key, val))
        end
    end
    return out
end

"""
1. look for repeated values in `arr_x` or `arr_y`, depending on `target`
2. look for the maximum value in `graph_matrix[2]`, at the indices in
`arr_x` and `arr_y` relative to the repeated values
3. among the repeated values in the target, chose the ones corresponding to
the maximum in `graps_matrix[2]`
4. return `arr_x` and `arr_y` without the removed indices
"""
@views function _remove_conflicting_match(
    # TODO: rewrite in the light of weights...
    arr_x,
    arr_y,
    graph_matrix::AbstractArray{Int64,2},
    target::Int64,
)

    if target == 0
        _target = arr_x
    elseif target == 1
        _target = arr_y
    end

    arr_mask = ones(Bool, size(_target, 1))
    nonunique = my_nonunique(_target)
    for (val, conflicting_idx) in nonunique
        to_keep_idx_of_idx =
            argmax(graph_matrix[arr_x[conflicting_idx], arr_y[conflicting_idx]])[2]
        arr_mask[conflicting_idx] .= false
        arr_mask[conflicting_idx[to_keep_idx_of_idx]] = true
    end

    return arr_x[arr_mask], arr_y[arr_mask]
end

"""
OLD

Takes a list of mapping indices, fills the graph matrix counting the number
of times a match happens in the mappings. Then start taking matching from
the most matched and iteratively adding new matching. If two conflicting
matching have the same number of counts, takes the matching which appears
in the longest mapping; in case of parity the first one is taken
"""
@views function merge_matching_indices(args::Vector{Matrix{Int64}}, weights::AbstractArray{Float64})::Matrix{Int64}

    # voting
    votes::Dict{Vector{Int64}, Float64} = Dict()
    for i in 1:length(args)
        weight = weights[i]
        for match in eachcol(args[i]')
            if haskey(votes, match)
                votes[match] += weight
            else
                votes[match] = weight
            end
        end
    end
    if length(votes) == 0
        # no matches found!
        return Matrix{Float64}(undef, 0, 0)
    end

    # sorting by vote
    sorted_votes::Vector{Pair{Vector{Int64}, Float64}} = sort(
        collect(votes); by=x->x[2], rev=true)
        
    # number of notes
    matches = getindex.(sorted_votes, 1)
    N = maximum(getindex.(matches, 1))
    M = maximum(getindex.(matches, 2))

    # taking unique matches
    rows = falses(N)
    cols = falses(M)
    merged = Vector{Vector{Int64}}(undef, 0)
    for (match, vote) in sorted_votes
        row, col = match
        if !rows[row] && !cols[col]
            rows[row] = true
            cols[col] = true
            push!(merged, match)
        end
    end
    # re-sort everything and return
    out = vcat(merged'...)
    # print(f"Added notes from merging: {len(ref) - L}")
    return out[sortperm(out[:, 1]), :]
end
end # module
