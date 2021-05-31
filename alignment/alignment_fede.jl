module FedeAlignment
include("clustering.jl")
include("dtw.jl")
include("fede_dtw.jl")
include("fede_dist.jl")
import Clustering: cutree
import Distances: pairwise!, Cityblock, PreMetric
using DelimitedFiles: readdlm
import PyCall: PyObject

"""
Given repeated notes, chose the the best one
"""
@views function _chose_sample(matched_idx::AbstractMatrix{Int64},
                              score_reverse::Vector{Int64}, perfm_reverse::Vector{Int64},
                              dist_mat::AbstractMatrix{Float64})::Array{Bool,1}

    to_keep = ones(Bool, size(matched_idx, 1))
    nonunique_indices = FedeDTW.my_nonunique(matched_idx[:, 1])
    for (val, indices) in nonunique_indices
        clusters1 = score_reverse[val]
        clusters2 = perfm_reverse[matched_idx[indices, 2]]
        # here clusters1 is only 1 number, corresponding to the cluster of the
        # element which is being repeated
        chosen = argmin(dist_mat[clusters1, clusters2])
        to_keep[indices] .= false
        to_keep[indices[chosen]] = true
    end
    return to_keep

end

"""
Computes a unique path from a path among notes by removing repeated notes.
It chooses the couple of notes associated with the two nearest clusters. In
case of parity, the first couple is taken.
"""
@views function _get_unique_path(matched_idx::Matrix{Int64}, score_reverse::Vector{Int64},
                                 perfm_reverse::Vector{Int64},
                                 dist_mat::Array{Float64,2})::Matrix{Int64}

    to_keep = _chose_sample(matched_idx, score_reverse, perfm_reverse, dist_mat)
    matched_idx = matched_idx[to_keep, :]
    to_keep = _chose_sample(matched_idx[:, end:-1:1],
                            # view(matched_idx, :, size(matched_idx, 2):-1:1),
                            perfm_reverse, score_reverse, transpose(dist_mat))
    matched_idx = matched_idx[to_keep, :]
    return matched_idx
end

"""
Compute distance matrix using `dist_func`
"""
function _compute_dist_mat(clusters1::Vector{Vector{Int64}},
                           clusters2::Vector{Vector{Int64}}, dist_func::PreMetric,
                           win_fn::Function)

    N = length(clusters1)
    M = length(clusters2)
    out = fill(Inf::Float64, N, M)

    for j in 1:M
        @simd for i in 1:N
            # limit the computation to radius
            # see slanted band in
            # https://github.com/DynamicTimeWarping/dtw-python/blob/master/dtw/window.py
            if win_fn(i, j)
                out[i, j] = dist_func(clusters1[i], clusters2[j])
            end
        end
    end
    # initial and last note have zero distance from any note
    out[1, :] .= 0
    out[end, :] .= 0
    out[:, 1] .= 0
    out[:, end] .= 0
    return out
end

struct ClusterResult
    features::Vector{Vector{Int64}}
    clusters::Vector{Vector{Int64}}
    reverse::Vector{Int64}
end

struct DistData{T<:AbstractMatrix,K<:Function}
    matrix::T
    window::K
    th::Float64
    simcarn::Bool
    weight::Float64
end

"""
Takes a list of cluster labels in [1, Inf) and transforms it in a list of lists of
indices/features; also returns a list of labels where labels are referred to the
transformed index of each cluster; this last list is similar to the input but
with sorted labels
"""
function _transform_clusters(clusters::Vector{Int64}, mat::Array{Float64,2})::ClusterResult
    # creating the list of clusters
    N = maximum(clusters)
    L = length(clusters)
    features = [Int64[] for _ in 1:N]
    transformed_clusters = [Int64[] for _ in 1:N]
    reverse = Vector{Int64}(undef, L)
    # a map from older clusters to new (ordered) number of clusters
    new_clusters = zeros(N)
    this_cluster = -1
    counter = 0
    new_cluster = -1
    for i in 1:L
        # for each note
        if clusters[i] != this_cluster
            this_cluster = clusters[i]
            new_cluster = new_clusters[this_cluster]
            if new_cluster == 0
                # a new cluster
                counter += 1
                new_cluster = counter
                new_clusters[this_cluster] = counter
            end
        end
        push!(features[new_cluster], floor(Int64, mat[i, 1]))
        push!(transformed_clusters[new_cluster], i)
        reverse[i] = new_cluster
    end
    return ClusterResult(features, transformed_clusters, reverse)
end

"""
Here, a cluster is a list of pitch. We take the list of clusters and
compare each note inside each cluster to match the corresponding notes. The
match is done based on the features (pitch), but the indices of the
corresponding notes are returned; this is why we need both the list of
cluster features and the list of cluster note indices.
"""
function _matching_notes_clusters(score_clusters, score_features, perfm_clusters,
                                  perfm_features, th::Float64, simcarn::Bool)::Matrix{Int64}
    out = Vector{Matrix{Int64}}(undef, 0)
    for i in 1:length(score_clusters)
        matched_idx1, matched_idx2 = FedeDist.my_intersect_idx(score_features[i],
                                                               perfm_features[i], th,
                                                               simcarn)

        if length(matched_idx1) > 0
            push!(out,
                  hcat(score_clusters[i][matched_idx1], perfm_clusters[i][matched_idx2]))
        end

    end
    return vcat(out...)
end

"""
Returns clusters of notes by onsets using single-linkage and stopping
agglomeration procedure when `threshold` is reached.

Returns the features of each cluster, ordered by onsets, the index of notes
in each cluster, ordered by onset, and the list of cluster of each note.
Features is only one (pitch).

If `num_clusters` is not None, it should be a int and `maxclust` criterion
is used in place of the threshold (which is not used)
"""
@views function _clusterize(mat::Array{Float64,2};
                            threshold::Union{Float64,Nothing}=nothing,
                            num_clusters::Union{Int64,Nothing}=nothing)::ClusterResult
    dist = Matrix{Float64}(undef, size(mat, 1), size(mat, 1))
    Z = Cluster.single_linkage!(pairwise!(dist, Cityblock(), mat[:, 2]))
    if threshold === nothing
        # clusterize the onsets using default linkage method in O(n²)
        clusters = cutree(Z; k=num_clusters)
    else
        clusters = cutree(Z; h=threshold)
    end

    # creating the list of clusters
    return _transform_clusters(clusters, mat)
end

function add_dummy_notes(matscore::Matrix{Float64}, matperfm::Matrix{Float64})
    # inserting starting and ending notes
    s::Float64 = -1.0
    e::Float64 = max(maximum(matscore[:, 3]), maximum(matperfm[:, 3])) + 1
    starting = fill(s, (1, size(matscore, 2)))
    ending = fill(e, (1, size(matscore, 2)))
    matscore = vcat(starting, matscore, ending)
    matperfm = vcat(starting, matperfm, ending)

    return matscore, matperfm
end

function _clustering(matscore, matperfm,
                     score_th::Union{Float64,Nothing}=0.0)::Tuple{ClusterResult,
                                                                  ClusterResult}
    # compute clusters
    perfm_clusters = _clusterize(matperfm; threshold=0.05)
    if score_th !== nothing
        score_clusters = _clusterize(matscore; threshold=score_th)
    else
        score_clusters = _clusterize(matscore; num_clusters=length(perfm_clusters.features))
    end
    return score_clusters, perfm_clusters

end

function _get_dist_data(score::ClusterResult, perfm::ClusterResult, th::Float64,
                        simcarn::Bool, α::Int64, β::Int64)::DistData

    dist_fn = FedeDist.JaccardDist(th, simcarn)
    window_fn = FedeDTW.get_fede_win(FedeDTW.FedeWindowParam(dist_fn, 5, α, β),
                                     score.features, perfm.features)

    dist_mat = _compute_dist_mat(score.features, perfm.features, dist_fn, window_fn)
    weight = 2 / 3
    if simcarn
        weight = 1 - weight
    end
    return DistData(dist_mat, window_fn, th, simcarn, weight)
end

"""
Returns a mapping of indices between notes in `matscore` and in `matperfm`
with shape (N, 2), where N is the number of matching notes.
Also returns `matscore` and `matperfm` with initial and ending virtual
notes: you can safely discard them by slicing with `[1:-1]`, but returned
indices are referred to the returned mats.

If `score_th` is None, then the number of clusters for the score is
inferred from the performance (whose threshold is fixed in `settings.py`)
"""
@views function _get_matching_notes(score::ClusterResult, perfm::ClusterResult,
                                    distdata::DistData,
                                    step_pattern::DTW.StepPattern)::Matrix{Int64}

    path = DTW.dtw_path(distdata.matrix; window_fn=distdata.window,
                        step_pattern=step_pattern)

    if path === nothing
        # # no path available
        # return an empty matrix
        return Matrix{Int64}(undef, 0, 0)
    end
    path1 = path[:, 1]
    path2 = path[:, 2]

    # computing the matched notes
    matched_idx = _matching_notes_clusters(score.clusters[path1], score.features[path1],
                                           perfm.clusters[path2], perfm.features[path2],
                                           distdata.th, distdata.simcarn)

    # compute unique matched notes
    matched_idx = _get_unique_path(matched_idx, score.reverse, perfm.reverse,
                                   distdata.matrix)
    return matched_idx
end

"""
Reapeatly calls ``_get_matching_notes`` and merge the indices.

Returns a mapping of indices between notes in `matscore` and in `matperfm`
with shape (N, 2), where N is the number of matching notes.
Also returns `matscore` and `matperfm` with initial and ending virtual
notes: you can safely discard them by slicing with `[1:-1]`, but returned
indices are referred to the returned mats.

If `score_th` is None, then the number of clusters for the score is
inferred from the performance (whose threshold is fixed in `settings.py`)

`thresholds` is a list of thresholds used; a value of 1 causes simcarn not
being used, a value != 1 causes simcarn to be used

`dist` is the function used for thresholds != 1 (using simcarn)

`sp_weights` must be a matrix where each row is a weight for the
`DTW.weight_step_pattern` function; it should have size (13, N), where N is the
number of desired different step-patterns DTW.

"""
function get_matching_notes(matscore::Array{Float64,2}, matperfm::Array{Float64,2},
                            α::Int64, β::Int64, sp_weights::Matrix{Float64},
                            score_th::Union{Float64,Nothing} = nothing,
                            thresholds::Vector{T}=[1.0, 0.5]) where {T<:Real}
    matscore, matperfm = add_dummy_notes(matscore, matperfm)

    # computing clustering once for all
    clusters::Tuple{ClusterResult,ClusterResult} = _clustering(matscore, matperfm, score_th)

    # computing distances once for each threshold
    distdata = Vector{DistData{Matrix{Float64},FedeDTW.FedeWindowFunc}}(undef,
                                                                        length(thresholds))
    # for (i, th) in collect(enumerate(thresholds))
    Threads.@threads for (i, th) in collect(enumerate(thresholds))
        distdata[i] = _get_dist_data(clusters[1], clusters[2], th, th != 1, α, β)
    end

    # compute step_patterns
    min_weight = sum(@view(sp_weights[end, :])) * 0.01
    step_patterns = [
        DTW.weight_step_pattern(col) for col in eachcol(sp_weights) if col[end] > min_weight
    ]

    config = [(dist, step, step.weight * dist.weight)
              for dist::DistData in distdata, step::DTW.StepPattern in step_patterns]
    matched_indices = Vector{Matrix{Int64}}(undef, length(config))

    # for (i, conf) in collect(enumerate(config))
    Threads.@threads for (i, conf) in collect(enumerate(config))
        matched_indices[i] = _get_matching_notes(clusters[1], clusters[2], conf[1], conf[2])
    end

    # merging matching indices
    matched_idx = FedeDTW.merge_matching_indices(matched_indices, [conf[3] for conf in config])
    return copy(matched_idx), matscore, matperfm
end

function get_matching_notes(matscore::Array{Float64,2}, matperfm::Array{Float64,2},
                            α::PyObject, β::PyObject,
                            sp_weights::Matrix{Float64},
                            score_th=Union{Float64,Nothing} = nothing,
                            thresholds::Vector{T}=[1.0, 0.5]) where {T<:Real}
    _α::Int64 = convert(Int64, α)
    _β::Int64 = convert(Int64, β)
    # _sp_weights = convert(Matrix{Float64}, sp_weights)
    return get_matching_notes(matscore, matperfm, _α, _β, sp_weights, score_th, thresholds)
end

"""
Just a simple function that you can use to precompile the core functions
"""
function precompile(path1, path2)
    matscore = readdlm(path1, ',', Float64)
    matperfm = readdlm(path2, ',', Float64)
    clusters::Tuple{ClusterResult,ClusterResult} = _clustering(matscore, matperfm, nothing)
    clusters = _clustering(matscore, matperfm, 0.05)
    step = DTW.get_default_step_patterns()[1]
    distdata = [_get_dist_data(clusters[1], clusters[2], 0.5, true, 5, 1),
                _get_dist_data(clusters[1], clusters[2], 1.0, false, 1, 1)]
    m1 = _get_matching_notes(clusters[1], clusters[2], distdata[1], step)
    m2 = _get_matching_notes(clusters[1], clusters[2], distdata[2], step)
    return matched_idx = FedeDTW.merge_matching_indices([m1, m2], [1.0, 1.0])
end
end # module
