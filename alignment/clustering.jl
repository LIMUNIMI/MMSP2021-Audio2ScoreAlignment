module Cluster

using Clustering: MinimalDistance, nnodes, nearest_neighbor, update_distances_upon_merge!,
                  push_merge!, HclustMerges, orderbranches_r!, Hclust

"""
Code copied from `Clustering.jl` but this saves RAM by:
* working in place
* using views

---

Efficient single link algorithm, according to Olson, O(n^2), fig 2.
Verified against R's implementation, correct, and about 2.5 x faster
For each i < j compute D(i,j) (this is already given)
For each 0 < i ≤ n compute Nearest Neighbor NN(i)
Repeat n-1 times
  find i,j that minimize D(i,j)
  merge clusters i and j
  update D(i,j) and NN(i) accordingly
"""
@views function single_linkage!(d::AbstractMatrix{T}) where {T<:Real}
    mindist = MinimalDistance(d)
    hmer = HclustMerges{T}(size(d, 1))
    n = nnodes(hmer)
    ## For each 0 < i ≤ n compute Nearest Neighbor NN[i]
    NN = [nearest_neighbor(d, i, n)[1] for i in 1:n]
    ## the main loop
    trees = collect(-(1:n))  # indices of active trees, initialized to all leaves
    while length(trees) > 1  # O(n)
        # find a pair of nearest trees, i and j
        i = 1
        NNmindist = i < NN[i] ? d[i, NN[i]] : d[NN[i], i]
        for k in 2:length(trees) # O(n)
            dist = k < NN[k] ? d[k, NN[k]] : d[NN[k], k]
            if dist < NNmindist
                NNmindist = dist
                i = k
            end
        end
        j = NN[i]
        if i > j
            i, j = j, i     # make sure i < j
        end
        last_tree = length(trees)
        update_distances_upon_merge!(d, mindist, i -> 0, i, j, last_tree)
        trees[i] = push_merge!(hmer, trees[i], trees[j], NNmindist)
        # reassign the last tree to position j
        trees[j] = trees[last_tree]
        NN[j] = NN[last_tree]
        pop!(NN)
        pop!(trees)
        ## update NN[k]
        for k in eachindex(NN)
            if NN[k] == j        # j is merged into i (only valid for the min!)
                NN[k] = i
            elseif NN[k] == last_tree # last_tree is moved into j
                NN[k] = j
            end
        end
        ## finally we need to update NN[i], because it was nearest to j
        NNmindist = typemax(T)
        NNi = 0
        for k in 1:(i - 1)
            if (NNi == 0) || (d[k, i] < NNmindist)
                NNmindist = d[k, i]
                NNi = k
            end
        end
        for k in (i + 1):length(trees)
            if (NNi == 0) || (d[i, k] < NNmindist)
                NNmindist = d[i, k]
                NNi = k
            end
        end
        NN[i] = NNi
    end
    orderbranches_r!(hmer)
    return Hclust(hmer, :single)
end
end # module
