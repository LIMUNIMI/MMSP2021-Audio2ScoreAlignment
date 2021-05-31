using Distributed: pmap, addprocs, nworkers, @everywhere
@everywhere using Infiltrator
const NUM_DTW = 6
const NWORKERS = 3

if nworkers() < NWORKERS
    new_workers = NWORKERS - nworkers() + 1
    println("adding $new_workers workers")
    # number of threads must be set vie env variable...
    ENV["JULIA_NUM_THREADS"] = 2 * NUM_DTW
    # note the `--project=.`
    addprocs(new_workers; exeflags="--project=.")
end

@everywhere begin
    using ProgressMeter: @showprogress
    using PyCall
    using Statistics: mean
    using Printf: @sprintf
    include("alignment_fede.jl")

    const DATASET_LEN = 0.05
    const MIN_NOTES = 0
    const MAX_NOTES = 2000
    const MAX_DURATION = 120
    const STEP_WEIGHTS = 13
    const NWEIGHT = $NUM_DTW * STEP_WEIGHTS
    const TOT_WEIGHTS = NWEIGHT + 2
end

@everywhere py"""
import sys
sys.path.insert(0, ".")
import skopt
import numpy as np
from alignment import fede_tuning
from alignment.evaluate_midi2midi import MissingExtraManager, Result
from alignment.asmd.asmd import asmd, dataset_utils
from skopt.space.space import Integer
"""

@everywhere const full_dataset = py"asmd.Dataset()"
@everywhere const dataset, _ = py"dataset_utils.choice"(full_dataset,
                                                        p=[DATASET_LEN, 1 - DATASET_LEN],
                                                        random_state=1992)

@everywhere function objective(hparams)
    α = convert(Int64, hparams[1])
    β = convert(Int64, hparams[2])
    # python was major mode, here each column is a set of weights, instead
    sp_weights = reshape(convert(Vector{Float64}, hparams[3:end]), STEP_WEIGHTS, :)
    # display("text/plain", sp_weights)

    L = convert(Int64, py"len"(dataset))
    println("Size of dataset: $L")
    res = Vector{Float64}(undef, 0)
    @showprogress 1 "Evaluating..." for i in 0:(L - 1)
        mem = py"MissingExtraManager"(dataset, i)
        if mem.num_notes > MAX_NOTES || mem.num_notes < MIN_NOTES
            continue
        else
            push!(res, eval_fede(mem, α, β, sp_weights))
        end
    end
    f1m = mean(res)
    println(@sprintf("Result: %.2e", f1m))
    println("N. of tests are $(length(res))")
    return 1 - f1m
end

@everywhere @views function eval_fede(mem, α, β, sp_weights)::Float64
    score = mem.get_score()
    perfm = mem.get_perfm()
    fede_match, _, _ = FedeAlignment.get_matching_notes(score, perfm, α, β, sp_weights, 0.0)
    if length(fede_match) > 0
        #transpose
        fede_match = permutedims(fede_match)
        mask = trues(size(fede_match)...)
        for i in 1:2
            # compute argmin and argmax
            m = argmin(fede_match[i, :])
            M = argmax(fede_match[i, :])
            # remove their column
            mask[i, m] = false
            mask[i, M] = false
        end
        fede_match = hcat(fede_match[1, mask[1, :]], fede_match[2, mask[2, :]])
    end
    fede_res = mem.evaluate(fede_match .- 2, "fede", 0.0)
    f1m = convert(Float64, fede_res.match_fmeasure)
    return f1m
end

struct EarlyStop
    range::Float64
    patience::Int64
end

function (self::EarlyStop)(res::Dict)
    m = minimum(res["func_vals"])
    vals = res["func_vals"] .- m
    if count(<(self.range), vals) > self.patience
        println("Early-stopping!")
        return true
    else
        return false
    end
end

function callbacks(res::Dict; early_stop::Union{Nothing,EarlyStop}=nothing)::Bool
    # early stopping
    if early_stop !== nothing
        return early_stop(res)
    end
    return false
end

function log(res, nworkers)
    println("----------------")
    println("""Total number of calls: $(length(res["func_vals"]))""")
    println("Values obtained: ")
    display("text/plain", res["func_vals"][(end - nworkers + 1):end])
    return println(@sprintf("Best val so far: %.2e", res["fun"]))
end

function tell!(optimizer, xx, yy, nworkers)
    # xx and yy are Julia vectors, that can be translated to numpy arrays or
    # lists; python lists are translated to Julia arrays. Moreover,
    # sometimes only the outer vector is translated into a list, while the
    # inners remain arrays, or the opposite.
    # so, let's do everything in python
    py"""
    xx = [x.tolist() if type(x) is np.ndarray else x for x in $xx]
    yy = [y.tolist() if type(y) is np.ndarray else y for y in $yy]
    res = $optimizer.tell(xx, yy)
    """
    res = py"res"
    log(res, nworkers)
    return res
end

"""
Perform `n_iter * n_jobs` calls and returns the sklearn ResultObject
"""
function optimize(optimizer::PyObject, fn::Function, n_iter; nworkers=NWORKERS,
                  early_stop::Union{Nothing,EarlyStop}=nothing,
                  x0::Vector{Vector{T}}=[]) where {T}

    for x in x0
        println("Testing provided initial point x=")
        println(x)
        y = fn(x)

        res = tell!(optimizer, [x], [y], 1)
    end
    for n in 1:n_iter
        println("----------------")
        println("Iteration number $n")

        if nworkers == 1
            xx = [optimizer.ask(n_points=nworkers)[1, :]]
        else
            xx = optimizer.ask(n_points=nworkers)
        end
        println("Testing x=")
        display("text/plain", xx)
        if nworkers == 1
            yy = [fn(xx[1])]
        else
            yy = pmap(fn, eachrow(xx))
        end

        res = tell!(optimizer, xx, yy, nworkers)
        if callbacks(res; early_stop=early_stop)
            break
        end
    end

    return optimizer.get_result()

end

function main()
#! format: off
    # x0 = [
    #     [
    #         20, 13,
    #         1, 3,
    #         1, 2,
    #         1, 1,
    #         1, 1, 3,
    #         1, 1, 2,
    #         2, 1,
    #         1, 3, 1,
    #         1, 2, 1,
    #         3, 1,
    #         1,

    #         1, 2,
    #         1, 3,
    #         1, 2,
    #         1, 2, 3,
    #         1, 2, 2,
    #         1, 1,
    #         1, 3, 2,
    #         1, 2, 1,
    #         2, 1,
    #         1,

    #         1, 2,
    #         1, 1,
    #         1, 3,
    #         1, 1, 3,
    #         1, 2, 2,
    #         1, 1,
    #         1, 3, 2,
    #         1, 2, 1,
    #         2, 1,
    #         1,

    #         2, 1,
    #         2, 1,
    #         2, 2,
    #         2, 1, 1,
    #         1, 2, 1,
    #         1, 2,
    #         1, 1, 2,
    #         1, 2, 1,
    #         1, 1,
    #         1,
         
    #         1, 3,
    #         1, 3,
    #         2, 1,
    #         2, 1, 3,
    #         1, 2, 1,
    #         1, 2,
    #         1, 1, 2,
    #         1, 1, 1,
    #         2, 1,
    #         1,

    #     ],
    # ]
    x0 = [
        [
            18, 9,
            1, 3, 1, 0,
            1, 2, 1, 0,
            1, 1, 1, 0,
            1,

            1, 1, 1, 0,
            1, 1, 1, 0,
            1, 1, 2, 0,
            1,

            1, 1, 1, 0,
            1, 2, 1, 0,
            1, 1, 1, 0,
            1,

            2, 1, 1, 0,
            1, 1, 1, 0,
            1, 1, 1, 0,
            1,

            1, 1, 1, 0,
            1, 1, 1, 0,
            1, 1, 2, 0,
            2,

            1, 1, 1, 0,
            1, 2, 1, 0,
            1, 1, 1, 0,
            2,
        ],
    ]
    optimizer = py"skopt.optimizer.Optimizer"(
        vcat([
            py"Integer"(10, 30, name="alpha", transform="normalize"),
            py"Integer"(5, 20, name="beta", transform="normalize")
        ], [
            py"Integer"(0, 7, name="step$i", transform="normalize") for i in 1:NWEIGHT
        ]);
        # base_estimator=SKDNGO(False),
        # base_estimator=py"skopt.learning.RandomForestRegressor"(n_estimators=10),
        # base_estimator="GBRT",
        base_estimator=py"skopt.learning.ExtraTreesRegressor"(
            n_estimators=1, min_samples_leaf=5, warm_start=true),
        # base_estimator="ET",
        # n_calls=4000,
        n_initial_points=100, random_state=1750,
        acq_func="EI",
        acq_func_kwargs=Dict("xi" => 0.1),
        # kappa=0.7,
        # kappa=BOFactor(update_kappa),
        acq_optimizer="sampling",
        # n_points=10^4,
        # verbose=true,
        # callback=[
        #     # py"skopt.callbacks.CheckpointSaver"("fede_model_dngo.pkl"),
        #     py"skopt.callbacks.DeltaYStopper"(0.01, 40),
        # ],
        # x0=x0,
        n_jobs=-1)
#! format: on
    res = optimize(optimizer, objective, 2000; early_stop=EarlyStop(0.01, 40), x0=x0)

    println("----------------")
    println(@sprintf("Best value: %.2e", res["fun"]))
    return println("""Point: $(res["x"])""")
end
