import numpy as np
from skopt import forest_minimize
from skopt.space.space import Categorical, Integer

from . import settings as s
from .align import align_to_midi
from .alignment_tafe import tafe_align
from .asmd.asmd import asmd, dataset_utils
from .utils import onsets_clusterize

DATASET_LEN = 0.05
N_JOBS = -1

rng = np.random.default_rng(1985)


# hack to let fastdtw accept float32
def _my_prep_inputs(x, y, dist):
    return x, y


# def process(i: int, dataset: asmd.Dataset, radius: int, dist: str, step:
#             str):
def process(i: int, dataset: asmd.Dataset, radius: int, dist: str):

    score, extra = dataset_utils.get_score_mat(dataset,
                                               i,
                                               score_type=['misaligned'],
                                               return_notes='extra')
    perfm, missing = dataset_utils.get_score_mat(
        dataset,
        i,
        score_type=['precise_alignment', 'broad_alignment'],
        return_notes='missing')
    # skipping long scores
    if score.shape[0] > s.MAX_NOTES or dataset.get_score_duration(
            i) > s.MAX_DURATION:
        return None
    score = onsets_clusterize(score, rng)

    ground_truth = dataset_utils.get_score_mat(
        dataset, i, score_type=['precise_alignment', 'broad_alignment'])
    ground_truth = ground_truth[:, 1:3]

    audio, sr = dataset.get_audio(i)

    # missing notes
    score = score[~extra]
    ground_truth = ground_truth[~extra]
    perfm = perfm[~missing]

    # alignment
    tafe_alignment = align_to_midi(
        np.copy(score), np.copy(perfm),
        # lambda x, y: tafe_align(x, y, radius=radius, dist=dist, step=step))
        lambda x, y: tafe_align(x, y, radius=radius, dist=dist))

    tafe_err = np.mean(np.abs(ground_truth - np.stack(tafe_alignment)[:, 1:3]))
    return tafe_err


def objective(hparams):

    # radius, dist, step = hparams
    radius, dist = hparams

    print("---------")
    print(hparams)

    dataset = asmd.Dataset()
    dataset, _ = dataset_utils.choice(dataset,
                                      p=[DATASET_LEN, 1 - DATASET_LEN],
                                      random_state=1992)

    results = dataset.parallel(process,
                               radius,
                               dist,
                               # step,
                               n_jobs=N_JOBS,
                               backend="multiprocessing")
    results = [res for res in results if res is not None]
    print(f"Used {len(results)} songs")
    return np.mean(results)


if __name__ == "__main__":

    dists = [
        'cosine', 'euclidean', 'canberra', 'chebyshev', 'braycurtis',
        'correlation', 'manhattan'
    ]
    # steps = [
    #     'symmetric1', 'symmetric2', 'symmetricP0', 'symmetricP05', 'symmetricP1',
    #     'symmetricP2', 'typeId', 'typeIId']

    res = forest_minimize(objective,
                          dimensions=[
                              Integer(1, 200, name='radius'),
                              Categorical(dists, name='dist'),
                              # Categorical(steps, name='steps'),
                          ],
                          n_calls=180,
                          random_state=1750,
                          verbose=True,
                          n_jobs=-1)
    print("----------------")
    print(f"Best value: {res['fun']:.2e}")
    print(f"Point: {res['x']}")
