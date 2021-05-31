import numpy as np
import sklearn
import skopt
from skopt.space.space import Integer
from tqdm import trange

from . import settings as s
from .alignment_fede import include_julia
from .asmd.asmd import asmd, dataset_utils
from .evaluate_midi2midi import MissingExtraManager, eval_fede
from .pybnn_module import dngo

DATASET_LEN = 0.05
STEP_WEIGHTS = 13
NWEIGHT = 4 * STEP_WEIGHTS
TOT_WEIGHTS = NWEIGHT + 2


def objective(hparams):
    s.ALPHA = hparams[0]
    s.BETA = hparams[1]
    s.SP_WEIGHT = np.asarray(hparams[2:], dtype=np.float64).reshape(
        (-1, STEP_WEIGHTS)).T
    print("---------")
    print(f"testing a={s.ALPHA}, b={s.BETA}, sp_weights=\n{s.SP_WEIGHT}")
    dataset = asmd.Dataset()
    dataset, _ = dataset_utils.choice(dataset,
                                      p=[DATASET_LEN, 1 - DATASET_LEN],
                                      random_state=1992)

    global process

    def process(i, dataset):
        mem = MissingExtraManager(dataset, i)
        if mem.num_notes > s.MAX_NOTES or mem.num_notes < s.MIN_NOTES:
            return None
        return eval_fede(mem)

    res = [process(i, dataset) for i in trange(len(dataset))]
    # res = dataset.parallel(process, n_jobs=5, backend='multiprocessing')
    # removing nones
    res = [r.match_fmeasure for r in res if r is not None]
    f1m = np.mean(res)
    print(f"Result: {f1m:.2e}")
    print(f"N. of tests are {len(res)}")
    print("---------")
    return 1 - f1m


class BOFactor(object):
    """
    A float which:

    * updates its counter at each use 
    * uses `func(counter)` instead of a fixed value
    """

    def __init__(self, func):
        self.counter = 0
        self.func = func

    def __add__(self, other):
        self.counter += 1
        return self.func(self.counter) + other

    def __radd__(self, other):
        self.counter += 1
        return self.func(self.counter) + other

    def __mul__(self, other):
        self.counter += 1
        return self.func(self.counter) * other

    def __rmul__(self, other):
        self.counter += 1
        return self.func(self.counter) * other

    def __sub__(self, other):
        return self.func(self.counter) - other

    def __rsub__(self, other):
        self.counter += 1
        return other - self.func(self.counter)

    def __truediv__(self, other):
        self.counter += 1
        return self.func(self.counter) / other

    def __rtruediv__(self, other):
        self.counter += 1
        return other / self.func(self.counter)


class SKDNGO(sklearn.base.RegressorMixin, dngo.DNGO):
    def __init__(self, do_optimize, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params = dict()
        self.do_optimize = do_optimize
        for name, val in vars().items():
            if name != "self":
                self.params[name] = val

    def get_params(self, *args, **kwargs):
        return self.params

    def set_params(self, *args, **kwargs):
        self.__init__(*args, **kwargs)
        return self

    def predict(self, x, return_std=False):
        x = np.asarray(x)
        if x.ndim() < 2:
            x = x[None]
        m, v = super(dngo.DNGO).predict(x)
        if return_std:
            return m, v
        else:
            return m

    def fit(self, xx, yy):
        return super(dngo.DNGO).train(np.asarray(xx), np.asarray(yy),
                                      self.do_optimize)


def update_kappa(c):
    # kappa = 0.6 + np.exp(-np.sqrt(c) / (2 * np.pi**2))
    kappa = 2 * np.pi * np.exp(-np.sqrt(c) / (2 * np.pi**2)) - 0.2
    print(f"kappa: {kappa:.2f}")
    return kappa


# TODO:
# instantiate the model and use `EI` acquisition function

if __name__ == "__main__":

    include_julia("alignment/alignment_fede.jl")

    # yapf: disable
    x0 = [
        [
            150, 9,
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

            # 1, 1, 1, 0,
            # 1, 1, 1, 0,
            # 1, 1, 2, 0,
            # 2,

            # 1, 1, 1, 0,
            # 1, 2, 1, 0,
            # 1, 1, 1, 0,
            # 2,
        ],
    ]
    # yapf: enable

    s.MAX_NOTES = 2000
    s.MAX_DURATION = 120

    res = skopt.optimizer.base.base_minimize(
        func=objective,
        dimensions=[
            Integer(2, 300, name='alpha', transform='normalize'),
            Integer(2, 40, name='beta', transform='normalize'),
        ] + [
            Integer(0, 5, name=f'step{i}', transform='normalize')
            for i in range(NWEIGHT)
        ],
        base_estimator=SKDNGO(False),  # skopt.learning.RandomForestRegressor(n_estimators=10),
        # base_estimator='ET',  # skopt.learning.RandomForestRegressor(n_estimators=10),
        n_calls=2000,
        n_initial_points=2,
        random_state=1750,
        acq_func="EI",
        xi=0.01,
        # kappa=0.7,
        # kappa=BOFactor(update_kappa),
        acq_optimizer="sampling",
        n_points=10**4,
        verbose=True,
        callback=[
            skopt.callbacks.CheckpointSaver("fede_model_dngo.pkl"),
            skopt.callbacks.DeltaYStopper(0.01, 40),
        ],
        x0=x0,
        n_jobs=-1)

    print("----------------")
    print(f"Best value: {res['fun']:.2e}")
    print(f"Point: {res['x']}")
