from pathlib import Path

import numpy as np

from . import fede_dist

EITA_PATH = Path('./eita_tool')

OUT_PATH = 'aligned.mid'

MAX_MEM = 32 * (2**30)  # 32 GB
MAX_TIME = 600  # 10 minutes

# compile to C extensions before running evaluation
BUILD = False
CLEAN = True
# one of 'julia' or 'python'
BACKEND = 'julia'

# Transcription
DEVICE = 'cuda'

# fede alignment settings
PARALLEL = True  # python only
PLOT = False  # python only
FEDE_DIST = fede_dist._jaccard_dist  # python only
FEDE_PITCH_TH = 0.5  # python only
CLUSTER_TH = 0.05  # python only
ALPHA = 144
BETA = 36
SP_WEIGHT = np.asarray([], dtype=np.float64)

# Evaluation
MIN_NOTES = 0
MAX_NOTES = np.inf
MAX_DURATION = np.inf
EVAL_FEDE = False
