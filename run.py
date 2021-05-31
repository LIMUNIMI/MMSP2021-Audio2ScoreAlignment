"""
Runs different kinds of alignments
"""

import argparse
import sys

import numpy as np

from alignment import align, alignment_eita, alignment_fede, alignment_tafe
from alignment import settings as s
from alignment import utils
from alignment.asmd.asmd.utils import open_audio
from Cython.Build import Cythonize

if s.BUILD:
    Cythonize.main(["alignment/*_alignment.py", "-3", "--inplace"])

argparser = argparse.ArgumentParser(
    f"Runs different types of alignments. A midi file named `{s.OUT_PATH}` is created in the current directory."
)

argparser.add_argument(
    "-m",
    "--midi",
    action="store_true",
    help="Performe midi2midi alignment instead of audio2midi alignment")

argparser.add_argument(
    "-a",
    "--algo",
    nargs=1,
    type=str,
    help=
    "The algorithm to use for midi-to-midi alignment; one among: `fede`, `eita`, `tafe`."
)

argparser.add_argument(
    "-f",
    "--feature-extraction",
    nargs=1,
    type=str,
    help=
    "The algorithm to used for feature-extraction from audio; one among: `bytedance`, `omnizart`. Not used if `-m` is used"
)

argparser.add_argument(
    "-r",
    "--reference",
    nargs=1,
    type=str,
    help=
    "The input that is used as reference; it should be an audio file (e.g. wav, mp3, flac, aiff, etc) or a midi file if `-m` is used"
)
argparser.add_argument(
    "-t",
    "--target",
    nargs=1,
    type=str,
    help=
    "The input that is used as target;  it should be a midi file. The output will contain the pitches in this midi file realigned to the reference"
)


def main(args):
    # loading the target
    target: np.ndarray = utils.midi2mat(args.target[0])

    # load the reference
    ref: np.ndarray
    if args.midi:
        ref = utils.midi2mat(args.reference[0])
    else:
        # here ref is Tuple[np.ndarray, int]
        ref = open_audio(args.reference[0])

    if args.algo[0] == 'tafe':
        midi2midi_method = alignment_tafe.tafe_align
    elif args.algo[0] == 'fede':
        midi2midi_method = alignment_fede.fede_align
    elif args.algo[0] == 'eita':
        midi2midi_method = alignment_eita.eita_align
    else:
        print("Algorithm not known!", file=sys.stderr)
        sys.exit(1)

    if args.midi:
        new_mat = align.align_to_midi(target, ref, midi2midi_method, audio_data=ref)
    else:
        new_mat = align.align_to_audio(target, ref, args.feature_extraction[0],
                                       midi2midi_method)

    utils.mat2midi(new_mat, s.OUT_PATH)
    if args.midi:
        ref[:, 3] = 64
        utils.mat2midi(ref, 'ref' + s.OUT_PATH)


if __name__ == "__main__":
    args = argparser.parse_known_args()[0]
    main(args)
