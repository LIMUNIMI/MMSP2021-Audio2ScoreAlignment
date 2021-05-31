import argparse
import os

from . import alignment_stats
from .conversion_tool import create_gt

THISDIR = os.path.dirname(os.path.realpath(__file__))

argparser = argparse.ArgumentParser(
    description='Generate ASMD ground-truth from other sources')

argparser.add_argument(
    '-m',
    '--misalign',
    action='store_true',
    help="Generate ground-truth artificial misalignment using a trained model; train it if not available")

argparser.add_argument(
    '-n',
    '--normal',
    action='store_true',
    help="Generate ground-truth w/o artificial misalignment")

argparser.add_argument(
    '-t',
    '--train',
    action='store_true',
    help="Collect alignment stats on the already generated ground-truth")

argparser.add_argument(
    '-w',
    '--whitelist',
    help=
    "List of datasets that will not be excluded from the generation (not from the training (default: all)",
    nargs='*')

argparser.add_argument(
    '-b',
    '--blacklist',
    help=
    "List of datasets that will be excluded from the generation not from the training (default: empty). Overwrites `--whitelist`",
    nargs='*')

args = argparser.parse_args()

if args.train:
    if os.path.exists(alignment_stats.FILE_STATS):
        os.remove(alignment_stats.FILE_STATS)
    alignment_stats.get_stats(train=True)

if args.normal:
    stats = None
    create_gt(os.path.join(THISDIR, 'datasets.json'),
              gztar=True,
              alignment_stats=stats,
              whitelist=args.whitelist,
              blacklist=args.blacklist)

if args.misalign:
    stats = alignment_stats.get_stats(train=True)
    create_gt(os.path.join(THISDIR, 'datasets.json'),
              gztar=True,
              alignment_stats=stats,
              whitelist=args.whitelist,
              blacklist=args.blacklist)
