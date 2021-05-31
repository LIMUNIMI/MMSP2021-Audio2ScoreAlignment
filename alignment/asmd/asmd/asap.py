"""
This is a simple script to import scores from ASAP
"""

import argparse
import csv
import json
import tempfile
import zipfile
from pathlib import Path
from typing import List, Mapping, Set, Tuple
from urllib.request import urlretrieve, urlcleanup
from alive_progress import alive_bar
import shutil

from .asmd import Dataset
from .dataset_utils import filter
from .idiot import THISDIR

ASAP_URL = "https://github.com/fosfrancesco/asap-dataset/archive/v1.1.zip"


def modify_maestro_definifion(index: List[Tuple[Path, Path]]) -> Mapping:
    """
    This function was run only once to add the proper group `asap` to the
    `Maestro` dataset
    """
    # create a daset for loading the Maestro definition
    dataset = Dataset()
    for definition in dataset.datasets:
        if definition['name'] == 'Maestro':
            break

    # convert index to Set of string for faster search (the `in` operation)
    _index: List[str] = [str(e[0]) for e in index]

    install_dir = Path(dataset.install_dir)
    # add `asap` to each song with ground_truth in the index
    for song in definition['songs']:
        path = str((install_dir / song['recording']['path'][0]).with_suffix(''))
        if path in _index:
            song['groups'].append("asap")
        del song['included']

    del definition['included']
    return definition


def download_asap() -> tempfile.TemporaryDirectory:
    """
    Download ASAP from github. return the Path to the downloaded dir
    """
    # downloading
    print("Downloading ASAP")

    asap_dir = tempfile.TemporaryDirectory()
    with alive_bar(unknown='notes2', spinner='notes_scrolling') as bar:
        temp_fn, _header = urlretrieve(ASAP_URL,
                                       reporthook=lambda x, y, z: bar)
    print("Uncompressing ASAP")
    with zipfile.ZipFile(temp_fn, 'r') as zip_ref:
        zip_ref.extractall(str(asap_dir))

    urlcleanup()
    return asap_dir


def make_index(asap_path: Path) -> List[Tuple[Path, Path]]:
    """
    Generate a list of tuples with values:
        Maestro midi paths, ASAP midi score path
    """
    # a random path inside asmd
    dataset = Dataset()
    asmd_maestro_random_path = filter(dataset,
                                      datasets=['Maestro']).get_gts_paths(0)[0]
    # the second occurrence of `/` in the random path
    _idx = asmd_maestro_random_path.index('/',
                                          asmd_maestro_random_path.index('/') + 1)

    # construct path to asmd Maestro
    asmd_maestro = Path(dataset.install_dir) / asmd_maestro_random_path[:_idx]

    out: List[Tuple[Path, Path]] = []
    # this glob allows to abstracting over directory structure and names
    for fname in asap_path.glob('**/metadata.csv'):
        with open(fname) as f:
            for row in csv.DictReader(f):
                maestro_path = row['maestro_midi_performance']
                if maestro_path:
                    out.append(
                        (Path(maestro_path.replace('{maestro}',
                                                   str(asmd_maestro))).with_suffix(''),
                         fname.parent / row['midi_score']))

    return out


def copy_scores(index: List[Tuple[Path, Path]]):
    """
    Moves the scores in `index` to the Maestro path using `.score.mid`
    extension
    """

    # moving files
    for maestro, asap in index:
        shutil.copy(asap, maestro.with_suffix('.score.mid'))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-m", "--modify", action='store_true')
    args = argparser.parse_args()

    asap_dir = download_asap()
    try:
        index = make_index(Path(str(asap_dir)))
        if args.modify:
            new_def = modify_maestro_definifion(index)
            json.dump(new_def,
                      open(Path(THISDIR) / 'definitions' / 'Maestro.json', 'wt'), indent=4)
        else:
            copy_scores(index)
    finally:
        shutil.rmtree(Path(str(asap_dir)).parent.parent)
