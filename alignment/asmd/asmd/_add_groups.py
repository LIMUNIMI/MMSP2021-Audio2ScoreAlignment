"""
This script is used only once to add the `groups` tag to all the definitions.
Leave it here for future reference for similar situations...
"""
import json
import pathlib

from . import asmd

MAESTRO_JSON = "maestro-v2.0.0.json"


def search_audio_filename_in_original_maestro(filename: str, maestro: dict):
    for song in maestro:
        if song["audio_filename"] == filename:
            return song["split"]
    return None


def maestro_splits():
    """
    Get list of indices for each split. Stolen from my work on Perceptual
    Evaluation of AMT Resynthesized.

    Leve here for reference.
    """
    d = asmd.Dataset().filter(datasets=['Maestro'])

    maestro = json.load(open(MAESTRO_JSON))
    train, validation, test = [], [], []
    for i in range(len(d)):
        filename = d.paths[i][0][0][23:]
        split = search_audio_filename_in_original_maestro(filename, maestro)
        if split == "train":
            train.append(i)
        elif split == "validation":
            validation.append(i)
        elif split == "test":
            test.append(i)
        else:
            raise RuntimeError(filename +
                               "  not found in maestro original json")
    return train, validation, test


def main():
    """
    1. copy maestro-v2.0.0.json in the working dir
    2. run this script with:

    >>> python -m asmd._add_groups
    """
    maestro = json.load(open(MAESTRO_JSON))
    for fname in pathlib.Path('.').glob("asmd/definitions/**/*.json"):
        # only Python >= 3.6
        data = json.load(open(fname, "r"))
        for song in data['songs']:
            if data['name'] == 'Maestro':
                wav_name = song['recording']['path'][0][23:]
                song['groups'] = [
                    search_audio_filename_in_original_maestro(
                        wav_name, maestro)
                ]
            else:
                song['groups'] = []
        # only Python >= 3.6
        # pretty printing stolen from official docs
        json.dump(data, open(fname, "w"), sort_keys=True, indent=4)

if __name__ == "__main__":
    main()
