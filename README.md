# Audio-to-Score Alignment Using Deep Automatic Music Transcription

This is the code connected with paper:

F. Simonetta, S. Ntalampiras, and F. Avanzini, "Audio-to-Score Alignment
Using Deep Automatic Music Transcription," in Proceeddings of the IEEE
MMSP 2021, 2021. [Link to PDF](https://arxiv.org/abs/2107.12854)

# !! Errata Corridge !!

After the publication at [MMSP2021](https://attend.ieee.org/mmsp-2021/), an
[error](https://github.com/LIMUNIMI/MMSP2021-Audio2ScoreAlignment/issues/1) was
found in the code. I have rerun all the experiments and updated the publication
on [Arxiv](https://arxiv.org/abs/2107.12854).

In a few words, most of the conclusions of the original publication
hold, but I found that in non-piano music, the *TAFE* and *Bytedance
EIFE* are better than the old *SEBA* method, as published in the paper.

The following are the updated images. They are contained in the
`mlruns` directory (see *Audio-to-midi* section):

1. Piano w/o missing notes:
    1.  [Onsets](./mlruns/1/ebda0e6e83d94ccf9b882b76bee52a8b/artifacts/macro_thresholds_ons.html)
    2.  [Offsets](./mlruns/1/ebda0e6e83d94ccf9b882b76bee52a8b/artifacts/macro_thresholds_offs.html)

2. Piano w/ missing notes:
    1.  [Onsets](./mlruns/1/98840cd63a744f70b680099ffcf337e3/artifacts/macro_thresholds_ons.html)
    2.  [Offsets](./mlruns/1/98840cd63a744f70b680099ffcf337e3/artifacts/macro_thresholds_offs.html)

3. Multi w/o missing notes:
    1.  [Onsets](./mlruns/1/60057c58d2314841b4e88d57352e7a1f/artifacts/macro_thresholds_ons.html)
    2.  [Offsets](./mlruns/1/60057c58d2314841b4e88d57352e7a1f/artifacts/macro_thresholds_offs.html)

4. Multi w/ missing notes:
    1.  [Onsets](./mlruns/1/71fd8329d43d47a69166a56375ba03e2/artifacts/macro_thresholds_ons.html)
    2.  [Offsets](./mlruns/1/71fd8329d43d47a69166a56375ba03e2/artifacts/macro_thresholds_offs.html)

## Setup

### Python

1.  Install pyenv: `curl https://pyenv.run | bash; exec $SHELL`
2.  Install python 3.8.6:
    `PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install 3.8.6`
3.  Install poetry:
    `curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -`
4.  Install project dependencies: `poetry install --no-root`
5.  Install magenta with pip (it breaks dependencies, but we don\'t mind
    about them): `poetry run pip install magenta`

### Julia and C/CPP

1.  Install a gcc or other C/C++ compiler
2.  Install Julia 1.6.0
3.  Run `poetry run python setup.py build_ext --inplace` to build C
    extensions using Cython and Julia
4.  Compile Eita Nakamura code with `./eita_tool/compile.sh`

N.B. If Julia has troubles installing, try to install python with the
command above, using `PYTHON_CONFIGURE_OPTS` environment variable; you
may need to clean the Julia environmnet (`rm -r ~/.julia/environmnts`)

Note that Julia is only needed for code not referenced in the paper.

### Other dependencies

1.  `poetry run omnizart download-checkpoints` to download omnizart
    models
2.  Install command `fluidsynth` in your PATH to test [seba]{.title-ref}
    method
3.  `poetry run python -m alignment.seba.align` to download soundfont
    for [seba]{.title-ref} method

## Datasets

1.  To install datasets from ASMD, run
    `poetry run python -m alignment.asmd.asmd.install` and follow
    instructions
2.  At the end of the procedure, you\'ll be able to download the
    generated ground-truth from the web; however, you can still
    regenerate them from scratch by running
    -   `poetry run python -m alignment.asmd.asmd.generate_ground_truth --normal`
    -   `poetry run python -m alignment.asmd.asmd.generate_ground_truth --misalign`
3.  If you want to recompute statistics, use the `--train` flag; note
    that the result will likely be slightly different because statistics
    are computed on songs where Eita method takes less than 20 seconds,
    and this varies based on your machine computational power.

## Evaluation

### Datasets

To evaluate the artificially misalignment approach, use:
`poetry run python -m alignment.asmd.asmd.alignment_stats`

Note that results will likely be different from ours, because the
statistics are computed on songs for which the Eita method takes less
than 20 seconds. As such, the statistics depend on your computational
power.

The following are the L1 errors between the generated data and the
matching notes in the real score in our test-set:

  ------ --------------- ---------------
         Ons             Offs

  HMM    18.6 ± 49.7     20.7 ± 50.6

  Hist   7.43 ± 15.5     8.95 ± 15.5
  ------ --------------- ---------------

When not sorting nor fixing offsets, the HMM worked better than Hist,
but I only computed DTW normalized distance; results are in the ASMD
repo (old commits).

### Audio-to-midi

1.  To evaluate audio-to-score alignment without missing/extra notes on
    music without solo piano, use:
    `poetry run python -m alignment.evaluate_audio2score`
2.  To simulate missing/extra notes use the flag `--missing`
3.  To do the same tests on solo piano music, use the flag `--piano`
4.  You can select ASMD datasets by using option `--dataset`
5.  To do experiments published in the paper in one pass, use
    `poetry run ./evaluate_audio2score.sh`

Results are shown in mlflow, so you need to run `mlflow ui` and access
it from your browser.

You can also see results from our evaluations by using `mlflow ui`.

Finally, you can see further statistics by reading the content of files
with `.notes` extensions.
