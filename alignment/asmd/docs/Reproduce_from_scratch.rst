Reproduce from scratch
======================

To recreate the ground-truth in our format you have to convert the annotations
using the scirpt ``generate_ground_truth.py``.

**N.B. You should have ``wget`` installed in your system, otherwise SMD
dataset can’t be downloaded.**

You can run the script with ``python 3``. You can also skip the already
existing datasets by using the ``--blacklist`` and ``--whitelist`` argument. If
you do this, their ground truth will not be added to the final archive, thus,
remember to backup the previous one and to merge the archives.

Generate misaligned data
------------------------

If you want, you can generate misaligned data using the ``--train`` and
``--misalign`` options of ``generate_ground_truth.py``. It will run
``alignment_stats.py``, which collects data about the datasets with real
non-aligned scores and saves stats in ``_alignment_stats.pkl`` file in the ASMD
module directory. Then, it runs ``generate_ground_truth.py`` using the collected
statistics:  it will generate misaligned data by using the same deviation
distribution of the available non-aligned data. 

Note that misaligned data should be annotated as ``2`` in the ``ground_truth``
value of the dataset groups description (see :doc:`./index` ), otherwise no
misaligned value will be added to the ``misaligned`` field. Moreover, the
dataset group data should have `precise_alignment` or `broad_alignment` filled
by the annotation conversion step, otherwise errors can raise during the
misalignment procedure.

For more info, see ``python -m asmd.generate_ground_truth -h``.

A usual pipeline is:

#. Generate music score data and other ground-truth except artificial one:
   ``python -m asmd.generate_ground_truth --normal``
#. Train a statistical model (can skip this): ``python -m asmd.generate_ground_truth --train``
#. Generate misalignment using the trained model (trains it if not available): ``python -m
   asmd.generate_ground_truth --misalign``
