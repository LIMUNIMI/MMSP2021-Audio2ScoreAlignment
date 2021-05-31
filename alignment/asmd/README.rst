Audio-Score Meta-Dataset
========================

ASMD is a framework for installing, using and creating music multimodal
datasets including (for now) audio and scores.

This is the repository for paper [1] 

Read more in the docs_.

* To install: ``pip install asmd``
* To install datasets: ``python -m asmd.install``
* To import API: ``from asmd import audioscoredataset as asd``

Other examples in the paper!

.. _docs: https://asmd.readthedocs.org

Changelog
=========

Version 0.3
^^^^^^^^^^^

#. Fixed MIDI values ([0, 128) for control changes and pitches)
#. Fixed metadata error while reading audio files
#. Fixed pedaling for tracks that have no pedaling
#. Fixed group selection
#. Added `get_songs`
#. Improved initialization of `Dataset` objects
#. Improved documentation

Version 0.2.2-2
^^^^^^^^^^^^^^^

#. Fixed major bug in install script
#. Fixed bug in conversion tool
#. Removed TRIOS dataset because no longer available
#. Updated ground_truth

Version 0.2.2
^^^^^^^^^^^^^

#. Improved ``parallel`` function
#. Improved documentation
#. Various fixings in ``get_pedaling``

Version 0.2.1
^^^^^^^^^^^^^

#. Added ``nframes`` utility to compute the number of frames in a given time lapse
#. Added ``group`` attribute to each track to create splits in a dataset
   (supported in only Maestro for now)
#. Changed ``.pyx`` to ``.py`` with cython in pure-python mode

Version 0.2
^^^^^^^^^^^

#. Added ``parallel`` utility to run code in parallel over a while dataset
#. Added ``get_pianoroll`` utility to get score as pianoroll
#. Added ``sustain``, ``sostenuto``, and ``soft`` to model pedaling information
#. Added utilities ``frame2time`` and ``time2frame`` to ease the development
#. Added ``get_audio_data`` to get data about audio without loading it
#. Added ``get_score_duration`` to get the full duration of a score without
   loading it
#. Added another name for the API: ``from asmd import asmd``
#. Deprecated ``from asmd import audioscoredataset``
#. Changed the ``generate_ground_truth`` command line options
#. Easier to generate misaligned data
#. Improved documentation

Roadmap
=======

#. Added `torch.DatasetDump` for preprocessing datasets and use them in pytorch
#. Add new modalities (video, images)
#. Improve the artificial misalignment
#. Add datasets for the artificial misalignment (e.g. ASAP, Giant-Midi Piano)
#. Add other datasets
#. Refactoring of the API (it's a bit long now...)

Cite us
=======

[1]  Simonetta, Federico ; Ntalampiras, Stavros ; Avanzini, Federico: *ASMD: an automatic framework for compiling multimodal datasets with audio and scores*. In: Proceedings of the 17th Sound and Music Computing Conference. Torino, 2020 arXiv:2003.01958_

.. _arXiv:2003.01958: https://arxiv.org/abs/2003.01958

---

Federico Simonetta 

#. https://federicosimonetta.eu.org
#. https://lim.di.unimi.it
