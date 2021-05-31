Python API
==========

.. toctree::
   :maxdepth: 4
   :caption: Contents:

Intro
-----

This project also provides a few API for filtering the datasets according
to some specified prerequisites and getting the data in a convenient format.

Python
~~~~~~

Import ``audioscoredataset`` and create a ``Dataset`` object, giving the
path of the ``datasets.json`` file in this directory as argument to the
constructor. Then, you can use the ``filter`` method to filter data
according to your needs (you can also re-filter them later without
reloading ``datasets.json``).

You will find a value ``paths`` in your ``Dataset`` instance containing
the correct paths to the files you are requesting.

Moreover, the method ``get_item`` returns an array of audio values and a
structured_array representing the ground_truth as loaded from the json
file.

Example:

.. code:: python

   from asmd import asmd

   d = asmd.Dataset()
   # d = asd.Dataset(paths=['path_to_my_definitions', 'path_to_default_definitions'])
   d.filter(instrument='piano', ensemble=False, composer='Mozart', ground_truth=['precise_alignment'])

   audio_array, sources_array, ground_truth_array = d.get_item(1)

   audio_array = d.get_mix(2)
   source_array = d.get_source(2)
   ground_truth_list = d.get_gts(2)

   mat = d.get_score(2, score_type=['precise_alignment'])

Note that you can inherit from ``asmd.Dataset`` and
``torch.utils.data.Dataset`` to create a PyTorch compatible dataset which only
load audio files when thay are accessed. You will just need to implement the
``__getitem__`` method.


Documentation
-------------

.. automodule:: asmd.asmd
   :members:
   :private-members:
   :special-members:

.. autoclass:: asmd.asmd.Dataset(paths=['default_path'], metadataset_path=['default_path'])
   :members:
   :private-members:
   :special-members:

.. automodule:: asmd.dataset_utils
   :members:
   :private-members:
   :special-members:
