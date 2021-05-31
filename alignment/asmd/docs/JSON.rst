Usage
=====

datasets.json
-------------

The root element is a dictionary with fields:

#. ``author``: string containing the name of the author
#. ``year``: int containing the year
#. ``install_dir``: string containing the install directory
#. ``datasets``: list of datasets object
#. ``decompress_path``: the path were files are decompressed

Definitions
-----------

Each dataset is described by a JSON file which. Each dataset has the
following field:

#. ``ensemble``: ``true`` if contains multiple instruments, ``false`` otherwise
#. ``groups``: list of strings representing the groups contained in this
   dataset; the default name ``all`` must be always present
#. ``instruments``: the list of the instruments contained in the dataset
#. ``sources``:

   #. ``format``: the format of the audio recordings of the single source-separated tracks

#. ``recording``:

   #. ``format``: the format of the audio recordings of the mixed tracks

#. ``ground_truth``: *N.B. each ground_truth has an ``int`` value, indicating ``0`` -> false, ``1`` -> true (manual or mechanical - Disklavier - annotation), ``2`` -> true (automatic annotation with state-of-art algorithms)*

   #. ``[group-name]`` : a dictionary representing the ground-truth contained by each dataset group

       #. ``misaligned``: if artificially misaligned scores are provided
       #. ``score``: if original scores are provided
       #. ``broad_alignment``: if broad_alignment scores are provided
       #. ``precise_alignment``: if precisely aligned scores are provided
       #. ``velocities``: if velocities are provided
       #. ``f0``: if f0 values are provided
       #. ``sustain``: if sustain values are provided
       #. ``soft``: if sustain values are provided
       #. ``sostenuto``: if sustain values are provided

#. ``songs``: the list of songs in the dataset

   #. ``composer``: the composer family name
   #. ``instruments``: list of instruments in the song
   #. ``recording``: dictionary
   
      #. ``path``: a list of paths to be mixed for reconstructing the full track (usually only one)
      
   #. ``sources``: dictionary
   
      #. ``path``: a list of paths to the single instrument tracks in the same order as ``instruments``
      
   #. ``ground_truth``: list of paths to the ground_truth json files.  One
      ground_truth path per instrument is always provided. The order of the
      ground_truth path is the same of sources and of the instruments. Note
      that some ground_truth paths can be identical (as in PHENICX for
      indicating that violin1 and violin2 are playing exactly the same
      thing).
   #. ``groups``: list of strings representing a group of the dataset. The
         group ``all`` must always be there; any other string is possible and
         should be exposed in the ``groups`` field at dataset-level
   
#. ``install``: where information for the installation process are stored

   #. ``url``: the url to download the dataset including the protocol
   #. ``post-process``: a list of shell commands to be executed to prepare the
      dataset; they can be lists themselves to allow the use of references
      to the installation directory with the syntax ``&install_dir``: every
      occurrence of ``&install_dir`` will be replaced with the value of
      ``install_dir`` in ``datasets.json``; final slash doesn't matter
   #. ``unpack``: ``true`` if the url needs to be unpacked (untar, unzip, ...)
   #. ``login``: true if you a login is needed - not used anymore, but maybe useful in future

In general, I maintained the following principles:

#. if a list of files is provided where you would logically expect one file,
   you should ‘sum’ the files in the list, whatever this means according to
   that type of file; this typically happens in the ``ground_truth`` files. or
   in the recording where only the single sources are available.
#. all the fields can have the value ‘unknown’ to indicate that it is not
   available in that dataset; if you treat ‘unknown’ with the meaning of
   unavailable everything will be fine; however, in some cases it can mean that
   the data are available but that information is not documented.

Ground-truth json format
------------------------

The ground_truth is contained in JSON files indexed in each definition
file. Each ground truth file contains only one isntrument in a
dictionary with the following structure:

#. ``score``:

   #. ``onsets``: onsets in seconds at 20 bpm
   #. ``offsets``: offsets in seconds at 20 bpm
   #. ``pitches``: list of midi pitches in onset ascending order and range [0-127]
   #. ``notes`: list of note names in onsets ascending order
   #. ``velocities``: list of velocities in onsets ascending order and range [0-127]
   #. ``beats``: list of times in which there was a beat in the original score;
         use this to reconstruct instant BPM

#. ``misaligned``:

   #. ``onsets``: onsets in seconds
   #. ``offsets``: offsets in seconds
   #. ``pitches``: list of midi pitches in onset ascending order and range [0-127]
   #. ``notes`: list of note names in onsets ascending order
   #. ``velocities``: list of velocities in onsets ascending order and range [0-127]

#. ``precise_alignment``:

   #. ``onsets``: onsets in seconds
   #. ``offsets``: offsets in seconds
   #. ``pitches``: list of midi pitches in onset ascending order and range [0-127]
   #. ``notes`: list of note names in onsets ascending order
   #. ``velocities``: list of velocities in onsets ascending order and range [0-127]

#. ``broad_alignment``: alignment which does not consider the asynchronies between simultaneous notes

   #. ``onsets``: onsets in seconds
   #. ``offsets``: offsets in seconds
   #. ``pitches``: list of midi pitches in onset ascending order and range [0-127]
   #. ``notes`: list of note names in onsets ascending order
   #. ``velocities``: list of velocities in onsets ascending order and range [0-127]

#. ``missing``: list of boolean values indicating which notes are missing in
     the score (i.e. notes that you can consider as being played but not in
     the score); use this value to mask the performance/score
#. ``extra``: list of boolean values indicating which notes are extra in
     the score (i.e. notes that you can consider as not being played but in
     the score); use this value to mask the performance/score

#. ``f0``: list of f0 frequencies, frame by frame; duration of each frame
   should be 46 ms with 10 ms of hop.

#. ``sustain``:

   #. ``values``: list of sustain changes; each susvalue is a number
      between 0 and 127, where values < 63 mean sustain OFF and values >= 63
      mean sustain ON, but intermediate values can be used (e.g. for
      half-pedaling).
   #. ``times``: list of floats representing the time of each sustain change in
      seconds.

#. ``soft``:

   #. ``values``: list of soft-pedal changes; each value is a number between 0
      and 127, where values < 63 mean soft pedal OFF and values >= 63 mean
      soft pedal ON, but intermediate values can be used (e.g. for
      half-pedaling).
   #. ``times``: list of floats representing the time of each soft pedal change
      in seconds.

#. ``sostenuto``:

   #. ``values``: list of sostenuto-pedal changes; each value is a number between 0
      and 127, where values < 63 mean sostenuto pedal OFF and values >= 63 mean
      sostenuto pedal ON, but intermediate values can be used (e.g. for
      half-pedaling).
   #. ``times``: list of floats representing the time of each sostenuto pedal change
      in seconds.

#. ``instrument``: General Midi program number associated with this instrument,
   starting from 0. 128 indicates a drum kit (should be synthesized on channel
   8 with a program number of your choice, usually 0). 255 indicates no
   instrument specified.

Note that json ground_truth files have extension ``.json.gz``,
indicating that they are compressed using the ``gzip`` Python
module. Thus, you need to decompress them:

.. code:: python

    import gzip
    import json

    ground_truth = json.load(gzip.open(‘ground_truth.json.gz’, ‘rt’))

    print(ground_truth)

