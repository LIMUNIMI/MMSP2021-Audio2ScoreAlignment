Adding new datasets
===================

In order to add new datasets, you have to create the correspondent
definition in a JSON file. The definitions can be in any directory but
you have to provide this path to the API and to the installation script
(you will be asked for this, so you can’t be wrong).

The dataset files, instead, should be in the installation directory and
the paths in the definition should not take into account the
installation directory.

If you also want to add the new dataset to the installation procedure,
you should:

#. Provide a conversion function for the ground truth
#. Add the conversion function with all parameters to the JSON definition (section ``install>conversion``)
#. Rerun the ``install.py`` and ``convert_gt.py`` scripts

Adding new definitions
----------------------

The most important thing is that one ground-truth file is provided for
each instrument.

If you want to add datasets to the installation procedure, taking advantage of
the artificially misalignment, add the paths to the files (ground-truth, audio,
etc.), even if they still do not exist, because ``convert_gt.py`` relies on
those paths to create the files. It is important to provide an index starting
with ``-`` at the end of the path (see the other sections as example), so that
it is possible to distinguish among multiple instruments (for instance, PHENICX
provides one ground-truth file for all the violins of a song, even if there are
4 different violins). The index allows ``convert_gt`` to better handle
different files and to pick the ground-truth wanted.

It is mandatory to provide a url, a name and so on. Also, provide a
composer and instrument list. Please, do not use new words for
instruments already existing (for instance, do not use ``saxophone`` if
``sax`` already exists in other datasets).

Provide a conversion function
-----------------------------
Docs available at :doc:`./Converting`

The conversion function takes as input the name of the file in the
original dataset. You can also use the bundled conversion functions (see
docs).

#. use ``deepcopy(gt)`` to create the output ground truth.
#. use decorator ``@convert`` to provide the input file extensions and parameters

You should consider three possible cases for creating the conversion
function:

#. there is a bijective relationship between instruments and ground_truth file
   you have, that is, you already have a convesion file per each instrument and
   you should just convert all of them (*1-to-1 relationship*)
#. in your dataset, all the instruments are inside just one ground-truth   file
   (*n-to-1 relationship*)
#. just one ground-truth file is provided that replicates for multiple
   instruments (one ground-truth for all the ``violins``, as if they   were a
   single instrument, *1-to-n relationship* )

Here is a brief description of how your conversion function should work
to tackle these three different situations. - In the 1st case, you can
just output a list with only one dictionary. - In the 2nd case, you can
output a list with all the dictionary inside it, in the same order as
the ground-truth file paths you added to ``datasets.json``. The script
will repeatly convert them and each times it will pick a different
element of the list. - In the 3rd case, you can still output a single
element list.

If you want to output a list with only one dict, you can also output the
dict itself. The decorator will take care of handling file names and of
putting the output dict inside a list.

Finally, you can also use multiple conversion functions if your
ground-truth is splitted among multiple files, but note that the final
ground-truth is produced as the sum of all the elements of all the
dictionaries created.

Add your function to the JSON definition
----------------------------------------

In the JSON definitions, you should declare the functions that should be
used for converting the ground-truth and their parameters. The section
where you can do this is in ``install>conversion``.

Here, you should put a list like the following:

.. code:: python

       [
           [
               "module1.function1", {
                   "argument1_name": argument1_value,
                   "argument2_name": argument2_value
               }
       ],
           [
               "module2.function2", {
                   "argument1_name": argument1_value,
                   "argument2_name": argument2_value
               }
           ]
       ]

Note that you have to provide the *name* of the function, which will be
evaluated with the ``eval`` python function. Also, you can use any
function in any module, included the bundled functions - in this case,
use just the function name w/o the module.

