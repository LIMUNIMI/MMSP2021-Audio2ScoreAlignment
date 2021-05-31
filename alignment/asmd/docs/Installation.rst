Installation
============

I suggest to clone this repo and to use it with python >= 3.6. If you
need to use it in multiple projects (or folders), just clone the code and
fix the ``install_dir`` in ``datasets.json``, so that you can have only
one copy of the huge datasets.

The following describes how to install dependecies needed for the usage of the
dataset API. I suggest to use  `poetry <https://python-poetry.org/>`__ to manage
different versions of python and virtual environments with an efficient
dependency resolver.

During the installation, the provided ground-truth will be extracted; however,
you can recreate them from scratch for tweaking parameters. The next section
will explain how you can achieve this.

The easy way
------------
#. ``pip install asmd`` 
#. Install ``wget`` if you want SMD dataset (next release will remove this dependency)
#. Run ``python -m asmd.install`` and follows the steps


The hard way (if you want to contribute)
----------------------------------------

Once you have cloned the repo follow these steps:

Install poetry, pyenv and python
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#. Install ``wget`` if you want SMD dataset (next release will remove this dependency)
#. Install ``python 3``
#. Install `poetry <https://python-poetry.org/docs/#installation>`__
#. Install `pyenv <https://github.com/pyenv/pyenv#installation>`__ and fix your
   ``.bashrc`` (optional)
#. ``pyenv install 3.6.9`` (optional, recommended python >= 3.6.9)
#. ``poetry new myproject``
#. ``cd myproject``
#. ``pyenv local 3.6.9`` (optional, recommended python >= 3.6.9)

Setup new project or testing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#. ``git clone https://gitlab.di.unimi.it/federicosimonetta/asmd/``
#. ``poetry add asmd/``
#. Execute ``poetry run python -m asmd.install``; alternatively run ``poetry
   shell`` and then ``python -m asmd.install``
#. Follow the steps

Now you can start developing in the parent directory (``myproject``) and
you can use ``from asmd import audioscoredataset as asd``.

Use ``poetry`` to manage packages of your project.

Alternative way
^^^^^^^^^^^^^^^

#. clone the project.
#. to build the modules in place, run ``poetry run python setup.py build_ext --inplace``
#. create an ad-hoc directory for testing anywhere:
   
   #. copy there the original ``pyproject.toml``
   #. install the needed dependencies with ``poetry update``
   #. export the ``PYTHONPATH`` environment variable: e.g.
      ``export PYTHONPATH="path/to/asmd"``.

You're now ready to use ASMD without downloading it from PyPI.

To create the python package, just run ``poetry build``.
