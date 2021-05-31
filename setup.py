import os
from setuptools import Extension, setup
from glob import glob

import julia
from Cython.Build import Cythonize, cythonize

from alignment import settings as s

if s.CLEAN:
    for path in glob('alignment/**.c*'):
        os.remove(path)

if s.BUILD:
    paths = set(glob("alignment/*.py"))
    paths -= set(["alignment/fede_tuning.py"])
    paths -= set(["alignment/settings.py"])
    paths -= set(glob("alignment/evaluate_*.py"))
    paths -= set(glob("alignment/__init__.py"))
    for path in paths:
        Cythonize.main([path, "-3", "--inplace"])

extensions = [
    Extension("alignment.cdist", ["alignment/cdist.pyx"],
              extra_compile_args=['-fopenmp'],
              extra_link_args=['-fopenmp'])
]

setup(name="cdist",
      ext_modules=cythonize(extensions,
                compiler_directives={
                    'language_level': "3",
                    'embedsignature': True,
                    'boundscheck': False,
                    'wraparound': False
                }))

julia.install()

# the following two after having installed PyCall and so on
from julia.api import LibJulia  # noqa: autoimport

api = LibJulia.load()
api.init_julia(["--project=."])

from julia import Main  # noqa: autoimport

Main.eval('using Pkg')
Main.eval('Pkg.instantiate()')
