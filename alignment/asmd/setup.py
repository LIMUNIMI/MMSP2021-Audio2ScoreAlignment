from distutils.core import setup
from build import build

global setup_kwargs

setup_kwargs = {}

build(setup_kwargs)
setup(**setup_kwargs)
