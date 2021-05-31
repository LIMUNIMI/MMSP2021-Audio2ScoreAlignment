from distutils.extension import Extension
from Cython.Build import cythonize
# import numpy

extensions = [
    # Extension("asmd.asmd", ["asmd/asmd.py"],
    #           include_dirs=[numpy.get_include()]),
    Extension("asmd.convert_from_file", ["asmd/convert_from_file.py"]),
    Extension("asmd.conversion_tool", ["asmd/conversion_tool.py"]),
    Extension("asmd.utils", ["asmd/utils.py"])
]


def build(setup_kwargs):
    setup_kwargs.update({
        'ext_modules':
        cythonize(extensions,
                  compiler_directives={
                      'language_level': "3",
                      'embedsignature': True
                  })
    })
