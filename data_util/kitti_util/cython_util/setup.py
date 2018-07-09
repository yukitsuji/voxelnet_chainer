#python setup.py build_ext --inplace
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os
from os.path import join as pjoin
import numpy as np
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

def find_in_path(name, path):
    "Find a file in a search path"
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()

def customize_compiler_for_nvcc(self):
    """inject deep into distutils to customize how the dispatch
    to g++/nvcc works.
    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on."""

    # save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        print(extra_postargs)
        postargs = extra_postargs['g++']

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # inject our redefined _compile method into the class
    self._compile = _compile


# run the customize_compiler
class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)

ext_modules = [
    Extension(
        "create_input",
        ["create_feature_input.pyx"],
        extra_compile_args={'g++': ["-std=c++11", "-Wno-cpp", "-Wno-unused-function", "-O3"]},
        include_dirs = [numpy_include],
        language="c++",
    ),
    Extension(
        "nms_3d",
        ["nms.pyx"],
        extra_compile_args={'g++': ["-std=c++11", "-Wno-cpp", "-Wno-unused-function", "-O3"]},
        include_dirs = [numpy_include],
        language="c++",
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={'build_ext': custom_build_ext},
)
