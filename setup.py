#!/usr/bin/env python3
"""
Setup script for vector-store package.

This compiles the C++ extension using pybind11 and installs
both the native module and the Python wrapper package.
"""
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11


class BuildExt(build_ext):
    """Custom build extension for C++17 support."""

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = []
        link_opts = []

        if ct == 'unix':
            opts.extend([
                '-std=c++17',
                '-O3',
                '-Wall',
                '-fvisibility=hidden',
            ])
            opts.append('-DVERSION_INFO="{}"'.format(
                self.distribution.get_version()))

        for ext in self.extensions:
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts

        build_ext.build_extensions(self)


# Define the C++ extension module
# Module name MUST match PYBIND11_MODULE name in python_bindings.cpp
ext_modules = [
    Extension(
        'vector_cluster_store_py',
        sources=[
            'src/python_bindings.cpp',
            'src/vector_cluster_store.cpp',
            'src/kmeans_clustering.cpp',
        ],
        include_dirs=[
            pybind11.get_include(),
            pybind11.get_include(user=True),
            'src',
        ],
        language='c++',
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
)
