from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "src.cython_kernels",
        ["src/cython_kernels.pyx"],
        include_dirs=[np.get_include()],
    ),
]

setup(
    name="metrik-ai",
    ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}),
)
