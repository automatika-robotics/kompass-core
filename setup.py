import glob
import os

from pybind11.setup_helpers import (
    Pybind11Extension,
    build_ext,
    ParallelCompile,
    naive_recompile,
)
from setuptools import find_packages, setup

__version__ = "0.0.1"

ext_modules = [
    Pybind11Extension(
        "ompl",
        glob.glob(os.path.join("src/ompl/src", "*.cpp")),
        include_dirs=["/usr/include/ompl-1.5", "/usr/include/eigen3"],
        libraries=["ompl", "boost_python310"],
        library_dirs=["/usr/lib/x86_64-linux-gnu"],
        define_macros=[("VERSION_INFO", __version__)],
    ),
    Pybind11Extension(
        "kompass_cpp",
        glob.glob(
            os.path.join("src/kompass_cpp/kompass_cpp/src/**", "*.cpp"), recursive=True
        ),
        include_dirs=[
            "src/kompass_cpp/kompass_cpp/include",
            "/usr/include/eigen3",
            "/usr/include/pcl-1.12",
        ],
        libraries=["boost_python310", "fcl", "pcl_io_ply", "pcl_common"],
        library_dirs=["/usr/lib/x86_64-linux-gnu"],
        define_macros=[("VERSION_INFO", __version__)],
    ),
]

# add parallel compilation if flag is set
ParallelCompile("CMAKE_BUILD_PARALLEL_LEVEL", needs_recompile=naive_recompile).install()

setup(
    packages=find_packages(where="src"),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
