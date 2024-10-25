import glob
import os

from pybind11.setup_helpers import (
    Pybind11Extension,
    build_ext,
    ParallelCompile,
    naive_recompile,
)
from setuptools import find_packages, setup
import subprocess

__version__ = "0.1.1"


def check_pkg_config():
    """Check if pkg-config is installed"""
    try:
        subprocess.check_output(["pkg-config", "--version"])
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            "pkg-config is not installed, but it is required for building this package."
        ) from e


def find_pkg(package, versions=None):
    """Find a given package using pkg-config"""
    if not versions:
        try:
            subprocess.check_output(["pkg-config", "--exists", package])
            print(f"FOUND {package}")
        except subprocess.CalledProcessError as e:
            raise ModuleNotFoundError(
                f"'{package}' is required but is not found"
            ) from e
        return package
    else:
        found_version = None
        for version in versions:
            try:
                subprocess.check_output([
                    "pkg-config",
                    "--exists",
                    f"{package}-{version}",
                ])
            except subprocess.CalledProcessError:
                continue
            print(f"FOUND {package} version {version}")
            found_version = version
            break
        if found_version:
            return f"{package}-{found_version}"
        else:
            raise ModuleNotFoundError(f"'{package}' is required but is not found")


def pkg_config(*packages, flag, versions=None):
    """Run pkg-config with given flag"""
    found_packages = []
    # Check if the package exists
    for pkg in list(packages):
        found_packages.append(find_pkg(pkg, versions))
    try:
        output = (
            subprocess.check_output(["pkg-config", flag] + found_packages)
            .decode("utf-8")
            .strip()
        )
        output_dirs = []
        for output_dir in output.split():
            output_dirs.append(output_dir.split("-I")[-1])
        return output_dirs
    except subprocess.CalledProcessError:
        return []


# Check that pkg-config is installed
check_pkg_config()

# OMPL dependencies
ompl_include_dirs = pkg_config(*["ompl", "eigen3"], flag="--cflags-only-I")
ompl_library_dirs = pkg_config(*["ompl"], flag="--libs-only-other")

# Accepted PCL versions
pcl_versions = ["1.14", "1.13", "1.12", "1.11", "1.10", "1.9"]

# Kompass CPP dependencies
eigen_include_dir = pkg_config(*["eigen3"], flag="--cflags-only-I")
pcl_include_dir = pkg_config(
    *["pcl_common"], flag="--cflags-only-I", versions=pcl_versions
)


ext_modules = [
    Pybind11Extension(
        "ompl",
        glob.glob(os.path.join("src/ompl/src", "*.cpp")),
        include_dirs=ompl_include_dirs,
        libraries=["ompl"],
        library_dirs=ompl_library_dirs,
        define_macros=[("VERSION_INFO", __version__)],
    ),
    Pybind11Extension(
        "kompass_cpp",
        glob.glob(
            os.path.join("src/kompass_cpp/kompass_cpp/src/**", "*.cpp"), recursive=True
        ),
        include_dirs=["src/kompass_cpp/kompass_cpp/include"]
        + eigen_include_dir
        + pcl_include_dir,
        libraries=["fcl", "pcl_io_ply", "pcl_common"],
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
