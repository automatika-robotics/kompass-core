import glob
import os
import sys

from pybind11.setup_helpers import (
    Pybind11Extension,
    build_ext,
    ParallelCompile,
    naive_recompile,
)
from setuptools import find_packages, setup
import subprocess

__version__ = "0.4.1"

# DEFAULTS FOR UBUNTU 22.04
OMPL_INCLUDE_DEFAULT_DIR = "/usr/include/ompl-1.5"
EIGEN_INCLUDE_DEFAULT_DIR = "/usr/include/eigen3"
PCL_INCLUDE_DEFAULT_DIR = "/usr/include/pcl-1.14"


def get_libraries_dir():
    """Get libraries dir"""
    lib_dirs = ["/usr/lib", "/usr/local/lib", "/usr/lib/x86_64-linux-gnu"]

    if "LD_LIBRARY_PATH" in os.environ:
        lib_dirs += os.environ["LD_LIBRARY_PATH"].split(":")
    return lib_dirs


def check_pkg_config():
    """Check if pkg-config is installed"""
    try:
        subprocess.check_output(["pkg-config", "--version"])
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            "pkg-config is not installed, but it is required for building this package."
        ) from e


def check_acpp():
    """Check if AdaptiveCPP is available"""
    try:
        subprocess.check_output(["acpp", "--version"])
        print("AdaptiveCPP found. Building with acpp.")
    except Exception:
        print("AdaptiveCPP is not installed. Building with default compiler available.")
        return False
    # change compiler to acpp
    os.environ["CXX"] = "acpp"
    return True


def find_pkg(package, versions=None):
    """Find a given package using pkg-config"""
    if not versions:
        try:
            subprocess.check_output(["pkg-config", "--exists", package])
            print(f"FOUND {package}")
            return package
        except subprocess.CalledProcessError:
            return
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


def pkg_config(packages, flag, versions=None):
    """Run pkg-config with given flag"""
    found_packages = []
    # Check if the package exists
    for pkg in packages:
        if found_pkg := find_pkg(pkg, versions):
            found_packages.append(found_pkg)
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

# get vcpkg paths when building wheels
vcpkg_path = os.environ.get("VCPKG_ROOT")


def get_vcpkg_includes():
    """Get include dir for vcpkg"""
    if vcpkg_path:
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        headers_dir = next(
            (s for s in os.listdir("/opt/_internal") if python_version in s), None
        )
        if not headers_dir:
            return [
                f"{vcpkg_path}/installed/x64-linux/include",
                f"{vcpkg_path}/installed/arm64-linux/include",
            ]
        return [
            f"{vcpkg_path}/installed/x64-linux/include",
            f"{vcpkg_path}/installed/arm64-linux/include",
            f"/opt/_internal/{headers_dir}/include/python{python_version}",
        ]
    else:
        return []


def get_vcpkg_extra_libs():
    """Get extra libs explicitly for linking because FCL port for vcpkg
    builds as a static lib that does not link its dependencies
    """
    if vcpkg_path:
        return ["octomap", "octomath", "ccd"]
    else:
        return []


# OMPL dependencies
ompl_include_dirs = pkg_config(["ompl"], flag="--cflags-only-I")

# Accepted PCL versions
pcl_versions = ["1.14", "1.13", "1.12", "1.11", "1.10", "1.9"]

# Kompass CPP dependencies
eigen_include_dir = pkg_config(["eigen3"], flag="--cflags-only-I")
pcl_include_dir = pkg_config(
    ["pcl_common"], flag="--cflags-only-I", versions=pcl_versions
) or [PCL_INCLUDE_DEFAULT_DIR]

# vcpkg paths when running in CI
vcpkg_includes_dir = get_vcpkg_includes()
vcpkg_extra_libs = get_vcpkg_extra_libs()


## Check include dirs
# for OMPL
ompl_module_includes = ompl_include_dirs + vcpkg_includes_dir
# try a static path when pkg_config does not return anything and not in CI
if not ompl_module_includes:
    ompl_module_includes = [OMPL_INCLUDE_DEFAULT_DIR]

eigen_include_dir = eigen_include_dir + vcpkg_includes_dir
# try a static path when pkg_config does not return anything and not in CI
if not eigen_include_dir:
    eigen_include_dir = [EIGEN_INCLUDE_DEFAULT_DIR]

# for Kompass CPP
kompass_cpp_dependency_includes = (
    eigen_include_dir + pcl_include_dir + vcpkg_includes_dir
)
kompass_cpp_module_includes = [
    "src/kompass_cpp/kompass_cpp/include"
] + kompass_cpp_dependency_includes
# try a static path when pkg_config does not return anything and not in CI
kompass_cpp_module_includes = (
    [
        "src/kompass_cpp/kompass_cpp/include",
        EIGEN_INCLUDE_DEFAULT_DIR,
        PCL_INCLUDE_DEFAULT_DIR,
    ]
    if not kompass_cpp_dependency_includes
    else ["src/kompass_cpp/kompass_cpp/include"] + kompass_cpp_dependency_includes
)
kompass_cpp_source_files = glob.glob(
    os.path.join("src/kompass_cpp/kompass_cpp/src/**", "*.cpp"), recursive=True
)
if not check_acpp():
    kompass_cpp_source_files = [
        f_name for f_name in kompass_cpp_source_files if not f_name.endswith("gpu.cpp")
    ]
    extra_args = []
else:
    extra_args = ["-DGPU=1"]

ext_modules = [
    Pybind11Extension(
        "ompl",
        glob.glob(os.path.join("src/ompl/src", "*.cpp")),
        include_dirs=ompl_module_includes + eigen_include_dir,
        libraries=["ompl"],
        library_dirs=get_libraries_dir(),
        define_macros=[("VERSION_INFO", __version__)],
        extra_compile_args=["-O3"],
    ),
    Pybind11Extension(
        "kompass_cpp",
        kompass_cpp_source_files,
        include_dirs=kompass_cpp_module_includes,
        libraries=["fcl", "pcl_io_ply", "pcl_common"] + vcpkg_extra_libs,
        library_dirs=get_libraries_dir(),
        define_macros=[("VERSION_INFO", __version__)],
        extra_compile_args=["-O3"] + extra_args,
    ),
]

# add parallel compilation if flag is set
ParallelCompile("CMAKE_BUILD_PARALLEL_LEVEL", needs_recompile=naive_recompile).install()

setup(
    packages=find_packages(where="src"),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    version=__version__,
)
