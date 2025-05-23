import os
import sys
import subprocess
from setuptools import setup, Extension
import pybind11
import numpy as np

# This setup.py builds the QP_SWIFT C sources inline, relative to this python/ folder
here = os.path.abspath(os.path.dirname(__file__))
project_root = os.path.abspath(os.path.join(here, os.pardir))

# Paths for includes and sources
include_dir = os.path.join(project_root, 'include')
src_dir = os.path.join(project_root, 'src')

# Verify directories
if not os.path.isdir(include_dir):
    sys.exit(f"ERROR: include directory not found: {include_dir}")
if not os.path.isdir(src_dir):
    sys.exit(f"ERROR: src directory not found: {src_dir}")


# Function to find Eigen include directory
def find_eigen():
    # Common paths where Eigen might be installed
    common_paths = [
        "/usr/include/eigen3",
        "/usr/local/include/eigen3",
        "/opt/homebrew/include/eigen3",
        "/usr/include",
        "/usr/local/include",
        "/opt/local/include/eigen3",
        "/opt/local/include",
    ]

    # Try to find Eigen/Core header in common paths
    for path in common_paths:
        if os.path.exists(os.path.join(path, "Eigen", "Core")):
            return path

    # Try using pkg-config if available
    try:
        eigen_cflags = subprocess.check_output(["pkg-config", "--cflags", "eigen3"],
                                               universal_newlines=True).strip()
        for item in eigen_cflags.split():
            if item.startswith("-I"):
                path = item[2:]
                if os.path.exists(os.path.join(path, "Eigen", "Core")):
                    return path
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    # Try using find command if available (Unix-like systems)
    try:
        find_output = subprocess.check_output(
            ["find", "/usr", "-name", "Eigen", "-type", "d", "-path", "*/eigen3/Eigen"],
            universal_newlines=True
        ).strip()

        if find_output:
            # Take the first result and get its parent directory
            eigen_dir = os.path.dirname(find_output.split('\n')[0])
            if os.path.exists(os.path.join(eigen_dir, "Eigen", "Core")):
                return eigen_dir
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    return None


# Find Eigen include directory
eigen_include_dir = find_eigen()

if eigen_include_dir is None:
    print("WARNING: Eigen headers not found automatically. Building might fail.")
    print("You may need to install Eigen with: apt install libeigen3-dev (Debian/Ubuntu),")
    print("brew install eigen (macOS), or download from http://eigen.tuxfamily.org/")
else:
    print(f"Found Eigen headers at: {eigen_include_dir}")

# Collect all C source files
c_sources = [
    os.path.join(src_dir, f)
    for f in [
        'amd_1.c', 'amd_2.c', 'amd_aat.c', 'amd_control.c', 'amd_defaults.c',
        'amd_dump.c', 'amd_global.c', 'amd_info.c', 'amd_order.c',
        'amd_post_tree.c', 'amd_postorder.c', 'amd_preprocess.c', 'amd_valid.c',
        'Auxilary.c', 'ldl.c', 'timer.c', 'qpSWIFT.c'
    ]
]

# Add the pybind11 wrapper
wrapper_cpp = os.path.join(here, 'pyqp_swift.cpp')
c_sources.append(wrapper_cpp)

# Setup include directories
include_dirs = [
    include_dir,
    pybind11.get_include(),
    pybind11.get_include(user=True),
    np.get_include(),
]

# Add Eigen include directory if found
if eigen_include_dir:
    include_dirs.append(eigen_include_dir)

# Prepare extra compile args
extra_compile_args = ["-O3", "-std=c++11"]

ext_modules = [
    Extension(
        name="qpSWIFT",
        sources=c_sources,
        include_dirs=include_dirs,
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
]

setup(
    name="qpSWIFT",
    version="0.1.0",
    description="Python bindings for the QP_SWIFT solver (inline C build)",
    ext_modules=ext_modules,
    install_requires=["numpy", "scipy", "pybind11"],
    zip_safe=False,
)