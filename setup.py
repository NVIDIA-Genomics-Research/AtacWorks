#!/usr/bin/env python3

#
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
"""Python setuptools setup."""

import os

from setuptools import find_namespace_packages, setup


def get_verified_absolute_path(path):
    """Verify and return absolute path of argument.

    Args:
        path : Relative/absolute path

    Returns:
        Absolute path
    """
    installed_path = os.path.abspath(path)
    if not os.path.exists(installed_path):
        raise RuntimeError("No valid path for requested component exists")
    return installed_path


def get_installation_requirments(file_path):
    """Parse pip requirements file.

    Args:
        file_path : path to pip requirements file

    Returns:
        list of requirement strings
    """
    with open(file_path, 'r') as file:
        requirements_file_content = \
            [line.strip() for line in file if
             line.strip() and not line.lstrip().startswith('#')]
    return requirements_file_content


# Get current dir (pyclaragenomics folder is copied into a temp directory
# created by pip)
current_dir = os.path.dirname(os.path.realpath(__file__))

# Classifiers for PyPI
pyaw_classifiers = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Bio-Informatics",
    "Natural Language :: English",
    "Operating System :: POSIX :: Linux",
    "Programming Language :: Python :: 3.5",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9"
]

required_packages = \
    get_installation_requirments(
        get_verified_absolute_path(
            os.path.join(current_dir, 'requirements.txt'))
    )


setup(name='atacworks',
      version='0.3.1',
      description='NVIDIA genomics python libraries and utiliites',
      author='NVIDIA Corporation',
      url="https://github.com/clara-genomics/AtacWorks",
      include_package_data=True,
      install_requires=required_packages,
      packages=find_namespace_packages(),
      python_requires='>=3.5',
      long_description='Python libraries and utilities for manipulating '
                       'genomics data',
      classifiers=pyaw_classifiers,
      entry_points={'console_scripts': ['atacworks = scripts.main:main']},
      data_files=[
          ('configs', ['configs/infer_config.yaml',
                       'configs/train_config.yaml',
                       'configs/model_structure.yaml']),
          ('reference', ['data/reference/hg19.chrom.sizes',
                         'data/reference/hg19.auto.sizes',
                         'data/reference/hg38.chrom.sizes',
                         'data/reference/hg38.auto.sizes'])],
      platforms=['any'],
      )
