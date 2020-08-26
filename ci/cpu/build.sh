#!/bin/bash
#
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

################################################################################
# AtacWorks CPU build script for CI #
################################################################################
set -e

# Stat time for logger
START_TIME=$(date +%s)

export PATH=/conda/bin:/usr/local/cuda/bin:$PATH
# Set home to the job's workspace
export HOME=$WORKSPACE
cd "${WORKSPACE}"

################################################################################
# Init
################################################################################

source ci/common/logger.sh

logger "Calling prep-init-env..."
source ci/common/prep-init-env.sh

################################################################################
# Install AtacWorks
################################################################################
logger "Insalling AtacWorks..."
cd "${WORKSPACE}"
source ci/common/install-package.sh

################################################################################
# AtacWorks tests
################################################################################
logger "Running AtacWorks tests..."
cd "${WORKSPACE}"
source ci/common/test-atacworks.sh "$WORKSPACE"

logger "Remove existing atacworks"
pip uninstall -y atacworks

################################################################################
# Create Wheel Package for AtacWorks
################################################################################
logger "Create Wheel package for AtacWorks"
python3 -m pip wheel . --global-option sdist --wheel-dir ${WORKSPACE}/atacworks_wheel --no-deps

################################################################################
# Install AtacWorks from Wheel Package
################################################################################
logger "Insalling AtacWorks from wheel..."
cd "${WORKSPACE}"
pip install --ignore-installed ${WORKSPACE}/atacworks_wheel/*

################################################################################
# AtacWorks tests
################################################################################
logger "Running AtacWorks tests..."
cd "${WORKSPACE}"
source ci/common/test-atacworks.sh "$WORKSPACE"

################################################################################
# Upload AtacWorks to PyPI
################################################################################
logger "Upload Wheel to PyPI..."
cd "${WORKSPACE}"
source ci/release/pypi_uploader.sh

logger "Done..."
