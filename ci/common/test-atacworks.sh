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
# AtacWorks CPU/GPU test script for CI #
################################################################################

# Run tests.
if [ "${TEST_ON_GPU}" == '1' ]; then
    ./tests/end-to-end/run.sh
    python -m pytest tests/
else
    logger "No CPU tests."
fi
