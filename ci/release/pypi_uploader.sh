#!/bin/bash
#
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

set -e

# Skip upload if CI is executed locally
if [[ ${RUNNING_CI_LOCALLY} = true  ]]; then
    echo "Skipping PyPi upload - running ci locally"
    return 0
fi

# Skip upload if the merging branch is not master
if [ "${COMMIT_HASH}" != "master" ]; then
    echo "Skipping PyPI upload - merge branch is not master"
    return 0
fi

for f in "${WORKSPACE}"/atacworks/atacworks_wheel/*.whl; do
    if [ ! -e "${f}" ]; then
        echo "atacworks Whl file does not exist"
        exit 1
    else
        conda install -c conda-forge twine
        # Change .whl package name to support PyPI upload
        MODIFIED_WHL_NAME=$(dirname ${f})/$(basename "${f}" | sed -r "s/(.*-.+-.+)-.+-.+.whl/\1-none-any.whl/")
        mv "${f}" "${MODIFIED_WHL_NAME}"
        echo "File name ${f} was changed into ${MODIFIED_WHL_NAME}"
        # Perform Upload
        python3 -m twine upload --skip-existing "${WORKSPACE}"/atacworks_wheel/*
    fi
done
