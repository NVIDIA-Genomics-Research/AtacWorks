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

################################################################################
# AtacWorks CPU/GPU package installation script for CI #
################################################################################

#Install external dependencies.
python -m pip install --ignore-installed -r requirements-base.txt && python -m pip install -r requirements-macs2.txt
pip install .

LOCAL_BIN_DIR="${WORKSPACE}/local_bin"
mkdir -p "${LOCAL_BIN_DIR}"
export PATH="$PATH:$LOCAL_BIN_DIR"

# Install custom binaries.Try the US server first, if it fails, then try the europe server.
rsync -aP rsync://hgdownload.soe.ucsc.edu/genome/admin/exe/linux.x86_64/bedGraphToBigWig "$LOCAL_BIN_DIR"/ ||
rsync -aP rsync://hgdownload-euro.soe.ucsc.edu/genome/admin/exe/linux.x86_64/bedGraphToBigWig "$LOCAL_BIN_DIR"/

rsync -aP rsync://hgdownload.soe.ucsc.edu/genome/admin/exe/linux.x86_64/bigWigToBedGraph "$LOCAL_BIN_DIR"/ ||
rsync -aP rsync://hgdownload-euro.soe.ucsc.edu/genome/admin/exe/linux.x86_64/bigWigToBedGraph "$LOCAL_BIN_DIR"/

