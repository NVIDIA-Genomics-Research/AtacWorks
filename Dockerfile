#
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

# Inherit from base NVIDIA CUDA 10.1 image with CUDNN
FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

# Set python version to 3.6
RUN apt update && apt install -y \
    python3.6 \
    rsync \
    git \
    python3-pip \
    libz-dev \
    hdf5-tools

RUN ln -nsf /usr/bin/python3.6 /usr/bin/python

# Download 3rd party binaries
RUN rsync -aP rsync://hgdownload.soe.ucsc.edu/genome/admin/exe/linux.x86_64/bedGraphToBigWig /usr/local/bin
RUN rsync -aP rsync://hgdownload.soe.ucsc.edu/genome/admin/exe/linux.x86_64/bigWigToBedGraph /usr/local/bin

# Download AtacWorks repo
RUN git clone --recursive https://github.com/clara-genomics/AtacWorks.git

# Install AtacWorks requirements
RUN pip3 install -r AtacWorks/requirements-base.txt && pip3 install -r AtacWorks/requirements-macs2.txt
RUN pip install .
