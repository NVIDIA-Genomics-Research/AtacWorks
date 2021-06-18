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

RUN apt-get update && apt-get install -y \
        software-properties-common
    RUN add-apt-repository ppa:deadsnakes/ppa
    RUN apt-get update && apt-get install -y \
        python3.7


# Set python version to 3.6
RUN apt update && apt install -y \
    rsync \
    git \
    python3-pip \
    libz-dev \
    vim \
    wget \
    curl \
    python3.7-dev

RUN ln -nsf /usr/bin/python3.7 /usr/bin/python

# Use python3.7 pip. Otherwise some packages won't be compatible.
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1

# Download 3rd party binaries
RUN rsync -aP rsync://hgdownload.soe.ucsc.edu/genome/admin/exe/linux.x86_64/bedGraphToBigWig /usr/local/bin
RUN rsync -aP rsync://hgdownload.soe.ucsc.edu/genome/admin/exe/linux.x86_64/bigWigToBedGraph /usr/local/bin

# Download AtacWorks repo
RUN pip3 install atacworks==0.3.4
RUN pip3 install macs2==2.2.4
