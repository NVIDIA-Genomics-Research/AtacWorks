# Inherit from base NVIDIA CUDA 10.1 image with CUDNN
FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

# Set python version to 3.6
RUN apt update && apt install -y \
    python3.6 \
    rsync

RUN ln -nsf /usr/bin/python3.6 /usr/bin/python

RUN rsync -aP rsync://hgdownload.soe.ucsc.edu/genome/admin/exe/linux.x86_64/bedGraphToBigWig /usr/local/bin
RUN rsync -aP rsync://hgdownload.soe.ucsc.edu/genome/admin/exe/linux.x86_64/bigWigToBedGraph /usr/local/bin

RUN apt install -y \
    git \
    python3-pip

# Download AtacWorks repo
RUN git clone --recursive https://github.com/clara-genomics/AtacWorks.git

RUN grep "^[^#]" AtacWorks/requirements-pip.txt | xargs pip3 install
