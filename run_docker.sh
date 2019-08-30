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


sudo docker run --runtime=nvidia --shm-size=480G -it \
    -v=/mnt/dcg04-ericx/code/AtacWorks1:/workspace/AtacWorks \
    nvcr.io/nvidian_general/keras-pytorch-atacworks:latest