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
source $utils_dir/utils.sh
echo ""
echo "Train "
echo ""
python $root_dir/main.py train\
    --out_home $out_dir \
    --label model \
    --distributed \
    --files_train $out_dir/train_data.h5 \
    --val_files $out_dir/val_data.h5 \
    --checkpoint_fname checkpoint.pth.tar \
    --epochs 1 --bs 4 \
    --width 50 --width_cla 50 --dil_cla 10 --pad 0
# Training is not deterministic, so we are not comparing results.
check_status $? "Training run not succesful!"

echo ""
echo "Test classification mode of training"
echo ""
python $root_dir/main.py train \
    --files_train $out_dir/train_data.h5 \
    --val_files $out_dir/val_data.h5 \
    --model logistic --field 8401 \
    --out_home $out_dir --label logistic \
    --task classification --bs 4 \
    --epochs 1 --pad 5000
# Training is not deterministic, so we are not comparing results.
check_status $? "Training run not succesful!"
