#! /bin/bash

#
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

# cd AtacWorks/pytorch; scripts/run_training.sh log-file-path

log='log.txt'
if [ "$1" != "" ]; then
    log="$1"
fi
echo "Dumping stdout to $log..."

script_dir=$(readlink -f $(dirname "$0"))

# # training exampleg
# python3 -u main.py \
#     --train \
#     --train_files data/example2/train \
#     --val_files data/example2/val \
#     --checkpoint_fname checkpoint.pth.tar \
#     --label HSC_unet \
#     --ratio 0.02 \
#     --model unet \
#     --clip_grad 1.0\
#     --epochs 20 \
#     --mse_weight 0.001 \
#     --distributed \
#     --task both \
#     --bs 2 \
#     --eval_freq 1 \
#     | tee $log

# # inference example
# python3 -u main.py \
#     --infer \
#     --distributed \
#     --infer_files data/example2/val \
#     --weights_path Cache/HSC_unet_2019.07.29_21.57/model_best.pth.tar \
#     --label infer-test \
#     --model unet \
#     --task both \
#     --bs 1 \
#     | tee $log


# eval example
# python3 -u $script_dir/main.py \
#     --eval \
#     --distributed \
#     --val_files data/example2/val \
#     --weights_path Cache/HSC_unet_2019.07.29_21.57/model_best.pth.tar \
#     --label eval-test \
#     --model unet \
#     --task both \
#     --bs 1 \
#     | tee $log


# multiprocessed postprocess
python3 $script_dir/postprocess_p.py \
        "/workspace/data/atac_postprocess_data/24000.genome_intervals.bed" \
        "/workspace/data/atac_postprocess_data/infer.GMP-5.epoch11_1000000.7cell.resnet.5.2.15.8.50.0803.pth.tar.h5" \
        "AtacWorks/example/reference/hg19.auto.sizes" \
        new_code \
        --channel 0 \
        --num_worker -1 \
        --tmp_dir tmp-test-satv
