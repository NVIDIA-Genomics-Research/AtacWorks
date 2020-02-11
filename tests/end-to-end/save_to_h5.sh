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
echo "Read clean and noisy data in these intervals and save them in .h5 format..."
echo ""
# Training data
python $root_dir/bw2h5.py \
    --noisybw $data_dir/HSC.5M.chr123.10mb.coverage.bw \
    --intervals $out_dir/result.training_intervals.bed \
    --out_dir $out_dir \
    --prefix train_data \
    --cleanbw $data_dir/HSC.80M.chr123.10mb.coverage.bw \
    --cleanpeakbw $out_dir/clean.peaks.bw \
    --nonzero
# Validation data
python $root_dir/bw2h5.py \
    --noisybw $data_dir/HSC.5M.chr123.10mb.coverage.bw \
    --intervals $out_dir/result.val_intervals.bed \
    --out_dir $out_dir \
    --prefix val_data \
    --cleanbw $data_dir/HSC.80M.chr123.10mb.coverage.bw \
    --cleanpeakbw $out_dir/clean.peaks.bw
# No label
python $root_dir/bw2h5.py \
    --noisybw $data_dir/HSC.5M.chr123.10mb.coverage.bw \
    --intervals $out_dir/result.holdout_intervals.bed \
    --out_dir $out_dir \
    --prefix no_label \
    --nolabel

echo ""
echo "Verifying output against expected results"
python $utils_dir/verify_diff.py --result_path $out_dir/train_data.h5 \
	              --expected_path $expected_results_dir/train_data.h5 \
		      --format "h5"
check_status $?

python $utils_dir/verify_diff.py --result_path $out_dir/val_data.h5 \
	              --expected_path $expected_results_dir/val_data.h5 \
		      --format "h5"
check_status $?

python $utils_dir/verify_diff.py --result_path $out_dir/no_label.h5 \
	              --expected_path $expected_results_dir/no_label.h5 \
		      --format "h5"
check_status $?
