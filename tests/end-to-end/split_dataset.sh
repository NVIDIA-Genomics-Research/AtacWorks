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
echo "Split the given regions of the genome into train, val, and holdout/test intervals..."
echo ""
# Each set of intervals will cover the first 10 Mb of a different chromosome
python $root_dir/get_intervals.py \
    --sizes $data_dir/example.sizes \
    --intervalsize 24000 \
    --out_dir $out_dir \
    --prefix result \
    --val chr2 --holdout chr3


echo ""
echo "Verifying output against expected results"
python $utils_dir/verify_diff.py --result_path $out_dir/result.holdout_intervals.bed \
	              --expected_path $expected_results_dir/result.holdout_intervals.bed \
		      --format "general_diff"
check_status $?

python $utils_dir/verify_diff.py --result_path $out_dir/result.val_intervals.bed \
	              --expected_path $expected_results_dir/result.val_intervals.bed \
		      --format "general_diff"
check_status $?

python $utils_dir/verify_diff.py --result_path $out_dir/result.training_intervals.bed \
	              --expected_path $expected_results_dir/result.training_intervals.bed \
		      --format "general_diff"
check_status $?
