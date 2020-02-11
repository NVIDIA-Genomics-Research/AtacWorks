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
echo "Convert peak files into bigWig format..."
echo ""
# Clean peaks
python $root_dir/peak2bw.py \
    --input $data_dir/HSC.80M.chr123.10mb.peaks.bed \
    --sizes $ref_dir/hg19.auto.sizes
    --out_dir $out_dir
# Noisy peaks
python $root_dir/peak2bw.py \
    --input $data_dir/HSC.5M.chr123.10mb.peaks.bed \
    --sizes $ref_dir/hg19.auto.sizes \
    --out_dir $out_dir

echo ""
echo "Verifying output against expected results"
python $utils_dir/verify_diff.py --result_path $out_dir/clean.peaks.bw \
                      --expected_path $expected_results_dir/clean.peaks.bw \
		      --format "general_diff"
check_status $?

python $utils_dir/verify_diff.py --result_path $out_dir/noisy.peaks.bw \
	           --expected_path $expected_results_dir/noisy.peaks.bw \
		   --format "general_diff"
check_status $?
