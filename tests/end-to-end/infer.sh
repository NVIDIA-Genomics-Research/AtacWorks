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
echo "Run inference on test set with default peak calling setting..."
echo ""

# Feature alert: usage of --config_mparams. 
##This option allows you to specify a custom model config file.

# Note: change --weights_path to the path for your saved model!
python $root_dir/main.py --infer \
    --infer_files $out_dir/no_label.h5 \
    --intervals_file $out_dir/result.holdout_intervals.bed \
    --sizes_file $ref_dir/hg19.auto.sizes \
    --infer_threshold 0.5 result_fname infer \
    --weights_path $expected_results_dir/model_latest/model_best.pth.tar \
    --out_home $out_dir --label inference --config_mparams $config_dir/model_structure.yaml \
    --num_workers 0 --gen_bigwig

echo ""
echo "Verifying output result against expected result."
python $utils_dir/verify_diff.py --result_path $out_dir/inference_latest/no_label_infer_results.h5.peaks.bedGraph \
    --expected_path $expected_results_dir/inference_latest/no_label_infer_results.h5.peaks.bedGraph \
    --format "general_diff"
check_status $?
echo ""
echo "Verifying output result against expected result."
python $utils_dir/verify_diff.py --result_path $out_dir/inference_latest/no_label_infer_results.h5.track.bedGraph \
    --expected_path $expected_results_dir/inference_latest/no_label_infer_results.h5.track.bedGraph \
    --format "general_diff"
check_status $?
