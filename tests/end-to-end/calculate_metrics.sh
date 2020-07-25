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
echo "Calculate metrics for track coverage after inference..."
echo ""
python $root_dir/calculate_baseline_metrics.py \
    --label_file $data_dir/HSC.80M.chr123.10mb.coverage.bw \
    --task regression \
    --test_file $out_dir/inference_latest/HSC_infer.track.bw \
    --intervals $out_dir/inference_latest/intervals/24000.regions_intervals.bed \
    --sep_peaks --peak_file $expected_results_dir/model_latest/bigwig_peakfiles/HSC.80M.chr123.10mb.peaks.bed.bw >& $out_dir/regression_metrics_log

echo ""
echo "Verifying output model against expected model"
python $utils_dir/verify_diff.py --result_path $out_dir/regression_metrics_log \
    --expected_path $expected_results_dir/regression_metrics_log --format text
check_status $? "Regression metrics do not match!"

echo ""
echo "Calculate metrics for peak classification after inference..."
echo ""
python $root_dir/calculate_baseline_metrics.py \
    --label_file $expected_results_dir/model_latest/bigwig_peakfiles/HSC.80M.chr123.10mb.peaks.bed.bw \
    --task classification \
    --test_file $out_dir/inference_latest/HSC_infer.peaks.bw \
    --intervals $out_dir/inference_latest/intervals/24000.regions_intervals.bed \
    --thresholds 0.5 >& $out_dir/classification_metrics_log

echo ""
echo "Verifying output model against expected model"
python $utils_dir/verify_diff.py --result_path $out_dir/classification_metrics_log \
    --expected_path $expected_results_dir/classification_metrics_log --format text
check_status $? "Classification metrics do not match!"
