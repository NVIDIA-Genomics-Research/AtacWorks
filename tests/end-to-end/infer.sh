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

python $root_dir/main.py infer \
    --out_home $out_dir --label inference \
    --weights_path $expected_results_dir/model_latest/model_best.pth.tar \
    --num_workers 0 \
    --files $out_dir/no_label.h5 \
    --intervals_file $out_dir/result.holdout_intervals.bed \
    --sizes_file $ref_dir/hg19.auto.sizes \
    --infer_threshold 0.5 \
    --result_fname infer \
    --config_mparams $config_dir/model_structure.yaml \
    --gen_bigwig --bs 4 \
    --width 50 --width_cla 50 --dil_cla 10 --pad 0

echo ""
echo "Verifying output result against expected result."
python $utils_dir/verify_diff.py --result_path $out_dir/inference_latest/no_label_infer.peaks.bedGraph \
    --expected_path $expected_results_dir/inference_latest/no_label_infer.peaks.bedGraph \
    --format "general_diff"
check_status $? "Inferred peak bedGraph files do not match!"
echo ""
echo "Verifying output result against expected result."
python $utils_dir/verify_diff.py --result_path $out_dir/inference_latest/no_label_infer.track.bedGraph \
    --expected_path $expected_results_dir/inference_latest/no_label_infer.track.bedGraph \
    --format "general_diff"
check_status $? "Inferred track bedGraph files do not match!"

echo ""
echo "Run inference in classification only mode..."
echo ""
python $root_dir/main.py infer \
	--files $out_dir/no_label.h5 \
	--model logistic --field 8401 \
	--out_home $out_dir --label logistic_inference \
	--task classification \
	--bs 64 --pad 0 --distributed \
	--intervals_file $out_dir/result.holdout_intervals.bed \
	--sizes_file $ref_dir/hg19.auto.sizes \
	--result_fname inferred \
	--weights_path $expected_results_dir/logistic_model_latest/model_best.pth.tar
echo ""
echo "Verifying output result against expected result."
python $utils_dir/verify_diff.py --result_path $out_dir/logistic_inference_latest/no_label_inferred.peaks.bedGraph \
    --expected_path $expected_results_dir/logistic_inference_latest/no_label_inferred.peaks.bedGraph \
    --format "general_diff"
check_status $? "Inferred peak bedGraph files for classification test do not match!"
