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
atacworks denoise \
    --out_home $out_dir --exp_name inference \
    --weights_path $expected_results_dir/model_latest/model_best.pth.tar \
    --num_workers 4 --distributed \
    --noisybw $data_dir/HSC.5M.chr123.10mb.coverage.bw  \
    --regions "chr3" --interval_size 24000 \
    --genome $data_dir/example.sizes \
    --threshold 0.5 \
    --config_mparams $config_dir/model_structure.yaml \
    --gen_bigwig --batch_size 4 \
    --width 50 --width_cla 50 --dil_cla 10 --pad 0

echo ""
echo "Verifying generated h5 files."
python $utils_dir/verify_diff.py --result_path $out_dir/inference_latest/bw2h5/HSC.5M.chr123.10mb.coverage.bw.denoise.h5 \
    --expected_path $expected_results_dir/inference_latest/bw2h5/HSC.5M.chr123.10mb.coverage.bw.denoise.h5 \
    --format "h5"
check_status $? "Inferred peak bedGraph files do not match!"
echo ""
echo "Verifying output result against expected result."
python $utils_dir/verify_diff.py --result_path $out_dir/inference_latest/HSC_infer.peaks.bedGraph \
    --expected_path $expected_results_dir/inference_latest/HSC_infer.peaks.bedGraph \
    --format "general_diff"
check_status $? "Inferred peak bedGraph files do not match!"
echo ""
echo "Verifying output result against expected result."
python $utils_dir/verify_diff.py --result_path $out_dir/inference_latest/HSC_infer.track.bedGraph \
    --expected_path $expected_results_dir/inference_latest/HSC_infer.track.bedGraph \
    --format "general_diff"
check_status $? "Inferred track bedGraph files do not match!"

echo ""
echo "Run inference on test set with default peak calling setting USING h5 files"
echo ""

atacworks denoise \
    --out_home $out_dir --exp_name h5_inference \
    --weights_path $expected_results_dir/model_latest/model_best.pth.tar \
    --num_workers 0 --gpu_idx 0 \
    --denoise_h5_files $expected_results_dir/inference_latest/bw2h5/HSC.5M.chr123.10mb.coverage.bw.denoise.h5 \
    --intervals_file $expected_results_dir/inference_latest/intervals/24000.regions_intervals.bed \
    --interval_size 24000 \
    --genome $data_dir/example.sizes \
    --threshold 0.5 \
    --config_mparams $config_dir/model_structure.yaml \
    --gen_bigwig --batch_size 4 \
    --width 50 --width_cla 50 --dil_cla 10 --pad 0

echo ""
echo "Verifying output result against expected result."
python $utils_dir/verify_diff.py --result_path $out_dir/h5_inference_latest/HSC_infer.peaks.bedGraph \
    --expected_path $expected_results_dir/inference_latest/HSC_infer.peaks.bedGraph \
    --format "general_diff"
check_status $? "Inferred peak bedGraph files do not match!"
echo ""
echo "Verifying output result against expected result."
python $utils_dir/verify_diff.py --result_path $out_dir/h5_inference_latest/HSC_infer.track.bedGraph \
    --expected_path $expected_results_dir/inference_latest/HSC_infer.track.bedGraph \
    --format "general_diff"
check_status $? "Inferred track bedGraph files do not match!"

echo ""
echo "Run inference in classification only mode..."
echo ""
atacworks denoise \
	--noisybw $data_dir/HSC.5M.chr123.10mb.coverage.bw  \
	--interval_size 24000 \
	--model logistic --field 8401 \
	--genome $data_dir/example.sizes \
	--out_home $out_dir --exp_name logistic_inference \
	--task classification --threshold 0.5 \
	--batch_size 64 --pad 0 --distributed \
	--gen_bigwig \
	--weights_path $expected_results_dir/logistic_latest/model_best.pth.tar

echo ""
echo "Verifying generated h5 files."
python $utils_dir/verify_diff.py --result_path $out_dir/logistic_inference_latest/bw2h5/HSC.5M.chr123.10mb.coverage.bw.denoise.h5 \
    --expected_path $expected_results_dir/logistic_inference_latest/bw2h5/HSC.5M.chr123.10mb.coverage.bw.wg.h5 \
    --format "h5"
check_status $? "Inferred peak h5 files do not match!"
echo ""
echo "Verifying output result against expected result."
python $utils_dir/verify_diff.py --result_path $out_dir/logistic_inference_latest/HSC_infer.peaks.bedGraph \
    --expected_path $expected_results_dir/logistic_inference_latest/HSC_infer.peaks.bedGraph \
    --format "general_diff"
check_status $? "Inferred peak bedGraph files do not match!"
