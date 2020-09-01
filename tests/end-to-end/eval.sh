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
atacworks eval \
    --out_home $out_dir --exp_name evaluate \
    --weights_path $expected_results_dir/model_latest/model_best.pth.tar \
    --num_workers 0 --gpu_idx 0 \
    --cleanbw $data_dir/HSC.80M.chr123.10mb.coverage.bw \
    --noisybw $data_dir/HSC.5M.chr123.10mb.coverage.bw  \
    --cleanpeakfile $data_dir/HSC.80M.chr123.10mb.peaks.bed \
    --regions "chr3" --interval_size 24000 \
    --genome $data_dir/example.sizes \
    --threshold 0.5 \
    --config_mparams $config_dir/model_structure.yaml \
    --gen_bigwig --batch_size 4 \
    --width 50 --width_cla 50 --dil_cla 10 --pad 0

echo ""
echo "Verifying bigiwg peakfiles."
python $utils_dir/verify_diff.py --result_path $out_dir/evaluate_latest/bigwig_peakfiles/HSC.80M.chr123.10mb.peaks.bed.bw \
    --expected_path $expected_results_dir/evaluate_latest/bigwig_peakfiles/HSC.80M.chr123.10mb.peaks.bed.bw \
    --format "general_diff"
check_status $? "Bigwig peakfiles do not match!"

echo ""
echo "Verifying generated h5 files."
python $utils_dir/verify_diff.py --result_path $out_dir/evaluate_latest/bw2h5/HSC.5M.chr123.10mb.coverage.bw.eval.h5 \
    --expected_path $expected_results_dir/evaluate_latest/bw2h5/HSC.5M.chr123.10mb.coverage.bw.eval.h5 \
    --format "h5"
check_status $? "Inferred peak bedGraph files do not match!"

echo ""
echo "Verifying created intervals files"
python $utils_dir/verify_diff.py --result_path $out_dir/evaluate_latest/intervals/24000.regions_intervals.bed \
    --expected_path $expected_results_dir/evaluate_latest/intervals/24000.regions_intervals.bed \
    --format "general_diff"
check_status $? "interval files for train data do not match!"

source $utils_dir/utils.sh
echo ""
echo "Run eval on test set with distributed and multi-threaded settings..."
echo ""
atacworks eval \
    --out_home $out_dir --exp_name evaluate_distributed \
    --weights_path $expected_results_dir/model_latest/model_best.pth.tar \
    --num_workers 4 --distributed \
    --cleanbw $data_dir/HSC.80M.chr123.10mb.coverage.bw \
    --noisybw $data_dir/HSC.5M.chr123.10mb.coverage.bw  \
    --cleanpeakfile $data_dir/HSC.80M.chr123.10mb.peaks.bed \
    --regions "chr3" --interval_size 24000 \
    --genome $data_dir/example.sizes \
    --threshold 0.5 \
    --config_mparams $config_dir/model_structure.yaml \
    --gen_bigwig --batch_size 4 \
    --width 50 --width_cla 50 --dil_cla 10 --pad 0

echo ""
echo "Verifying bigiwg peakfiles."
python $utils_dir/verify_diff.py --result_path $out_dir/evaluate_distributed_latest/bigwig_peakfiles/HSC.80M.chr123.10mb.peaks.bed.bw \
    --expected_path $expected_results_dir/evaluate_latest/bigwig_peakfiles/HSC.80M.chr123.10mb.peaks.bed.bw \
    --format "general_diff"
check_status $? "Bigwig peakfiles do not match!"

echo ""
echo "Verifying generated h5 files."
python $utils_dir/verify_diff.py --result_path $out_dir/evaluate_distributed_latest/bw2h5/HSC.5M.chr123.10mb.coverage.bw.eval.h5 \
    --expected_path $expected_results_dir/evaluate_latest/bw2h5/HSC.5M.chr123.10mb.coverage.bw.eval.h5 \
    --format "h5"
check_status $? "Inferred peak bedGraph files do not match!"

echo ""
echo "Verifying created intervals files"
python $utils_dir/verify_diff.py --result_path $out_dir/evaluate_distributed_latest/intervals/24000.regions_intervals.bed \
    --expected_path $expected_results_dir/evaluate_latest/intervals/24000.regions_intervals.bed \
    --format "general_diff"
check_status $? "interval files for train data do not match!"
