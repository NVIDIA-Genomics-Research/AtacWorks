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
atacworks train\
    --out_home $out_dir \
    --exp_name model \
    --distributed \
    --cleanbw $data_dir/HSC.80M.chr123.10mb.coverage.bw \
    --noisybw $data_dir/HSC.5M.chr123.10mb.coverage.bw \
    --genome $data_dir/example.sizes \
    --interval_size 24000 --val_chrom chr2 --holdout_chrom chr3 \
    --cleanpeakfile $data_dir/HSC.80M.chr123.10mb.peaks.bed \
    --checkpoint_fname checkpoint.pth.tar \
    --epochs 1 --batch_size 4 \
    --width 50 --width_cla 50 --dil_cla 10 --pad 0
# Training is not deterministic, so we are not comparing results.
check_status $? "Training run not succesful!"

# Check all the output files are bitwise equal
echo ""
echo "Verifying bigiwg peakfiles."
python $utils_dir/verify_diff.py --result_path $out_dir/model_latest/bigwig_peakfiles/HSC.80M.chr123.10mb.peaks.bed.bw \
    --expected_path $expected_results_dir/model_latest/bigwig_peakfiles/HSC.80M.chr123.10mb.peaks.bed.bw \
    --format "general_diff"
check_status $? "Bigwig peakfiles do not match!"

echo ""
echo "Verifying created h5 files."
python $utils_dir/verify_diff.py --result_path $out_dir/model_latest/bw2h5/HSC.80M.chr123.10mb.coverage.bw.train.h5 \
    --expected_path $expected_results_dir/model_latest/bw2h5/HSC.80M.chr123.10mb.coverage.bw.train.h5 \
    --format "h5"
check_status $? "h5 train files do not match!"
python $utils_dir/verify_diff.py --result_path $out_dir/model_latest/bw2h5/HSC.80M.chr123.10mb.coverage.bw.val.h5 \
    --expected_path $expected_results_dir/model_latest/bw2h5/HSC.80M.chr123.10mb.coverage.bw.val.h5 \
    --format "h5"
check_status $? "h5 val files do not match!"

echo ""
echo "Verifying created intervals files"
python $utils_dir/verify_diff.py --result_path $out_dir/model_latest/intervals/24000.training_intervals.bed \
    --expected_path $expected_results_dir/model_latest/intervals/24000.training_intervals.bed \
    --format "general_diff"
check_status $? "interval files for train data do not match!"
python $utils_dir/verify_diff.py --result_path $out_dir/model_latest/intervals/24000.val_intervals.bed \
    --expected_path $expected_results_dir/model_latest/intervals/24000.val_intervals.bed \
    --format "general_diff"
check_status $? "interval files for validation data do not match!"
python $utils_dir/verify_diff.py --result_path $out_dir/model_latest/intervals/24000.holdout_intervals.bed \
    --expected_path $expected_results_dir/model_latest/intervals/24000.holdout_intervals.bed \
    --format "general_diff"
check_status $? "interval files for holdout data do not match!"


echo ""
echo "Train with h5 files as input"
echo ""
atacworks train\
    --out_home $out_dir \
    --exp_name h5_model \
    --distributed \
    --train_h5_files $expected_results_dir/model_latest/bw2h5/HSC.80M.chr123.10mb.coverage.bw.train.h5 \
    --val_h5_files  $expected_results_dir/model_latest/bw2h5/HSC.80M.chr123.10mb.coverage.bw.val.h5 \
    --genome $data_dir/example.sizes \
    --interval_size 24000 \
    --checkpoint_fname checkpoint.pth.tar \
    --epochs 1 --batch_size 4 \
    --width 50 --width_cla 50 --dil_cla 10 --pad 0
# Training is not deterministic, so we are not comparing results.
check_status $? "Training run not succesful!"

echo ""
echo "Test classification mode of training"
echo ""
atacworks train \
    --cleanbw $data_dir/HSC.80M.chr123.10mb.coverage.bw \
    --noisybw $data_dir/HSC.5M.chr123.10mb.coverage.bw \
    --genome $data_dir/example.sizes --gpu 0 \
    --interval_size 24000 --val_chrom chr2 --holdout_chrom chr3 \
    --cleanpeakfile $data_dir/HSC.80M.chr123.10mb.peaks.bed \
    --model logistic --field 8401 \
    --out_home $out_dir --exp_name logistic \
    --task classification --batch_size 4 \
    --epochs 1 --pad 5000
# Training is not deterministic, so we are not comparing results.
check_status $? "Training run not succesful!"

# The above code already tests for all the diffs, no need to test again.
