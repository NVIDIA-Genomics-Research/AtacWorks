#!/bin/bash

#
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#


set -e

# End-to-end example

# Files provided:

# 1. Clean Track (bigWig) : HSC.80M.chr123.10mb.coverage.bw  
# 2. Clean Peaks (bed): HSC.80M.chr123.10mb.peaks.bed
# 3. Noisy Track (bigWig): HSC.5M.chr123.10mb.coverage.bw
# 4. Noisy Peaks (bed):  HSC.5M.chr123.10mb.peaks.bed
# (These cover the first 10 Mb each of chr1, chr2, and chr3)

# 5. example.sizes - lists the regions of the genome to cover (first 10 Mb each of chr1, chr2, and chr3)

echo ""
echo "Step 0: Initialize environment..."
echo ""
example_dir=$(readlink -f $(dirname "$0"))

data_dir="$example_dir/data"
ref_dir="$example_dir/reference"
out_dir="$example_dir/result"

root_dir=$(readlink -f "$example_dir/..")
saved_model_dir="$root_dir/data/pretrained_models"

# Switch to root directory before running script.
cd $root_dir

if [ -d "$out_dir" ]; then
    rm -rf $out_dir
fi
mkdir -p $out_dir

echo ""
echo "Step 1: Convert peak files into bigWig format..."
echo ""
# Clean peaks
python $root_dir/peak2bw.py \
    $data_dir/HSC.80M.chr123.10mb.peaks.bed \
    $ref_dir/hg19.auto.sizes \
    --prefix=$out_dir/HSC.80M.chr123.10mb.peaks.bed
# Noisy peaks
python $root_dir/peak2bw.py \
    $data_dir/HSC.5M.chr123.10mb.peaks.bed \
    $ref_dir/hg19.auto.sizes \
    --prefix=$out_dir/HSC.5M.chr123.10mb.peaks.bed

echo ""
echo "Step 2: Split the given regions of the genome into train, val, and holdout/test intervals..."
echo ""
# Each set of intervals will cover the first 10 Mb of a different chromosome
python $root_dir/get_intervals.py \
    $data_dir/example.sizes 24000 $out_dir/example \
    --val chr2 --holdout chr3

echo ""
echo "Step 3: Read clean and noisy data in these intervals and save them in .h5 format..."
echo ""
# Training data
python $root_dir/bw2h5.py \
    --noisybw $data_dir/HSC.5M.chr123.10mb.coverage.bw \
    --intervals $out_dir/example.training_intervals.bed \
    --batch_size 4 \
    --prefix $out_dir/train_data \
    --cleanbw $data_dir/HSC.80M.chr123.10mb.coverage.bw \
    --cleanpeakbw $out_dir/HSC.80M.chr123.10mb.peaks.bed.bw \
    --nonzero
# Validation data
python $root_dir/bw2h5.py \
    --noisybw $data_dir/HSC.5M.chr123.10mb.coverage.bw \
    --intervals $out_dir/example.val_intervals.bed \
    --batch_size 64 \
    --prefix $out_dir/val_data \
    --cleanbw $data_dir/HSC.80M.chr123.10mb.coverage.bw \
    --cleanpeakbw $out_dir/HSC.80M.chr123.10mb.peaks.bed.bw
# Test data
python $root_dir/bw2h5.py \
    --noisybw $data_dir/HSC.5M.chr123.10mb.coverage.bw \
    --intervals $out_dir/example.holdout_intervals.bed \
    --batch_size 64 \
    --prefix $out_dir/test_data \
    --cleanbw $data_dir/HSC.80M.chr123.10mb.coverage.bw \
    --cleanpeakbw $out_dir/HSC.80M.chr123.10mb.peaks.bed.bw
#No label
python $root_dir/bw2h5.py \
    --noisybw $data_dir/HSC.5M.chr123.10mb.coverage.bw \
    --intervals $out_dir/example.holdout_intervals.bed \
    --batch_size 64 \
    --prefix $out_dir/no_label \
    --nolabel

echo ""
echo "Step 4: Train and validate model..."
echo ""
python $root_dir/main.py --train \
    --train_files $out_dir/train_data.h5 \
    --val_files $out_dir/val_data.h5 \
    --model resnet --nblocks 5 --nfilt 15 --width 50 \
    --dil 8 --task both --epochs 2 --afunc relu --mse_weight 0.001 \
    --nblocks_cla 2 --nfilt_cla 15 --width_cla 50 --dil_cla 10 \
    --pearson_weight 1 --bs 8 \
    --out_home $out_dir --label HSC.5M.model \
    --checkpoint_fname checkpoint.pth.tar \
    --save_freq=1 --eval_freq=1 --distributed

echo ""
echo "Step 5: Calculate baseline metrics on the test set..."
echo ""
# Regression metrics on noisy data
python $root_dir/calculate_baseline_metrics.py \
    --label_file $out_dir/test_data.h5 --task regression --sep_peaks

# Classification metrics on the noisy data peak calls
python $root_dir/calculate_baseline_metrics.py \
    --label_file $out_dir/test_data.h5 --task classification \
    --test_file $out_dir/HSC.5M.chr123.10mb.peaks.bed.bw \
    --intervals $out_dir/example.holdout_intervals.bed

echo ""
echo "Step 6: Run inference on test set..."
echo ""
# Note: change  --weights_path to the path for your saved model!
python $root_dir/main.py --infer \
    --infer_files $out_dir/test_data.h5 \
    --weights_path $out_dir/HSC.5M.model_latest/model_best.pth.tar \
    --out_home $out_dir --label inference \
    --result_fname HSC.5M.output.h5 \
    --model resnet --nblocks 5 --nfilt 15 --width 50 --dil 8 \
    --nblocks_cla 2 --nfilt_cla 15 --width_cla 50 --dil_cla 10 \
    --task both --num_workers 0

echo ""
echo "Step 7: Calculate metrics after inference..."
echo ""
python $root_dir/calculate_baseline_metrics.py \
    --label_file $out_dir/test_data.h5 --task both \
    --test_file $out_dir/inference_latest/test_data_HSC.5M.output.h5 \
    --sep_peaks --thresholds 0.5

echo ""
echo "Step 8: Postprocess the model output to produce bedGraph and bigWig files..."
echo ""
# NOTE - replace 3rd argument with your inference result file!
# To get predicted coverage track
python $root_dir/postprocess.py \
    $out_dir/example.holdout_intervals.bed \
    $out_dir/inference_latest/test_data_HSC.5M.output.h5 \
    $ref_dir/hg19.auto.sizes \
    $out_dir/HSC.5M.output.track \
    --channel 0 \
    --round 3 \
    --num_worker -1
# To get predicted peaks
python $root_dir/postprocess.py \
    $out_dir/example.holdout_intervals.bed \
    $out_dir/inference_latest/test_data_HSC.5M.output.h5 \
    $ref_dir/hg19.auto.sizes $out_dir/HSC.5M.output.peaks \
    --channel 1 --threshold 0.5

echo ""
echo "Step 9: Summarize the predicted peaks in a BED file..."
echo ""
python $root_dir/peaksummary.py \
    --peakbw $out_dir/HSC.5M.output.peaks.bw \
    --scorebw $out_dir/HSC.5M.output.track.bw \
    --prefix $out_dir/HSC.5M.output.peaks.summarized

#######

echo ""
echo "An alternative method to call peaks..."
echo ""

# To get predicted probabilities for peaks
python $root_dir/postprocess.py \
    $out_dir/example.holdout_intervals.bed \
    $out_dir/inference_latest/test_data_HSC.5M.output.h5 \
    $ref_dir/hg19.auto.sizes $out_dir/HSC.5M.output.probs \
    --channel 1 --round 3

macs2 bdgpeakcall -i $out_dir/HSC.5M.output.probs.bedGraph -o $out_dir/HSC.5M.output.peaks.narrowPeak -c 0.5

#######

echo ""
echo "Alternatively, run inference using a pretrained model on dataset without label..."
echo ""
python $root_dir/main.py --infer \
    --infer_files $out_dir/no_label.h5 \
    --weights_path  $saved_model_dir/bulk_blood_data/5000000.7cell.resnet.5.2.15.8.50.0803.pth.tar\
    --out_home $out_dir --label inference.pretrained \
    --result_fname HSC.5M.output.pretrained.h5 \
    --model resnet --nblocks 5 --nfilt 15 --width 50 --dil 8 \
    --nblocks_cla 2 --nfilt_cla 15 --width_cla 50 --dil_cla 10 \
    --task both --num_workers 0

echo ""
echo "Calculate metrics after inference..."
echo ""
python $root_dir/calculate_baseline_metrics.py \
    --label_file $out_dir/test_data.h5 --task both \
    --test_file $out_dir/inference.pretrained_latest/no_label_HSC.5M.output.pretrained.h5 \
    --sep_peaks --thresholds 0.5

