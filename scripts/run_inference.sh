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

script_dir=$(readlink -f $(dirname "$0"))
default_config_dir=$script_dir/../configs

# Define args
TEST_DATA_BW=$1
MODEL_FILE=$2
SIZES_FILE=$3
OUT_DIR=$4
CONFIGS_FOLDER=${5:-$default_config_dir}

echo "Set environment"
set -e

function print_help {
    echo "$atacworks/scripts/run_inference.sh <path-to-test-bigWig-file> <path-to-model-file>
    <path-to-chromosome-sizes-file> <out_dir> <path-to-configs-folder>"
    echo "<path-to-test-file>            path to a single bigWig file containing noisy ATAC-seq data for inference"
    echo " "
    echo "<path-to-model-file>           path to .pth.tar file containing saved model"
    echo " "
    echo "<path-to-chromosome-sizes-file> tab separated file with names and sizes of all chromosomes to test on"
    echo " "
    echo "<out_dir>                      path to output directory."
    echo " "
    echo "<path-to-configs-folder>       must contain config_params.yaml and model_structure.yaml. Optional. If not provided, defaults to AtacWorks/configs."
}

# Check test data
echo "BigWig file containing test ATAC-seq data: $TEST_DATA_BW"
test_bw_ext="${TEST_DATA_BW##*.}"
if [ $test_bw_ext != "bw" ];
then
    echo "The test ATAC-seq coverage track file is expected to have extension bw. This file has extension $test_bw_ext."
    echo "See help"
    echo " "
    print_help
    exit
fi

# Check test data
echo "Model file: $MODEL_FILE"
model_file_ext="${MODEL_FILE##*.}"
if [ $model_file_ext != "tar" ];
then
    echo "The test ATAC-seq coverage track file is expected to have extension tar. This file has extension $model_file_ext."
    echo "See help"
    echo " "
    print_help
    exit
fi

# Check config files
echo "Configs folder: $CONFIGS_FOLDER"
config_file=$(ls $CONFIGS_FOLDER/* | grep "$CONFIGS_FOLDER/config_params.yaml")
model_structure=$(ls $CONFIGS_FOLDER/* | grep -o "$CONFIGS_FOLDER/model_structure.yaml")
if [ "$config_file" != "$CONFIGS_FOLDER/config_params.yaml" ] || [ "$model_structure" != "$CONFIGS_FOLDER/model_structure.yaml" ]
then
    echo "config_params.yaml and model_structure.yaml files expected inside $CONFIGS_FOLDER!"
    echo "See help. Note that file names are case sensitive."
    echo " "
    print_help
    exit
fi

# get_intervals
echo "Make test intervals"
python $script_dir/get_intervals.py --sizes $SIZES_FILE --intervalsize 50000 --out_dir $OUT_DIR --wg

#bw2h5
echo "Read test data over selected intervals and save into .h5 format"
python $script_dir/bw2h5.py --noisybw $TEST_DATA_BW --intervals $OUT_DIR/genome_intervals.bed --out_dir $OUT_DIR --prefix test_data --pad 5000 --nolabel

#inference
echo "Inference on selected intervals, producing denoised track and binary peak calls"
python $script_dir/main.py --infer --infer_files $OUT_DIR/test_data.h5 --weights_path $MODEL_FILE --sizes_file $SIZES_FILE --intervals_file $OUT_DIR/genome_intervals.bed --config $config_file --config_mparams $model_structure

#peaksummary
# Run only if an inference threshold is provided
if grep -q "infer_threshold: [0-9]" $config_file
then
    inferred_dir=$OUT_DIR/$(grep "label:" $config_file| cut -f 2 -d " ")_latest
    track_file=$(ls $inferred_dir | grep "track.bw")
    peak_file=$(ls $inferred_dir | grep "peaks.bw")
    echo "Summarize peak calls."
    python $script_dir/peaksummary.py \
        --peakbw $inferred_dir/$peak_file \
        --trackbw $inferred_dir/$track_file \
        --out_dir $inferred_dir \
        --prefix infer_results_peaks \
        --minlen 20
fi