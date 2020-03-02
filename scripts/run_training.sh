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
NOISY_DATA_BW=$1
CLEAN_DATA_BW=$2
CLEAN_PEAKS=$3
SIZES_FILE=$4
OUT_DIR=$5
VAL_CHR=$6
HOLDOUT_CHR=$7
CONFIGS_FOLDER=${8:-$default_config_dir}


function print_help {
    echo "$atacworks/scripts/run_training.sh <path-to-noisy-data-bigWig> <path-to-clean-data-bigWig> <path-to-clean-peaks> <path-to-chromosome-sizes-file> <path-to-output-folder> <chromosome-for-validation> <chromosome-for-holdout> <path-to-configs-folder>"
    echo "<path-to-noisy-data-bigWig>      bigWig file containing a noisy ATAC-seq coverage track. This file should have extension .bw"
    echo ""
    echo "<path-to-clean-data-bigWig> bigWig file containing a clean ATAC-seq coverage track from the same cell type. This file should have extension .bw"
    echo ""
    echo"<path-to-clean-peaks> narrowPeak or BED file containing peak calls from the clean ATAC-seq coverage track. The first line of this file is expected to be a header. This file should have extension .bed or .narrowPeak"
    echo " "
    echo "<out_dir>                         path to output directory."
    echo " "
    echo "<path-to-chromosome-sizes-file>   tab-separated file containing chromosome names and sizes"
    echo " "
    echo "<chromosome for validation>       name of chromosome to use for validation"
    echo " "
    echo "<chromosome for holdout>          name of chromosome to hold out from training"
    echo "<path-to-configs-folder>          must contain train_config.yaml and model_structure.yaml. Optional. If not provided, defaults to AtacWorks/configs."
    echo " "

}


# Check noisy data
echo "BigWig file containing noisy ATAC-seq data: $NOISY_DATA_BW"
noisy_bw_ext="${NOISY_DATA_BW##*.}"
if [ $noisy_bw_ext != "bw" ];
then
    echo "The noisy ATAC-seq coverage track file is expected to have extension bw. This file has extension $noisy_bw_ext."
    echo "See help"
    echo " "
    print_help
    exit
fi

# Check clean data
echo "BigWig file containing clean ATAC-seq data: $CLEAN_DATA_BW"
clean_bw_ext="${CLEAN_DATA_BW##*.}"
if [ $clean_bw_ext != "bw" ];
then
    echo "The clean ATAC-seq coverage track file is expected to have extension bw. This file has extension $clean_bw_ext."
    echo "See help"
    echo " "
    print_help
    exit
fi

# Check clean peaks
echo "File containing clean ATAC-seq peak calls: $CLEAN_PEAKS"
clean_peaks_ext="${CLEAN_PEAKS##*.}"
if [ $clean_peaks_ext != "bed" ] && [ $clean_peaks_ext != "narrowPeak" ];
then
    echo "The clean ATAC-seq peak file is expected to have extension bed or narrowPeak. This file has extension $clean_peaks_ext."
    echo "See help"
    echo " "
    print_help
    exit
fi

# Check config files
config_file=$(ls $CONFIGS_FOLDER/* | grep "$CONFIGS_FOLDER/train_config.yaml")
model_structure=$(ls $CONFIGS_FOLDER/* | grep -o "$CONFIGS_FOLDER/model_structure.yaml")
if [ "$config_file" != "$CONFIGS_FOLDER/train_config.yaml" ] || [ "$model_structure" != "$CONFIGS_FOLDER/model_structure.yaml" ]
then
    echo "train_config.yaml and model_structure.yaml files expected inside $CONFIGS_FOLDER!"
    echo "See help. Note that file names are case sensitive."
    echo " "
    print_help
    exit
fi

# Set environment
echo "Set environment"
set -e

# Set variables
clean_peaks_prefix=$(basename $CLEAN_PEAKS)

# peak2bw
echo "Convert clean peak file into bigWig format"
python $script_dir/peak2bw.py --input $CLEAN_PEAKS \
    --sizes $SIZES_FILE \
    --out_dir $OUT_DIR \
    --skip 1

# get_intervals
echo "Generate train/val/test intervals"
python $script_dir/get_intervals.py --sizes $SIZES_FILE \
    --intervalsize 50000 \
    --out_dir $OUT_DIR \
    --val $VAL_CHR --holdout $HOLDOUT_CHR

# Extract and encode training data
echo "Save the training data and labels into .h5 format"
python $script_dir/bw2h5.py --noisybw $NOISY_DATA_BW \
    --cleanbw $CLEAN_DATA_BW \
    --cleanpeakbw $OUT_DIR/$clean_peaks_prefix.bw \
    --intervals $OUT_DIR/training_intervals.bed \
    --out_dir $OUT_DIR --prefix train \
    --pad 5000 --nonzero

# Extract and encode validation data
echo "Save the validation data and labels into .h5 format"
python $script_dir/bw2h5.py --noisybw $NOISY_DATA_BW \
    --cleanbw $CLEAN_DATA_BW \
    --cleanpeakbw $OUT_DIR/$clean_peaks_prefix.bw \
    --intervals $OUT_DIR/val_intervals.bed \
    --out_dir $OUT_DIR --prefix val \
    --pad 5000

echo "Train and validate a model using model parameters saved in the config files"
python $script_dir/main.py train \
    --config $config_file \
    --config_mparams $model_structure \
    --train_files $OUT_DIR/train.h5 \
    --val_files $OUT_DIR/val.h5 \
    --out_home $OUT_DIR
