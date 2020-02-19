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

function print_help {
    echo "$atacworks/tutorials/run_training.sh <path-to-noisy-clean-data-folder> <path-to-configs-folder> <path-to-chromosome-sizes-file> <chromosome for validation> <chromosome for holdout> <out_dir>"
    echo "<path-to-noisy-clean-data-folder> must contain noisy_data folder, clean_data folder."
    echo "                                  noisy_data folder contains a single bigwig file."
    echo "                                  clean_data folder contains a bigwig file and a narrowPeak or BED file. The first line of the narrowPeak or BED file is expected to be a header."
    echo " "
    echo "<path-to-configs-folder>          must contain config_params.yaml and model_structure.yaml"
    echo " "
    echo "<path-to-chromosome-sizes-file>   tab-separated file containing chromosome names and sizes"
    echo " "
    echo "<chromosome for validation>       name of chromosome to use for validation"
    echo " "
    echo "<chromosome for holdout>          name of chromosome to hold out from training"
    echo " "
    echo "<out_dir>                         path to output directory."

}

# Check number of args
if [ $# -ne 6 ]
then
    echo "6 arguments expected, provided only $#. See help!"
    echo " "
    print_help
    exit
fi    

# Check clean and noisy data
clean_data=$(ls -d $1/* | grep -o "clean_data")
noisy_data=$(ls -d $1/* | grep -o "noisy_data")
if [ "$clean_data" != "clean_data" ] || [ "$noisy_data" != "noisy_data" ]
then
    echo "clean_data and noisy_data folders expected inside $1!"
    echo "See help. Note that folder names are case sensitive."
    echo " "
    print_help
    exit
fi

noisy_bw=$(ls $1/noisy_data | grep ".bw" | wc -l)
echo "noisy_bw: $noisy_bw"
if [ $noisy_bw -ne 1 ];
then
    echo "Only one bigwig file expected inside $1/noisy_data, found $noisy_bw!"
    echo "See help"
    echo " "
    print_help
    exit
fi

clean_bw=$(ls $1/clean_data | grep ".bw" | wc -l)
echo "clean_bw: $clean_bw"
if [ $clean_bw -ne 1 ];
then
    echo "Only one bigwig file expected inside $1/clean_data, found $clean_bw!"
    echo "See help"
    echo " "
    print_help
    exit
fi
clean_peak=$(ls $1/clean_data | grep ".narrowPeak\|.bed" | wc -l)
echo "clean_peak: $clean_peak"
if [ $clean_peak -ne 1 ];
then
    echo "Only one peak file expected inside $1/clean_data, found $clean_peak!"
    echo "See help"
    echo " "
    print_help
    exit
fi

# Check config files
config_file=$(ls $2/* | grep "$2/config_params.yaml")
model_structure=$(ls $2/* | grep -o "$2/model_structure.yaml")
if [ "$config_file" != "$2/config_params.yaml" ] || [ "$model_structure" != "$2/model_structure.yaml" ]
then
    echo "config_params.yaml and model_structure.yaml files expected inside $2!"
    echo "See help. Note that file names are case sensitive."
    echo " "
    print_help
    exit
fi

# Set environment
echo "Set environment"
set -e

# Set variables
noisy_atacdata=$(find $1/noisy_data -name "*.bw")
clean_atacdata=$(find $1/clean_data -name "*.bw")
clean_np=$(find $1/clean_data \(-name "*.narrowPeak" -o -name "*.bed"\))
config_params=$2/config_params.yaml
model_structure=$2/model_structure.yaml
sizes_file=$3
valchr=$4
holdoutchr=$5
out_dir=$6

echo "Convert clean peak file into bigWig format"
python $script_dir/peak2bw.py --input $clean_np --sizes $sizes_file --out_dir $out_dir --skip 1

echo "Generate train/val/test intervals"
python $script_dir/get_intervals.py --sizes $sizes_file --intervalsize 50000 --out_dir $out_dir --val $valchr --holdout $holdoutchr

echo "Save the training data and labels into .h5 format"
python $script_dir/bw2h5.py --noisybw $noisy_atacdata --cleanbw $clean_atacdata --cleanpeakbw $out_dir/$clean_np.bw --intervals $out_dir/training_intervals.bed --out_dir $out_dir --prefix train --pad 5000 --nonzero

echo "Save the validation data and labels into .h5 format"
python $script_dir/bw2h5.py --noisybw $noisy_atacdata --cleanbw $clean_atacdata --cleanpeakbw $out_dir/$clean_np.bw --intervals $out_dir/val_intervals.bed --out_dir $out_dir --prefix val --pad 5000

echo "Train and validate a model using model parameters saved in the config files"
python $script_dir/main.py --train --config $config_params --config_mparams $model_structure --train_files $out_dir/train.h5 --val_files $out_dir/val.h5
