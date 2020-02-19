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

echo "Set environment"
set -e

script_dir=$(readlink -f $(dirname "$0"))

function print_help {
    echo "$atacworks/tutorials/run_training.sh <path-to-model-file> <path-to-test-file>  <path-to-configs-folder> <path-to-chromosome-sizes-file> <out_dir>"
    echo "<path-to-model-file>           path to .pth.tar file"
    echo " "
    echo "<path-to-test-file>            single bigWig file"
    echo " "
    echo "<path-to-configs-folder>       must contain config_params.yaml and model_structure.yaml"
    echo " "
    echo "<path-to-chromosome-sizes-file> tab separated file with names and sizes of all chromosomes to test on"
    echo " "
    echo "<out_dir>                      path to output directory."
}


# Check number of args
if [ $# -ne 4 ]
then
    echo "4 arguments expected, provided $#. See help!"
    echo " "
    print_help
    exit
fi    

# Todo: check that $1 ends with extension .pth.tar
# Todo: check that $2 ends with extension .bw

# Check config files
config_file=$(ls $3/* | grep "$3/config_params.yaml")
model_structure=$(ls $3/* | grep -o "$3/model_structure.yaml")
if [ "$config_file" != "$3/config_params.yaml" ] || [ "$model_structure" != "$3/model_structure.yaml" ]
then
    echo "config_params.yaml and model_structure.yaml files expected inside $3!"
    echo "See help. Note that file names are case sensitive."
    echo " "
    print_help
    exit
fi

model=$1
test_atacdata=$2
config_params=$3/config_params.yaml
model_structure=$3/model_structure.yaml
sizes_file=$4

echo "Make test intervals"
python $script_dir/get_intervals.py --sizes $sizes_file --intervalsize 50000 --out_dir $out_dir --wg

echo "Read test data over selected intervals and save into .h5 format"
python $script_dir/bw2h5.py --noisybw $test_atacdata --intervals $out_dir/genome_intervals.bed --prefix test_data --pad 5000 --nolabel

echo "Inference on selected intervals, producing denoised track and binary peak calls"
python $script_dir/main.py --infer --infer_files $out_dir/test_data.h5 --sizes_file $sizes_file --config $config_params --config_mparams $model_structure
