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

tutorial_dir=$(readlink -f $(dirname "$0"))
atacworks=$(readlink -f "$tutorial_dir/..")

function print_help {
    echo "$atacworks/tutorials/run_training.sh <path-to-model-dir> <path-to-test-folder> <path-to-intervals-folder> <path-to-configs-folder>"
    echo "<path-to-model-dir>        must contain model.pth.tar file"
    echo " "
    echo "<path-to-test-folder>      must contain noisy_data folder and single bigWig file"
    echo " "
    echo "<path-to-intervals-folder> must contain file with suffix test_intervals.bed"
    echo " "
    echo "<path-to-configs-folder>   must contain config_params.yaml and model_structure.yaml"
}


if [ $# -ne 4 ]
then
    echo "4 arguments expected, provided $#. See help!"
    echo " "
    print_help
    exit
fi    

model_dir=$(find $1/ -name "model_best.pth.tar")
if [ "$model_dir" != "$1/model_best.pth.tar" ];
then
    echo "Could not find model_best.pth.tar inside $1!"
    echo "See help. Note that folder and file names are case sensitive."
    echo " "
    print_help
    exit
fi

noisy_data=$(ls -d $2/* | grep -o "noisy_data")
if [ "$noisy_data" != "noisy_data" ]
then
    echo "noisy_data folders expected inside $1!"
    echo "See help. Note that folder names are case sensitive."
    echo " "
    print_help
    exit
fi


noisy_bw=$(ls $2/noisy_data | grep ".bw" | wc -l)
if [ $noisy_bw -ne 1 ];
then
    echo "Only one bigwig file expected inside $2/noisy_data, found $noisy_bw!"
    echo "See help"
    echo " "
    print_help
    exit
fi


test_interval=$(ls $3/* | grep -o "test_intervals.bed")
if [ "$training_interval" != "test_intervals.bed" ]
then
    echo "*test_intervals.bed expected inside $3!"
    echo "See help. Note that file names are case sensitive."
    echo " "
    print_help
    exit
fi


config_file=$(ls $4/* | grep "$4/config_params.yaml")
model_structure=$(ls $4/* | grep -o "$4/model_structure.yaml")
if [ "$config_file" != "$4/config_params.yaml" ] || [ "$model_structure" != "$4/model_structure.yaml" ]
then
    echo "config_params.yaml and model_structure.yaml files expected inside $4!"
    echo "See help. Note that file names are case sensitive."
    echo " "
    print_help
    exit
fi

model="$1/model_best.pth.tar"
config_params=$4/config_params.yaml
model_structure=$4/model_structure.yaml
test_atacdata=$(find $2/noisy_data -name "*.bw")
test_intervals=$(ls $3/* | grep "test_intervals.bed")

echo "Read test data over selected intervals and save into .h5 format"
python $atacworks/bw2h5.py \
           --noisybw $test_atacdata \
           --intervals $test_intervals \
           --prefix test_data \
           --pad 5000 \
           --batch_size 2000 \
           --nolabel

echo "Inference on selected intervals, producing denoised track and binary peak calls"
python $atacworks/main.py --infer \
    --infer_files test_data.h5 \
    --sizes_file $atacworks/example/reference/hg19.auto.sizes \
    --config $config_params \
    --config_mparams $model_structure \
    --infer_threshold 0.5
