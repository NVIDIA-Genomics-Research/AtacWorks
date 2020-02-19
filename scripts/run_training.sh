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

tutorial_dir=$(readlink -f $(dirname "$0"))
atacworks=$(readlink -f "$tutorial_dir/..")

function print_help {
    echo "$atacworks/tutorials/run_training.sh <path-to-noisy-and-clean-data-folder> <path-to-intervals-folder> <path-to-configs-folder>"
    echo "<path-to-noisy-clean-data-folder> must contain noisy_data folder, clean_data folder."
    echo "                                  noisy_data folder contains a single bigwig file."
    echo "                                  clean_data folder contains a bigwig file and a narrowPeak file."
    echo " "
    echo "<path-to-intervals-folder>        must contain training_intervals.bed and val_intervals.bed"
    echo " "
    echo "<path-to-configs-folder>          must contain config_params.yaml and model_structure.yaml"
}

if [ $# -ne 3 ]
then
    echo "3 arguments expected, provided only $#. See help!"
    echo " "
    print_help
    exit
fi    

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
clean_narrowpeak=$(ls $1/clean_data | grep ".narrowPeak" | wc -l)
echo "clean_narrowpeak: $clean_narrowpeak"
if [ $clean_narrowpeak -ne 1 ];
then
    echo "Only one bigwig file expected inside $1/clean_data, found $clean_narrowpeak!"
    echo "See help"
    echo " "
    print_help
    exit
fi

training_interval=$(ls $2/* | grep -o "training_intervals.bed")
val_interval=$(ls $2/* | grep -o "val_intervals.bed")
if [ "$training_interval" != "training_intervals.bed" ] || [ "$val_interval" != "val_intervals.bed" ]
then
    echo "training_intervals.bed and val_intervals.bed files expected inside $2!"
    echo "See help. Note that file names are case sensitive."
    echo " "
    print_help
    exit
fi

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

echo "Set environment"
set -e

noisy_atacdata=$(find $1/noisy_data -name "*.bw")
clean_atacdata=$(find $1/clean_data -name "*.bw")
clean_np=$(find $1/clean_data -name "*.narrowPeak")
training_intervals=$(ls $2/* | grep "training_intervals.bed")
echo "training_intervl: $training_intervals"
val_intervals=$(ls $2/* | grep "val_intervals.bed")
echo "val_intervl: $val_intervals"
config_params=$3/config_params.yaml
model_structure=$3/model_structure.yaml

echo "Convert clean peak file into bigWig format"
python $atacworks/peak2bw.py $clean_np $atacworks/example/reference/hg19.chrom.sizes --skip 1

echo "Save the training data and labels into .h5 format"
python $atacworks/bw2h5.py \
           --noisybw $noisy_atacdata \
           --cleanbw $clean_atacdata \
           --cleanpeakbw $clean_np.bw \
           --intervals $training_intervals \
           --prefix train \
	   --pad 5000 \
	   --batch_size 2000 \
           --nonzero

echo "Save the validation data and labels into .h5 format"
python $atacworks/bw2h5.py \
           --noisybw $noisy_atacdata \
           --cleanbw $clean_atacdata \
           --cleanpeakbw $clean_np.bw \
           --intervals $val_intervals \
           --prefix val \
	   --pad 5000 \
	   --batch_size 2000

echo "Train and validate a model using model parameters saved in the config files"
python $atacworks/main.py --train \
        --config $config_params \
        --config_mparams $model_structure \
        --train_files train.h5 \
        --val_files val.h5 \
        --epochs 5
