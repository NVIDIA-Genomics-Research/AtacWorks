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

function print_help {
    echo "$atacworks/scripts/run_inference.sh -bw <path-to-test-bigWig-file> -m <path-to-model-file>
    -f <path-to-chromosome-sizes-file> -o <out_dir> -c <path-to-configs-folder>"
    echo "-bw | --bigwig      path to a single bigWig file containing noisy ATAC-seq data for inference"
    echo " "
    echo "-m | --model        path to .pth.tar file containing saved model"
    echo " "
    echo "-f | --sizesfile    tab separated file with names and sizes of all chromosomes to test on"
    echo " "
    echo "-o | --outdir       path to output directory."
    echo " "
    echo "-c | --cfgdir       must contain infer_config.yaml and model_structure.yaml. Optional. If not provided, defaults to AtacWorks/configs."
}

function is_initialized {
if [ -z "$1" ]
then
    echo "Required variables not provided. See help!"
    print_help
    exit
fi
}


script_dir=$(readlink -f $(dirname "$0"))
default_config_dir=$script_dir/../configs
CONFIGS_FOLDER=$default_config_dir

echo "Set environment"
set -e
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -bw|--bigwig)
    TEST_DATA_BW="$2"
    shift # past argument
    shift # past value
    ;;
    -m|--model)
    MODEL_FILE="$2"
    shift # past argument
    shift # past value
    ;;
    -f|--sizesfile)
    SIZES_FILE="$2"
    shift # past argument
    shift # past value
    ;;
    -o|--outdir)
    OUT_DIR="$2"
    shift # past argument
    shift # past value
    ;;
    -c|--cfgdir)
    CONFIGS_FOLDER="$2"
    shift # past argument
    shift # past value
    ;;
    --default)
    DEFAULT=YES
    shift # past argument
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

# Check to see all required arguments are initialized.
is_initialized $TEST_DATA_BW
is_initialized $MODEL_FILE
is_initialized $SIZES_FILE
is_initialized $OUT_DIR

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
config_file=$(ls $CONFIGS_FOLDER/* | grep "$CONFIGS_FOLDER/infer_config.yaml")
model_structure=$(ls $CONFIGS_FOLDER/* | grep -o "$CONFIGS_FOLDER/model_structure.yaml")
if [ "$config_file" != "$CONFIGS_FOLDER/infer_config.yaml" ] || [ "$model_structure" != "$CONFIGS_FOLDER/model_structure.yaml" ]
then
    echo "infer_config.yaml and model_structure.yaml files expected inside $CONFIGS_FOLDER!"
    echo "See help. Note that file names are case sensitive."
    echo " "
    print_help
    exit
fi

# get_intervals
echo "Make test intervals"
python $script_dir/get_intervals.py --sizes $SIZES_FILE \
    --intervalsize 50000 --out_dir $OUT_DIR --wg

#Extract and encode data
echo "Read test data over selected intervals and save into .h5 format"
python $script_dir/bw2h5.py --noisybw $TEST_DATA_BW \
    --intervals $OUT_DIR/genome_intervals.bed \
    --out_dir $OUT_DIR --prefix test_data \
    --pad 5000 --nolabel

#inference
echo "Inference on selected intervals, producing denoised track and binary peak calls"
python $script_dir/main.py infer \
    --infer_files $OUT_DIR/test_data.h5 \
    --weights_path $MODEL_FILE \
    --sizes_file $SIZES_FILE \
    --intervals_file $OUT_DIR/genome_intervals.bed \
    --config $config_file --config_mparams $model_structure

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
