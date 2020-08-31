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

set -e

echo ""
echo "Initialize environment..."
echo ""
export test_dir=$(readlink -f $(dirname "$0"))
export test_root_dir=$(readlink -f "$test_dir/..")
export root_dir=$(readlink -f "$test_root_dir/../scripts")
export data_dir="$test_root_dir/data/end-to-end"
export ref_dir="$test_root_dir/reference"
export out_dir="$test_root_dir/result/end-to-end"
export expected_results_dir="$test_root_dir/expected_results/end-to-end"
export utils_dir="$test_root_dir/utils"
export saved_model_dir="$test_root_dir/data/end-to-end/pretrained_models"
export config_dir="$root_dir/../configs"

# Switch to root directory before running script.
cd $root_dir

if [ -d "$out_dir" ]; then
    rm -rf $out_dir
fi
mkdir -p $out_dir

bash $test_dir/train.sh
bash $test_dir/infer.sh
bash $test_dir/eval.sh
bash $test_dir/get_summary.sh
bash $test_dir/calculate_metrics.sh
