
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

export test_dir=$(readlink -f $(dirname "$0"))
export root_dir=$(readlink -f "$test_root_dir/../..")
export out_dir="$test_root_dir/result/end-to-end"
export utils_dir="$test_root_dir/utils"

source $utils_dir/utils.sh

if [ -d "$out_dir" ]; then
    rm -rf $out_dir
fi
mkdir -p $out_dir

echo ""
echo "Parsing tutorial scripts for code snippets."
# Parse the tutorials and create bash shell scripts.
python $test_dir/parse_tutorials.py --root_dir $root_dir --out_dir $root_dir
check_status $? "Parsing of markdown files failed!"

#Run the shell scripts
bash $root_dir/tutorial1.sh
check_status $? "Tutorial1 code failed!"
bash $root_dir/tutorial2.sh
check_status $? "Tutorial2 code failed!"
