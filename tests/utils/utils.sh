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

# Check status and abort test if return value is not 0.
# Additionally print custom message if provided.
function check_status {
    if [ $1 != 0 ]
    then
        if [ ! -z "$2" ]
	then
            echo "$2"
	fi
        echo "Abort test!"
	exit $1
    else
	echo "Success!"
    fi
}
