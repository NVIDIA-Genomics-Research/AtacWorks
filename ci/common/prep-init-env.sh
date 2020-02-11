#
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

################################################################################
# SETUP - Check environment
################################################################################

logger "Get env..."
env

logger "Check versions..."
gcc --version
g++ --version

logger "Activate anaconda enviroment..."
CONDA_NEW_ACTIVATION_CMD_VERSION="4.4"
CONDA_VERSION=$(conda --version | awk '{print $2}')
if [ "$CONDA_NEW_ACTIVATION_CMD_VERSION" == "$(echo -e "$CONDA_VERSION\n$CONDA_NEW_ACTIVATION_CMD_VERSION" | sort -V | head -1)" ]; then
  logger "Version is higer than ${CONDA_NEW_ACTIVATION_CMD_VERSION}, using conda activate"
  source /conda/etc/profile.d/conda.sh
  conda activate "${2}"
else
  logger "Version is lower than ${CONDA_NEW_ACTIVATION_CMD_VERSION}, using source activate"
  source activate "${2}"
fi

# FIX Added to deal with Anancoda SSL verification issues during conda builds
conda config --set ssl_verify False

logger "Conda install ClaraGenomicsAnalysis custom packages - clang-format"
conda install --override-channels -c conda-forge hdf5
################################################################################
# BUILD - Conda package builds 
################################################################################

CUDA_REL=${CUDA:0:3}
if [ "${CUDA:0:2}" == '10' ]; then
  # CUDA 10 release
  CUDA_REL=${CUDA:0:4}
fi

# Cleanup local git
cd $1
git clean -xdf

cd $1
