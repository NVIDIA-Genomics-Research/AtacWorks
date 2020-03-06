# AtacWorks

AtacWorks is a deep learning toolkit for coverage track denoising and peak calling from low-coverage or low-quality ATAC-Seq data.

![AtacWorks](data/readme/atacworks_slides.gif)

## Installation

### 1. Clone repository

#### Latest released version
This will clone the repo to the `master` branch, which contains code for latest released version
and hot-fixes.

```
git clone --recursive -b master https://github.com/clara-genomics/AtacWorks.git
```

#### Latest development version
This will clone the repo to the default branch, which is set to be the latest development branch.
This branch is subject to change frequently as features and bug fixes are pushed.

```bash
git clone --recursive https://github.com/clara-genomics/AtacWorks.git
```

### 2. System Setup

#### System requirements

* Ubuntu 16.04+
* CUDA 9.0+
* Python 3.6.7+
* GCC 5+
* (Optional) A conda or virtualenv setup
* Any NVIDIA GPU. AtacWorks training and inference currently does not run on CPU.

#### Install dependencies

* Download `bedGraphToBigWig` and `bigWigToBedGraph` binaries and add to your $PATH
    ```
    rsync -aP rsync://hgdownload.soe.ucsc.edu/genome/admin/exe/linux.x86_64/bedGraphToBigWig <custom_path>
    rsync -aP rsync://hgdownload.soe.ucsc.edu/genome/admin/exe/linux.x86_64/bigWigToBedGraph <custom_path>
    export PATH="$PATH:<custom_path>"
    sudo apt-get install hdf5-tools
    ```

* Install pip dependencies

    ```
    pip install -r requirements-base.txt && pip install -r requirements-macs2.txt
    ```

* Install atacworks

    ```
    pip install .
    ```
Note: The above non-standard installation is necessary to ensure the requirements for macs2 are installed
before macs2 itself.

### 3. Tests

Run unit tests:

    ```
    python -m pytest tests/
    ```

## Workflow

AtacWorks trains a deep neural network to learn a mapping between noisy (low coverage/low quality) ATAC-Seq data and matching clean (high coverage/high quality) ATAC-Seq data from the same cell type. Once this mapping is learned, the trained model can be applied to improve other noisy ATAC-Seq datasets. 

### 1. Training an AtacWorks model

#### Input files
To train an AtacWorks model, you need a pair of ATAC-Seq datasets from the same cell type, where one dataset has lower coverage or lower quality than the other. You can also use multiple such pairs of datasets. For each such pair of datasets, AtacWorks requires three input files:

1. A coverage track representing the number of sequencing reads mapped to each position on the genome in the low-coverage or low-quality dataset. This may be smoothed or processed. Format: [bigWig](https://genome.ucsc.edu/goldenPath/help/bigWig.html)

2. A coverage track representing the number of sequencing reads mapped to each position on the genome in the high-coverage or high-quality dataset. This may be smoothed or processed in the same way as the previous track. Format: [bigWig](https://genome.ucsc.edu/goldenPath/help/bigWig.html) 

3. The genomic positions of peaks called on the high-coverage or high-quality dataset. These can be obtained by using [MACS2](https://github.com/taoliu/MACS) or any other peak caller. Format: either [BED](http://genome.ucsc.edu/FAQ/FAQformat) or the narrowPeak format produced by MACS2.

The model learns a mapping from (1) to both (2) and (3); in other words, from the noisy coverage track, it learns to predict both the clean coverage track, and the positions of peaks in the clean dataset.

See [Tutorial 1](tutorials/tutorial1.md) for a workflow detailing the steps of data processing, encoding and model training and how to modify the parameters used in these steps.

### 2. Denoising and peak calling using a trained AtacWorks model

All models described in [Lal & Chiang, et al. (2019)](https://www.biorxiv.org/content/10.1101/829481) are available for download and use at `https://atacworks-paper.s3.us-east-2.amazonaws.com`. See below for instructions to use these models or your own trained models:

#### Input files

To denoise and call peaks from low-coverage/low-quality ATAC-seq data, you need three input files:

1. A trained AtacWorks model file with extension `.pth.tar`.

2. A coverage track representing the number of sequencing reads mapped to each position on the genome in the low-coverage or low-quality dataset. This may be smoothed or processed in the same way as the files used for training the model. Format: [bigWig](https://genome.ucsc.edu/goldenPath/help/bigWig.html)

3. Chromosome sizes file - a tab-separated text file containing the names and sizes of chromosomes in the genome.

#### One step denoising + peak calling command
```
bash Atacworks/scripts/run_inference.sh -bw <path to bigWig file with test ATAC-seq data> -m <path to model file> -f <path to chromosome sizes file> -o <output directory> -c <path to folder containing config files (optional)>
```
This command produces a folder containing several files:
1. <prefix>_infer_results.track.bw: A bigWig file containing the denoised ATAC-seq coverage track. 
2. infer_results_peaks.bed: A BED file containing the peaks called from the denoised ATAC-seq track. This file has 8 columns, in order: 
- chromosome
- peak start position
- peak end position
- peak length (bp)
- Mean coverage over peak
- Maximum coverage in peak
- Position of summit (relative to start)
- Position of summit (absolute). 
3. <prefix>_infer_results.peaks.bw: The same peak calls, in the form of a bigWig track for genome browser visualization.

`run_inference.sh` optionally takes a folder containing config files - specifically, this folder needs to contain two files, `infer_config.yaml` which specifies parameters for inference, and `model_structure.yaml` which specifies the structure of the deep learning model. If no folder containing config files is supplied, the folder `AtacWorks/configs` containing default parameter values will be used.

In order to vary output file names or formats, or inference parameters, you can change the arguments supplied in `infer_config.yaml`. Type `python AtacWorks/scripts/main.py infer --help` to understand which arguments to change.

In particular, the threshold for peak calling is controlled by the `infer_threshold` parameter in `infer_config.yaml`. By default, this is set to 0.5. If `infer_threshold` is set to "None" in the config file, `run_inference.sh` will instead produce a bigWig file in which each base is labeled with the probability (between 0 and 1) that it is part of a peak. 

#### Advanced usage: step-by-step denoising + peak calling with subcommands
See [Tutorial 2](tutorials/tutorial2.md) for an advanced workflow detailing the individual steps of data processing, encoding and prediction using a trained model, and how to modify the parameters used in these steps. 

## FAQ
1. What's the preferred way for setting up the environment?
    > A virtual environment or conda installation is preferred. You can follow conda installation instructions on their website and then follow the instructions in the README.

## Citation

Please cite AtacWorks as follows:

Lal, A., Chiang, Z.D., Yakovenko, N., Duarte, F.M., Israeli, J. and Buenrostro, J.D., 2019. AtacWorks: A deep convolutional neural network toolkit for epigenomics. BioRxiv, p.829481.

Link: https://www.biorxiv.org/content/10.1101/829481v1
