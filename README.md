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
    ```

* Install pip dependencies

    ```
    pip install -r requirements-base.txt && pip install -r requirements-macs2.txt
    ```

Note: The above non-standard installation is necessary to ensure the requirements for macs2 are installed
before macs2 itself.

### 3. Tests

Run unit tests:

    ```
    python -m pytest tests/
    ```

Run the following script to validate your setup.

    ```
    ./example/run.sh
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

#### Training command
```
TBA: end-to-end command
```
This command produces a directory (directory name) containing several trained models, of which the best model will be saved as `model_best.pth.tar`. 

#### Advanced usage: step-by-step training with subcommands
See [Tutorial 1](tutorials/tutorial1.md) for an advanced workflow detailing the individual steps of data processing, encoding and training and how to modify the parameters used in these steps.

### 2. Denoising and peak calling using a trained AtacWorks model

All models described in reference (1) are available for download and use. A list of these models and download instructions can be found here: TBA.

#### Input files

To denoise and call peaks from low-coverage/low-quality ATAC-seq data, you need only one input file:
1. A coverage track representing the number of sequencing reads mapped to each position on the genome in the low-coverage or low-quality dataset. This may be smoothed or processed in the same way as the files used for training the model. Format: [bigWig]

#### Denoising + peak calling command
```
TBA: end-to-end command
```
This command produces a directory (directory name) containing two files:
1. (Track file name).bw: A bigWig file containing the denoised ATAC-seq track
2. (Summarized peaks file name): A BED file containing the peaks called from the denoised ATAC-seq track. This BED file hs 8 columns. These are, in order: chromosome, peak start position, peak end position, peak length (bp), Mean coverage over peak, Maximum coverage in peak, Position of summit (relative to start), and Position of summit (absolute).

#### Advanced usage: step-by-step denoising + peak calling using a trained AtacWorks model with subcommands
See [Tutorial 2](tutorials/tutorial2.md) for an advanced workflow detailing the individual steps of data processing, encoding and prediction using a trained model, and how to modify the parameters used in these steps.

## FAQ
1. What's the preferred way for setting up the environment ?
    > A virtual environment or conda installation is preferred. You can follow conda installation instructions on their website and then follow the instructions in the README.

## Citation

Lal, A., Chiang, Z.D., Yakovenko, N., Duarte, F.M., Israeli, J. and Buenrostro, J.D., 2019. AtacWorks: A deep convolutional neural network toolkit for epigenomics. BioRxiv, p.829481.
https://www.biorxiv.org/content/10.1101/829481
