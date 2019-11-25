# AtacWorks

AtacWorks is a deep learning toolkit for track denoising and peak calling from low-coverage or low-quality ATAC-Seq data.

<iframe src="https://giphy.com/embed/U3DqwNYL0ehzwnTMvH" width="480" height="270" frameBorder="0" class="giphy-embed" allowFullScreen></iframe><p><a href="https://giphy.com/gifs/U3DqwNYL0ehzwnTMvH">via GIPHY</a></p>

AtacWorks trains a deep neural network to learn a mapping between noisy (low coverage/low quality) ATAC-Seq data and matching clean (high coverage/high quality) ATAC-Seq data from the same cell type. Once this mapping is learned, the trained model can be applied to improve other noisy ATAC-Seq datasets. 

AtacWorks models can be trained using one or more pairs of matching ATAC-Seq datasets from the same cell type. AtacWorks requires three specific inputs for each such pair of datasets:
1. A coverage track representing the number of sequencing reads mapped to each position on the genome in the low-quality dataset.
2. A coverage track representing the number of sequencing reads mapped to each position on the genome in the high-quality dataset. 
3. The genomic positions of peaks called on the high-quality dataset. These can be obtained by using MACS2 or any other peak caller.
The model learns a mapping from (1) to both (2) and (3); in other words, from the noisy coverage track, it learns to predict both the clean coverage track, and the positions of peaks in the clean dataset. We also provide pretrained models that can be applied to a noisy dataset.

Much more information and examples can be found in the AtacWorks preprint: https://www.biorxiv.org/content/10.1101/829481

## Runtime

Training: Approximately 22 minutes per epoch to train on single whole genome.

Inference: Approximately 28 minutes for inference and postprocessing on a whole genome.

Training and inference were performed on a single Tesla V100 GPU. Training time can be significantly reduced by using multiple GPUs.

We are working to improve runtime, particularly for inference. Improvements are tracked on our project board: https://github.com/clara-genomics/AtacWorks/projects 

## System Setup

0. System requirements

* Ubuntu 16.04+
* CUDA 9.0+
* Python 3.6.7+
* GCC 5+
* (Optional) A conda or virtualenv setup
* Any NVIDIA GPU. AtacWorks training and inference currently does not run on CPU.

1. Download `bedGraphToBigWig` and `bigWigToBedGraph` binaries and add to your $PATH
    ```
    rsync -aP rsync://hgdownload.soe.ucsc.edu/genome/admin/exe/linux.x86_64/bedGraphToBigWig <custom_path>
    rsync -aP rsync://hgdownload.soe.ucsc.edu/genome/admin/exe/linux.x86_64/bigWigToBedGraph <custom_path>
    export PATH="$PATH:<custom_path>"
    ```

2. Install pip dependencies

    ```
    pip install -r requirements-pip.txt
    ```

3. Unit tests

    ```
    python -m pytest tests/
    ```

## Workflow

1. Convert the MACS2 peak calls on the clean data to bigWig format with `peak2bw.py`
2. Generate genomic intervals for training/validation/holdout with `get_intervals.py`
3. Encode the training/validation/holdout data into .h5 format with `bw2h5.py`
4. Train a model with `main.py`
5. Apply the trained model for inference on another dataset with `main.py`
6. Convert the predictions into bigWig format with `bedGraphToBigWig`

### Workflow input

Training:
1. bigWig file for clean ATAC-Seq
2. bigWig file for noisy ATAC-Seq
3. MACS2 output for clean ATAC-Seq (.narrowPeak or .bed file)

Testing:
1. bigWig file for noisy ATAC-Seq

### Workflow Example

1. Run the following script to validate your setup.

    ```
    ./example/run.sh
    ```

### Pretrained models

3 pretrained models are provided in `data/pretrained_models/bulk_blood_data/`.
These are based on bulk ATAC-Seq data from 7 blood cell types. They are trained using clean data of depth 80 million reads, subsampled to a depth of 1 million (1000000.7cell.resnet.5.2.15.8.50.0803.pth.tar), 2 million (2000000.7cell.resnet.5.2.15.8.50.0803.pth.tar), or 5 million (5000000.7cell.resnet.5.2.15.8.50.0803.pth.tar) reads.

### FAQ
1. What's the preferred way for setting up the environment ?
A. A virtual environment or conda installation is preferred. You can follow conda installation instructions on their website and then follow the instructions in the README.

2. If you face "no module named numpy" error while installing requirement-pip.txt.
A. In your terminal, run pip install numpy==<version-from-requirements-pip.txt> and then run pip install -r requirements-pip.txt. If you are running inside a conda or venv, run these commands inside your environment.

### Citation

Lal, A., Chiang, Z.D., Yakovenko, N., Duarte, F.M., Israeli, J. and Buenrostro, J.D., 2019. AtacWorks: A deep convolutional neural network toolkit for epigenomics. BioRxiv, p.829481.