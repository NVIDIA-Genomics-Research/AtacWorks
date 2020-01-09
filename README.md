# AtacWorks

AtacWorks is a deep learning toolkit for track denoising and peak calling from low-coverage or low-quality ATAC-Seq data.

![AtacWorks](data/readme/atacworks_slides.gif)


## System Setup

### System requirements

* Ubuntu 16.04+
* CUDA 9.0+
* Python 3.6.7+
* GCC 5+
* (Optional) A conda or virtualenv setup
* Any NVIDIA GPU. AtacWorks training and inference currently does not run on CPU.

### Install dependencies

* Download `bedGraphToBigWig` and `bigWigToBedGraph` binaries and add to your $PATH
    ```
    rsync -aP rsync://hgdownload.soe.ucsc.edu/genome/admin/exe/linux.x86_64/bedGraphToBigWig <custom_path>
    rsync -aP rsync://hgdownload.soe.ucsc.edu/genome/admin/exe/linux.x86_64/bigWigToBedGraph <custom_path>
    export PATH="$PATH:<custom_path>"
    ```

* Install pip dependencies

    ```
    pip install -r requirements-pip.txt
    ```
## Sample installation test
* Run this command to ensure your environment is setup correctly.

## Environment setup and pre-processing of data
* Setup environment variables
    ```
    example_dir=$(readlink -f $(dirname "$0"));
    data_dir="$example_dir/data";
    ref_dir="$example_dir/reference";
    out_dir="$example_dir/result";
    root_dir=$(readlink -f "$example_dir/..");
    saved_model_dir="$root_dir/data/pretrained_models";
    ```

* Convert peak files into bigwig format
    ```
    # Clean peaks
    python $root_dir/peak2bw.py \
        $data_dir/HSC.80M.chr123.10mb.peaks.bed \
        $ref_dir/hg19.auto.sizes \
        --prefix=$out_dir/HSC.80M.chr123.10mb.peaks.bed
    ```
    ```
    # Noisy peaks
    python $root_dir/peak2bw.py \
        $data_dir/HSC.5M.chr123.10mb.peaks.bed \
        $ref_dir/hg19.auto.sizes \
        --prefix=$out_dir/HSC.5M.chr123.10mb.peaks.bed
    ```

* Split the dataset into train, val and holdout/test intervals.
    ```
    python $root_dir/get_intervals.py \
        $data_dir/example.sizes 24000 $out_dir/example \
        --val chr2 --holdout chr3
    ```

* Save the data in H5.
    ```
    # Training data
    python $root_dir/bw2h5.py \
        --noisybw $data_dir/HSC.5M.chr123.10mb.coverage.bw \
        --intervals $out_dir/example.training_intervals.bed \
        --batch_size 4 \
        --prefix $out_dir/train_data \
        --cleanbw $data_dir/HSC.80M.chr123.10mb.coverage.bw \
        --cleanpeakbw $out_dir/HSC.80M.chr123.10mb.peaks.bed.bw \
        --nonzero
    # Validation data
    python $root_dir/bw2h5.py \
        --noisybw $data_dir/HSC.5M.chr123.10mb.coverage.bw \
        --intervals $out_dir/example.val_intervals.bed \
        --batch_size 64 \
        --prefix $out_dir/val_data \
        --cleanbw $data_dir/HSC.80M.chr123.10mb.coverage.bw \
        --cleanpeakbw $out_dir/HSC.80M.chr123.10mb.peaks.bed.bw
    # Test data
    python $root_dir/bw2h5.py \
        --noisybw $data_dir/HSC.5M.chr123.10mb.coverage.bw \
        --intervals $out_dir/example.holdout_intervals.bed \
        --batch_size 64 \
        --prefix $out_dir/test_data \
        --cleanbw $data_dir/HSC.80M.chr123.10mb.coverage.bw \
        --cleanpeakbw $out_dir/HSC.80M.chr123.10mb.peaks.bed.bw
    #No label
    python $root_dir/bw2h5.py \
        --noisybw $data_dir/HSC.5M.chr123.10mb.coverage.bw \
        --intervals $out_dir/example.holdout_intervals.bed \
        --batch_size 64 \
        --prefix $out_dir/no_label \
        --nolabel
    ```

## Tutorial 1 - Inference and metrics using pretrained model
* Follow steps under [pre-processing](#Environment-setup-and-pre-processing-of-data)
* Run inference
    ```
    python $root_dir/main.py --infer \
        --infer_files $out_dir/no_label.h5 \
        --intervals_file $out_dir/example.holdout_intervals.bed \
        --sizes_file $ref_dir/hg19.auto.sizes \
        --weights_path $saved_model_dir/bulk_blood_data/5000000.7cell.resnet.5.2.15.8.50.0803.pth.tar \
        --out_home $out_dir --label inference.pretrained \
        --result_fname HSC.5M.output.pretrained --reg_rounding 0 --cla_rounding 3 \
        --model resnet --nblocks 5 --nfilt 15 --width 50 --dil 8 \
        --nblocks_cla 2 --nfilt_cla 15 --width_cla 50 --dil_cla 10 \
        --task both --num_workers 0 --gen_bigwig
    ```
* Calculate regression metrics
    ```
    python $root_dir/calculate_baseline_metrics.py \
        --label_file $out_dir/test_data.h5 --task regression \
        --test_file $out_dir/inference.pretrained_latest/no_label_HSC.5M.output.pretrained.track.bw \
        --intervals $out_dir/example.holdout_intervals.bed \
        --sizes $ref_dir/hg19.auto.sizes \
        --sep_peaks
    ```
* Calculate classification metrics
    ```
    python $root_dir/calculate_baseline_metrics.py \
        --label_file $out_dir/test_data.h5 --task classification \
        --test_file $out_dir/inference.pretrained_latest/no_label_HSC.5M.output.pretrained.peaks.bw \
        --intervals $out_dir/example.holdout_intervals.bed \
        --sizes $ref_dir/hg19.auto.sizes \
        --thresholds 0.5
    ```

## Tutorial 2 - Train a model
* Follow steps under [pre-processing](#Environment-setup-and-pre-processing-of-data)
* Train and validate
    ```
    python $root_dir/main.py --train \
        --train_files $out_dir/train_data.h5 \
        --val_files $out_dir/val_data.h5 \
        --out_home $out_dir --label HSC.5M.model \
        --checkpoint_fname checkpoint.pth.tar \
        --distributed
    ```
## Runtime

Training: Approximately 22 minutes per epoch to train on single whole genome.

Inference: Approximately 28 minutes for inference and postprocessing on a whole genome.

Training and inference were performed on a single Tesla V100 GPU. Training time can be significantly reduced by using multiple GPUs.

We are working to improve runtime, particularly for inference. Improvements are tracked on our project board: https://github.com/clara-genomics/AtacWorks/projects 

## Clone repository

### Latest released version
This will clone the repo to the `master` branch, which contains code for latest released version
and hot-fixes.

```
git clone --recursive -b master https://github.com/clara-genomics/AtacWorks.git
```

### Latest development version
This will clone the repo to the default branch, which is set to be the latest development branch.
This branch is subject to change frequently as features and bug fixes are pushed.

```bash
git clone --recursive https://github.com/clara-genomics/AtacWorks.git
```


### Unit tests

    ```
    python -m pytest tests/
    ```

## Workflow

1. Convert peak calls on the clean data to bigWig format with `peak2bw.py`
2. Generate genomic intervals for training/validation/holdout with `get_intervals.py`
3. Encode the training/validation/holdout data into .h5 format with `bw2h5.py`
4. Train a model with `main.py`
5. Apply the trained model for inference on another dataset with `main.py`, producing output in bigWig or bedGraph format.

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
