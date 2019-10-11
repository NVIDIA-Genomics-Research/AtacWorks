# AtacWorks

## System Setup

0. System requirements

* Ubuntu 16.04+
* CUDA 9.0+
* Python 3.6.7+
* (Optional) A conda or virtualenv setup

1. Download `bedGraphToBigWig` binary and add to your $PATH
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

1. Convert MACS2 output to bigWig with `peak2bw.py`
2. Generate training/val/holdout intervals with `get_intervals.py`
3. Save training/val/holdout data with `bw2h5.py`
4. Train and validate a model with `main.py`
5. Convert the predictions into bigWig format with `postprocess.py`

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
