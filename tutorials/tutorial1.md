# Tutorial 1: Training an AtacWorks model 

## Introduction

In this tutorial we train an AtacWorks model to denoise the signal track and call peaks from aggregate single-cell ATAC-seq data derived from a small number of cells. We use the dsc-ATAC-seq dataset presented in (1), section (refer to page number, section). This dataset consists of single-cell ATAC-seq data from several types of human blood cells.

Note that all the AtacWorks models described in reference (1) are available to download (link) and you may be able to use one of these instead of training a new model. To learn how to download and use an existing model, refer to Tutorial 2 (link).
 
We selected 2400 Monocytes from this dataset - this is our ‘clean’, high-coverage dataset. We then randomly sampled 50 of these 2400 Monocytes. Here's what the ATAC-seq signal from 50 cells and 2400 cells looks like, for a region on chromosome 10:

<insert picture>

Compared to the 'clean' signal from 2400 cells, the aggregated ATAC-Seq signal track from these 50 cells is noisy. The Pearson correlation between the 50-cell signal and the 2400-cell signal is only <insert correlation> on chromosome 10. Because the signal is noisy, peak calls calculated by MACS2 on this data are also inaccurate; the AUPRC of peak calling from the noisy data is only <insert AUPRC> on chromosome 10.

We train an AtacWorks model to learn a mapping from the 50-cell ATAC-seq signals to the 2400-cell ATAC-seq signal and peak calls. In other words, given a noisy ATAC-seq signal from 50 cells, this model learns what the signal would look like - and where the peaks would be called - if we had sequenced 2400 cells.

## Step 1: Set parameters

Replace 'path_to_atacworks' with the path to your cloned 'AtacWorks' github repository.
```
atacworks=<path_to_atacworks>
```

## Step 2: Download data

We will download from AWS all of the data needed for this experiment. (Note: the S3 bucket is not yet public. We can use wget for download once it is.)

### Noisy ATAC-seq signal from 50 Monocytes
```
aws s3 cp s3://atacworks-paper/dsc_atac_blood_cell_denoising_experiments/train_data/noisy_data/dsc.1.Mono.50.cutsites.smoothed.200.bw ./
 ```
### Clean ATAC-seq signal from 2400 Monocytes
```
aws s3 cp s3://atacworks-paper/dsc_atac_blood_cell_denoising_experiments/train_data/clean_data/dsc.Mono.2400.cutsites.smoothed.200.bw ./
 ```
### Clean ATAC-seq peaks from 2400 Monocytes
```
aws s3 cp s3://atacworks-paper/dsc_atac_blood_cell_denoising_experiments/train_data/clean_data/dsc.Mono.2400.cutsites.smoothed.200.3.narrowPeak ./
```
### Genomic intervals to define regions for training and validation

We need to define which regions of the genome will be used to train and test the model. We want to train models on some portion of the genome ('training set') and evaluate their performance on a separate portion ('validation set'). We will choose the model that performs best on the validation set as the best model. Later, we will evaluate the performance of this best model on a third portion of the genome ('holdout set').

Here, we use chromosome 20 as the validation set, chromosome 10 as the holdout set, and the remaining autosomes as the training set. Since a whole chromosome is too long to feed into the model at once, we split each of these chromosomes into 50,000-base long intervals. 

We will download the intervals for the training set.
```
aws s3 cp s3://atacworks-paper/dsc_atac_blood_cell_denoising_experiments/intervals/hg19.50000.training_intervals.bed ./intervals/hg19.50000.training_intervals.bed
```
We can look at these intervals:

```
# head intervals/hg19.50000.training_intervals.bed 
chr1  0 50000
chr1  50000 100000
chr1  100000  150000
chr1  150000  200000
chr1  200000  250000
chr1  250000  300000
chr1  300000  350000
chr1  350000  400000
chr1  400000  450000
chr1  450000  500000
```
Next, we download the intervals for the validation set:
```
aws s3 cp s3://atacworks-paper/dsc_atac_blood_cell_denoising_experiments/intervals/hg19.50000.val_intervals.bed ./intervals/hg19.50000.val_intervals.bed
```

### Config files
We also need to download the 'configs' directory containing config files for this experiment. The config files describe the parameters of the experiment, including the structure of the deep learning model.
```
aws s3 cp s3://atacworks-paper/dsc_atac_blood_cell_denoising_experiments/configs/ ./configs --recursive
```

## Step 3: Convert clean peak file into bigWig format

The clean peak calls (`dsc.Mono.2400.cutsites.smoothed.200.3.narrowPeak`) were produced by MACS2 and are in .narrowPeak format. We need to convert them to bigWig format for use. This also requires us to supply a chromosome sizes file describing the reference genome that we use. 

Chromosome sizes files for the hg19 and hg38 human reference genomes are supplied with AtacWorks in the folder `AtacWorks/example/reference`. Here, we are using hg19.

```
python $atacworks/peak2bw.py dsc.Mono.2400.cutsites.smoothed.200.3.narrowPeak $atacworks/example/reference/hg19.chrom.sizes --skip 1
```

The `--skip 1` argument tells the script to ignore the first line of the narrowPeak file as it contains a header.

This command reads the peak positions from the .narrowPeak file and writes them to a bigWig file named `dsc.Mono.2400.cutsites.smoothed.200.3.narrowPeak.bw`.

```
INFO:2020-01-22 20:32:05,270:AtacWorks-peak2bw] Reading input files
INFO:2020-01-22 20:32:05,387:AtacWorks-peak2bw] Retaining 105959 of 105959 peaks in given chromosomes.
INFO:2020-01-22 20:32:05,387:AtacWorks-peak2bw] Adding score
INFO:2020-01-22 20:32:05,388:AtacWorks-peak2bw] Writing peaks to bedGraph file
INFO:2020-01-22 20:32:05,855:AtacWorks-peak2bw] Writing peaks to bigWig file dsc.Mono.2400.cutsites.smoothed.200.3.narrowPeak.bw
INFO:2020-01-22 20:32:06,042:AtacWorks-peak2bw] Done!
```

## Step 4: Read the training data and labels and save in .h5 format

We take the three bigWig files containing noisy ATAC-seq signal, the clean ATAC-seq signal, and the clean ATAC-seq peak calls. For these three files, we read the values in the specified intervals, and save these values in a format that can be read by our model. First, we read values for the intervals in the training set (`hg19.50000.training_intervals.bed`), spanning all autosomes except chr10 and chr20.

```
python $atacworks/bw2h5.py \
           --noisybw dsc.1.Mono.50.cutsites.smoothed.200.bw \
           --cleanbw dsc.Mono.2400.cutsites.smoothed.200.bw \
           --cleanpeakbw dsc.Mono.2400.cutsites.smoothed.200.3.narrowPeak.bw \
           --intervals intervals/hg19.50000.training_intervals.bed \
           --prefix Mono.50.2400.train \
           --pad 5000 \
           --batch_size 2000 \
           --nonzero
```
This produces a .h5 file (`Mono.50.2400.train.h5`) containing the training data for the model.

## Step 5: Read the validation data and labels and save in .h5 format

Next we read and save the validation data for the model.

```
python $atacworks/bw2h5.py \
           --noisybw dsc.1.Mono.50.cutsites.smoothed.200.bw \
           --cleanbw dsc.Mono.2400.cutsites.smoothed.200.bw \
           --cleanpeakbw dsc.Mono.2400.cutsites.smoothed.200.3.narrowPeak.bw \
           --intervals intervals/hg19.50000.val_intervals.bed \
           --prefix Mono.50.2400.val \
           --pad 5000 \
           --batch_size 2000
```
This produces a .h5 file (`Mono.50.2400.val.h5`) containing the validation data for the model.


## Step 6: Train and validate a model using the parameters in the given config files

We next train an AtacWorks model to learn a mapping from the noisy (50-cell) ATAC-seq signal to the clean (2400-cell) ATAC-seq signal and peak calls. The two .yaml files that we downloaded into the `configs` directory contain all the parameters that describe how to train the model. `configs/model_structure.yaml` contains parameters that control the architecture of the model and  `configs/config_params.yaml` contains parameters that control the process of training, such as the learning rate and batch size.

To train the model, we supply the training and validation datasets as well as the two config files.

```
python $atacworks/main.py --train \
        --config configs/config_params.yaml \
        --config_mparams configs/model_structure.yaml \
        --train_files Mono.50.2400.train.h5 \
        --val_files Mono.50.2400.val.h5 \
        --epochs 5
```
This command trains a deep learning model using the supplied clean and noisy ATAC-seq data, for 5 epochs (5 full passes through the dataset). At the end of every epoch, the current state of the model is saved in the directory `output_latest`, and the performance of the current model is measured on the validation set. At the end, out of the 5 saved models, the one with the best performance on the validation set is saved as `output_latest/model_best.pth.tar`

This model has learned a mapping from the 50-cell signal to the 2400-cell signal and peak calls. Given a new 50-cell ATAC-seq track, it can denoise the track and produce high-quality peak calls.

See Tutorial 2 for step-by-step instructions on how to apply this trained model to another dataset and evaluate its performance.

## References
(1) Lal, A., Chiang, Z.D., Yakovenko, N., Duarte, F.M., Israeli, J. and Buenrostro, J.D., 2019. AtacWorks: A deep convolutional neural network toolkit for epigenomics. BioRxiv, p.829481. (https://www.biorxiv.org/content/10.1101/829481v1)

## Appendix 1: Customize the training command using config files

To change any of the parameters for the deep learning model, you can edit the parameters in `configs/config_params.yaml` or `configs/model_structure.yaml` and run the command in step 6 above. See the documentation in these files for an explanation of the parameters. <Note - we must include enough documentation in these files>

## Appendix 2: Reproducing the model reported in the AtacWorks preprint (1)

In the paper (1), we report this experiment, with two differences:

1. The model is trained on a larger dataset.
2. The model is trained for 25 epochs instead of 5.

To download the exact model used in the paper, see Tutorial 2.

In order to train the same model reported in the paper, follow the following steps. 
- Download all the training and validation data
```
aws s3 cp s3://atacworks-paper/dsc_atac_blood_cell_denoising_experiments/train_data/noisy_data/ ./train_data/noisy_data/ --recursive
aws s3 cp s3://atacworks-paper/dsc_atac_blood_cell_denoising_experiments/train_data/clean_data/ ./train_data/clean_data/ --recursive
```
- Encode all the training data and save in the `train_h5` directory.
```
mkdir train_h5
cell_types = (CD19 Mono)
samples=(1 2 3 4 5)
for cell_type in ${cell_types[*]}; do
    for sample in ${samples[*]}; do
        python $atacworks/bw2h5.py \
           --noisybw train_data/noisy_data/dsc.$sample.${cell_type}.50.cutsites.smoothed.200.bw \
           --cleanbw train_data/clean_data/dsc.${cell_type}.2400.cutsites.smoothed.200.bw \
           --cleanpeakbw train_data/clean_data/dsc.${cell_type}.2400.cutsites.smoothed.200.3.narrowPeak.bw \
           --intervals intervals/hg19.50000.training_intervals.bed \
           --prefix train_h5/${cell_type}.$sample.50.2400.train \
           --pad 5000 \
           --batch_size 2000 \
           --nonzero
    done
done
```
- Encode all the validation data and save in the `val_h5` directory.
```
mkdir val_h5
cell_types = (CD19 Mono)
for cell_type in ${cell_types[*]}; do
    python $atacworks/bw2h5.py \
           --noisybw train_data/noisy_data/dsc.1.${cell_type}.50.cutsites.smoothed.200.bw \
           --cleanbw train_data/clean_data/dsc.${cell_type}.2400.cutsites.smoothed.200.bw \
           --cleanpeakbw train_data/clean_data/dsc.${cell_type}.2400.cutsites.smoothed.200.3.narrowPeak.bw \
           --intervals intervals/hg19.50000.val_intervals.bed \
           --prefix val_h5/${cell_type}.50.2400.val \
           --pad 5000 \
           --batch_size 2000
done
```
- Train using all of the training and validation data, for 25 epochs. Here, we supply the directories `train_h5` and `val_h5`, and the model uses all the files within these directories for training and validation respectively.
```
python $atacworks/main.py --train \
        --config configs/config_params.yaml \
        --config_mparams configs/model_structure.yaml \
        --train_files train_h5 \
        --val_files val_h5 \
        --epochs 5
```
