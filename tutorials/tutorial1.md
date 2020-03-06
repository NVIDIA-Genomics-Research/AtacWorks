# Tutorial 1: Training an AtacWorks model 

## Introduction

In this tutorial we train an AtacWorks model to denoise the signal track and call peaks from aggregate single-cell ATAC-seq data derived from a small number of cells. We use the dsc-ATAC-seq dataset presented in reference (1) (Section "AtacWorks enhances ATAC-seq results from small numbers of single cells", also Supplementary Table 8). This dataset consists of single-cell ATAC-seq data from several types of human blood cells.

Note that all the AtacWorks models described in reference (1) are available to download (https://atacworks-paper.s3.us-east-2.amazonaws.com) and you may be able to use one of these instead of training a new model. To learn how to download and use an existing model, refer to [Tutorial 2](tutorial2.md).
 
We selected 2400 Monocytes from this dataset - this is our ‘clean’, high-coverage dataset. We then randomly sampled 50 of these 2400 Monocytes. Here's what the ATAC-seq signal from 50 cells and 2400 cells looks like, for a region on chromosome 10:

![Monocytes subsampled signal](Mono.2400.50.png)

Compared to the 'clean' signal from 2400 cells, the aggregated ATAC-Seq signal track from these 50 cells is noisy. Because of noise in the signal, peak calls calculated by MACS2 on this data are also inaccurate.

We train an AtacWorks model to learn a mapping from the 50-cell ATAC-seq signals to the 2400-cell ATAC-seq signal and peak calls. In other words, given a noisy ATAC-seq signal from 50 cells, this model learns what the signal would look like - and where the peaks would be called - if we had sequenced 2400 cells.

## Step 1: Set parameters

Replace 'path_to_atacworks' with the path to your cloned and set up 'AtacWorks' github repository.
```
atacworks=<path_to_atacworks>
```

## Step 2: Download data

We will download all of the data needed for this experiment from AWS.

### Noisy ATAC-seq signal from 50 Monocytes
```
wget https://atacworks-paper.s3.us-east-2.amazonaws.com/dsc_atac_blood_cell_denoising_experiments/50_cells/train_data/noisy_data/dsc.1.Mono.50.cutsites.smoothed.200.bw
 ```
### Clean ATAC-seq signal from 2400 Monocytes
```
wget https://atacworks-paper.s3.us-east-2.amazonaws.com/dsc_atac_blood_cell_denoising_experiments/50_cells/train_data/clean_data/dsc.Mono.2400.cutsites.smoothed.200.bw
 ```
### Clean ATAC-seq peaks from 2400 Monocytes
```
wget https://atacworks-paper.s3.us-east-2.amazonaws.com/dsc_atac_blood_cell_denoising_experiments/50_cells/train_data/clean_data/dsc.Mono.2400.cutsites.smoothed.200.3.narrowPeak
```

### Config files
We also need to download config files for this experiment. The config files describe the structure of the deep learning model and the parameters to train it. We will place these in the `configs` folder. 
```
mkdir configs
wget -P configs https://atacworks-paper.s3.us-east-2.amazonaws.com/dsc_atac_blood_cell_denoising_experiments/50_cells/configs/train_config.yaml
wget -P configs https://atacworks-paper.s3.us-east-2.amazonaws.com/dsc_atac_blood_cell_denoising_experiments/50_cells/configs/model_structure.yaml
```

## Step 3: Convert clean peak file into bigWig format

The clean peak calls (`dsc.Mono.2400.cutsites.smoothed.200.3.narrowPeak`) were produced by MACS2 and are in .narrowPeak format. We need to convert them to bigWig format for use. This also requires us to supply a chromosome sizes file describing the reference genome that we use. 

Chromosome sizes files for the hg19 and hg38 human reference genomes are supplied with AtacWorks in the folder `AtacWorks/example/reference`. Here, we are using hg19.

```
python $atacworks/scripts/peak2bw.py \
    --input dsc.Mono.2400.cutsites.smoothed.200.3.narrowPeak \
    --sizes $atacworks/example/reference/hg19.chrom.sizes \
    --out_dir ./ \
    --skip 1
```

The `--skip 1` argument tells the script to ignore the first line of the narrowPeak file as it contains a header.

This command reads the peak positions from the .narrowPeak file and writes them to a bigWig file in the current directory,  named `dsc.Mono.2400.cutsites.smoothed.200.3.narrowPeak.bw`.

```
INFO:2020-01-22 20:32:05,270:AtacWorks-peak2bw] Reading input files
INFO:2020-01-22 20:32:05,387:AtacWorks-peak2bw] Retaining 105959 of 105959 peaks in given chromosomes.
INFO:2020-01-22 20:32:05,387:AtacWorks-peak2bw] Adding score
INFO:2020-01-22 20:32:05,388:AtacWorks-peak2bw] Writing peaks to bedGraph file
INFO:2020-01-22 20:32:05,855:AtacWorks-peak2bw] Writing peaks to bigWig file dsc.Mono.2400.cutsites.smoothed.200.3.narrowPeak.bw
INFO:2020-01-22 20:32:06,042:AtacWorks-peak2bw] Done!
```

For more information type `python $atacworks/scripts/peak2bw.py --help`

## Step 4: Create genomic intervals to define regions for training and validation

We need to define which regions of the genome will be used to train and test the model. We want to train models on some portion of the genome ('training set') and evaluate their performance on a separate portion ('validation set'). We will choose the model that performs best on the validation set as the best model. Later, we will evaluate the performance of this best model on a third portion of the genome ('holdout set').

We provide a chromosome sizes file 'hg19.auto.sizes' that contains sizes for all the autosomes of the hg19 reference genome. We split off chromosome 20 to use as the validation set, and chromosome 10 to use as the holdout set, and use the remaining autosomes as the training set. Since a whole chromosome is too long to feed into the model at once, we split each of these chromosomes into 50,000-bp long intervals.

```
python $atacworks/scripts/get_intervals.py \
     --sizes $atacworks/example/reference/hg19.auto.sizes \
     --intervalsize 50000 \
     --out_dir ./ \
     --val chr20 \
     --holdout chr10
```
This command generates three BED files in the current directory: `training_intervals.bed`, `val_intervals.bed`, and `holdout_intervals.bed`. These BED files contain 50,000-bp long intervals spanning the given chromosomes. We can look at these intervals:

```
# head training_intervals.bed 
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
For more information type `python $atacworks/scripts/get_intervals.py --help`

## Step 5: Read the training data and labels and save in .h5 format

We take the three bigWig files containing noisy ATAC-seq signal, the clean ATAC-seq signal, and the clean ATAC-seq peak calls. For these three files, we read the values in the regions defined by tge training intervals, and save these values in a format that can be read by our model. First, we read values for the intervals in the training set (`training_intervals.bed`), spanning all autosomes except chr10 and chr20.

```
python $atacworks/scripts/bw2h5.py \
           --noisybw dsc.1.Mono.50.cutsites.smoothed.200.bw \
           --cleanbw dsc.Mono.2400.cutsites.smoothed.200.bw \
           --cleanpeakbw dsc.Mono.2400.cutsites.smoothed.200.3.narrowPeak.bw \
           --intervals training_intervals.bed \
           --out_dir ./ \
           --prefix Mono.50.2400.train \
           --pad 5000 \
           --nonzero
```
This produces a .h5 file in the current directory (`Mono.50.2400.train.h5`) containing the training data for the model. The `--nonzero` flag ignores intervals that contain zero coverage. We use this flag for training data as these intervals do not help the model to learn.

For more information type `python $atacworks/scripts/bw2h5.py --help`


## Step 6: Read the validation data and labels and save in .h5 format

Next we read and save the validation data for the model.

```
python $atacworks/scripts/bw2h5.py \
           --noisybw dsc.1.Mono.50.cutsites.smoothed.200.bw \
           --cleanbw dsc.Mono.2400.cutsites.smoothed.200.bw \
           --cleanpeakbw dsc.Mono.2400.cutsites.smoothed.200.3.narrowPeak.bw \
           --intervals val_intervals.bed \
           --out_dir ./ \
           --prefix Mono.50.2400.val \
           --pad 5000
```
This produces a .h5 file in the current directory (`Mono.50.2400.val.h5`) containing the validation data for the model.


## Step 7: Train and validate a model using the parameters in the given config files

We next train an AtacWorks model to learn a mapping from the noisy (50-cell) ATAC-seq signal to the clean (2400-cell) ATAC-seq signal and peak calls. The two .yaml files that we downloaded into the `configs` directory contain all the parameters that describe how to train the model. `configs/model_structure.yaml` contains parameters that control the architecture of the model and  `configs/config_params.yaml` contains parameters that control the process of training, such as the learning rate and batch size.

To train the model, we supply the training and validation datasets as well as the two config files.

```
python $atacworks/scripts/main.py train \
        --config configs/train_config.yaml \
        --config_mparams configs/model_structure.yaml \
        --train_files Mono.50.2400.train.h5 \
        --val_files Mono.50.2400.val.h5
```
This command trains a deep learning model using the supplied clean and noisy ATAC-seq data, for 5 epochs (5 full passes through the dataset). At the end of every epoch, the current state of the model is saved in the directory `output_latest`, and the performance of the current model is measured on the validation set. At the end, out of the 5 saved models, the one with the best performance on the validation set is saved as `output_latest/model_best.pth.tar`

This model has learned a mapping from the 50-cell signal to the 2400-cell signal and peak calls. Given a new 50-cell ATAC-seq track, it can denoise the track and produce high-quality peak calls.

See [Tutorial 2](tutorial2.md) for step-by-step instructions on how to apply this trained model to another dataset.

To change any of the parameters for the deep learning model, you can edit the appropriate parameters in `configs/train_config.yaml` or `configs/model_structure.yaml` and run the command in step 7 above. Type `python $atacworks/scripts/main.py train --help` for an explanation of the parameters.

## References
(1) Lal, A., Chiang, Z.D., Yakovenko, N., Duarte, F.M., Israeli, J. and Buenrostro, J.D., 2019. AtacWorks: A deep convolutional neural network toolkit for epigenomics. BioRxiv, p.829481. (https://www.biorxiv.org/content/10.1101/829481v1)

## Appendix 1: Training on multiple pairs of clean and noisy datasets

If using multiple pairs of clean and noisy datasets for training, use steps 5 and 6 on each pair to create a training h5 file and a validation h5 file for each pair. Save all of the training h5 files into a single folder and all of the validation h5 files into another folder.

Run step 7 as follows:
```
python $atacworks/scripts/main.py train \
        --config configs/train_config.yaml \
        --config_mparams configs/model_structure.yaml \
        --train_files <path to folder containing all h5 files for training> \
        --val_files <path to folder containing all h5 files for validation>
```
See Appendix 2 below for an example.

## Appendix 2: Reproducing the model reported in the AtacWorks preprint (Reference 1)

In Section "AtacWorks enhances ATAC-seq results from small numbers of single cells" (also Supplementary Table 8), we report this experiment, although the model we use there is trained on more data.

To download the exact model used in the paper, see [Tutorial 2](tutorial2.md).

In order to train the same model reported in the paper, follow the following steps. 
- Download all the training data
```
mkdir train_data/noisy_data
cell_types=(CD19 Mono)
subsamples=(1 2 3 4 5)
for cell_type in ${cell_types[*]}; do
    for subsample in ${subsamples[*]}; do
        wget -P train_data/noisy_data https://atacworks-paper.s3.us-east-2.amazonaws.com/dsc_atac_blood_cell_denoising_experiments/50_cells/train_data/noisy_data/dsc.$subsample.$cell_type.50.cutsites.smoothed.200.bw
    done
done

mkdir train_data/clean_data

cell_types=(CD19 Mono)
for cell_type in ${cell_types[*]}; do
    wget -P train_data/clean_data https://atacworks-paper.s3.us-east-2.amazonaws.com/dsc_atac_blood_cell_denoising_experiments/50_cells/train_data/clean_data/dsc.$cell_type.2400.cutsites.smoothed.200.bw
    wget -P train_data/clean_data https://atacworks-paper.s3.us-east-2.amazonaws.com/dsc_atac_blood_cell_denoising_experiments/50_cells/train_data/clean_data/dsc.$cell_type.2400.cutsites.smoothed.200.3.narrowPeak
done
```
- Encode all the training data and save in the `train_h5` directory.
```
mkdir train_h5
cell_types = (CD19 Mono)
subsamples=(1 2 3 4 5)
for cell_type in ${cell_types[*]}; do
    for subsample in ${subsamples[*]}; do
        python $atacworks/scripts/bw2h5.py \
           --noisybw train_data/noisy_data/dsc.$subsample.${cell_type}.50.cutsites.smoothed.200.bw \
           --cleanbw train_data/clean_data/dsc.${cell_type}.2400.cutsites.smoothed.200.bw \
           --cleanpeakbw train_data/clean_data/dsc.${cell_type}.2400.cutsites.smoothed.200.3.narrowPeak.bw \
           --intervals training_intervals.bed \
           --out_dir train_h5 \
           --prefix ${cell_type}.$subsample.50.2400.train \
           --pad 5000 \
           --nonzero
    done
done
```
- Encode all the validation data and save in the `val_h5` directory.
```
mkdir val_h5
cell_types = (CD19 Mono)
for cell_type in ${cell_types[*]}; do
    python $atacworks/scripts/bw2h5.py \
           --noisybw train_data/noisy_data/dsc.1.${cell_type}.50.cutsites.smoothed.200.bw \
           --cleanbw train_data/clean_data/dsc.${cell_type}.2400.cutsites.smoothed.200.bw \
           --cleanpeakbw train_data/clean_data/dsc.${cell_type}.2400.cutsites.smoothed.200.3.narrowPeak.bw \
           --intervals intervals/val_intervals.bed \
           --out_dir val_h5 \
           --prefix ${cell_type}.50.2400.val \
           --pad 5000
done
```
- Train using all of the training and validation data. Here, we supply the directories `train_h5` and `val_h5`, and the model uses all the files within these directories for training and validation respectively.
```
python $atacworks/scripts/main.py train \
        --config configs/train_config.yaml \
        --config_mparams configs/model_structure.yaml \
        --train_files train_h5 \
        --val_files val_h5 \
```
