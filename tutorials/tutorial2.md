# Tutorial 2: Using a trained AtacWorks model to denoise ATAC-seq data and call peaks. 

## Introduction

In this tutorial we use a pre-trained AtacWorks model to denoise and call peaks from low-coverage aggregate single-cell ATAC-seq data. We use the dsc-ATAC-seq dataset presented in reference (1), section (when text is ready, add a reference to page number, section, table). This dataset consists of single-cell ATAC-seq data from several types of human blood cells.

We selected 2400 NK cells from this dataset - this is our ‘clean’, high-coverage dataset. We then randomly sampled 50 of these 2400 NK cells. Here's what the ATAC-seq signal from 50 cells and 2400 cells looks like, for a region on chromosome 10:

![subsampled_NK_cells](NK.2400.50.png)

Compared to the 'clean' signal from 2400 cells, the aggregated ATAC-seq profile of these 50 cells is noisy. Because the signal is noisy, peak calls calculated by MACS2 on this data (shown as red bars below the signal tracks) are also inaccurate. The AUPRC of peak calling by MACS2 on the noisy data is only 0.20.

As reported in our paper, we trained an AtacWorks model to learn a mapping from 50-cell signal to 2400-cell signals and peak calls. In other words, given a noisy ATAC-seq signal from 50 cells, this model learned what the signal would look like - and where the peaks would be called - if we had sequenced 2400 cells. This model was trained on data from Monocytes and B cells, so it has not encountered data from NK cells.

If you want to train your own AtacWorks model instead of using the model reported in the paper, refer to [Tutorial 1](tutorial1.md).


## Step 1: Set parameters

```
atacworks=<path to atacworks>
```

## Step 2: Download model

Download a pre-trained deep learning model (model.pth.tar) trained with dsc-ATAC-seq data from Monocytes and B cells. This model was reported and used in the AtacWorks paper (1).
```
mkdir models
wget -P models https://atacworks-paper.s3.us-east-2.amazonaws.com/dsc_atac_blood_cell_denoising_experiments/50_cells/models/model.pth.tar
```

## Step 3: Download config files

We also need to download the 'configs' directory containing config files for this experiment. The config files describe the structure of the deep learning model and the parameters used to run inference.
```
mkdir configs
wget -P configs https://atacworks-paper.s3.us-east-2.amazonaws.com/dsc_atac_blood_cell_denoising_experiments/50_cells/configs/infer_config.yaml
wget -P configs https://atacworks-paper.s3.us-east-2.amazonaws.com/dsc_atac_blood_cell_denoising_experiments/50_cells/configs/model_structure.yaml
```

## Step 4: Download the test dsc-ATAC-seq signal from 50 NK cells (~1M reads), in bigWig format

```
wget https://atacworks-paper.s3.us-east-2.amazonaws.com/dsc_atac_blood_cell_denoising_experiments/50_cells/test_data/noisy_data/dsc.1.NK.50.cutsites.smoothed.200.bw
```

## Step 5: Create genomic intervals to define regions for testing

The model we downloaded takes the input ATAC-seq signal in non-overlapping genomic intervals spanning 50,000 bp. To define the genomic regions for the model to read, we take the chromosomes on which we want to apply the model and split their lengths into 50,000-bp intervals, which we save in BED format. 
In this example, we will apply the model to chromosomes 1-22. The reference genome we use is hg19. We use the prepared chromosome sizes file `hg19.auto.sizes`, which contains the sizes of chromosomes 1-22 in hg19.
```
mkdir intervals
python $atacworks/scripts/get_intervals.py \
    --sizes $atacworks/example/reference/hg19.auto.sizes \
    --intervalsize 50000 \
    --out_dir intervals \
    --prefix hg19.50000 \
    --wg
```
This produces a BED file (`intervals/hg19.50000.genome_intervals.bed`).

For more information type `python $atacworks/scripts/get_intervals.py --help`

## Step 6: Read test data over the selected intervals, and save in .h5 format

We supply to `bw2h5.py` the bigWig file containing the noisy ATAC-seq signal, and the BED file containing the intervals on which to apply the model. This script reads the ATAC-seq signal within each supplied interval and saves it to a .h5 file.

```
python $atacworks/scripts/bw2h5.py \
           --noisybw dsc.1.NK.50.cutsites.smoothed.200.bw \
           --intervals intervals/hg19.50000.genome_intervals.bed \
           --out_dir ./ \
           --prefix NK.50_cells \
           --pad 5000 \
           --nolabel
```
This creates a file `NK.50_cells.h5`, which contains the noisy ATAC-seq signal to be fed to the pre-trained model.

For more information type `python $atacworks/scripts/bw2h5.py --help`

## Step 7: Inference on selected intervals, producing denoised track and binary peak calls

```
python $atacworks/scripts/main.py infer \
    --files NK.50_cells.h5 \
    --sizes_file $atacworks/example/reference/hg19.auto.sizes \
    --config configs/infer_config.yaml \
    --config_mparams configs/model_structure.yaml \
```

The inference results will be saved in the folder `output_latest`. This folder will contain four files: 
1. `NK_inferred.track.bedGraph` 
2. `NK_inferred.track.bw` 
3. `NK_inferred.peaks.bedGraph`. 
4. `NK_inferred.peaks.bw`

`NK_inferred.track.bedGraph` and `NK_inferred.track.bw` contain the denoised ATAC-seq track. `NK_inferred.peaks.bedGraph` and `NK_inferred.peaks.bw` contain the positions in the genome that are designated as peaks (the model predicts that the probability of these positions being part of a peak is at least 0.5)

To change any of the parameters for inference with the deep learning model, you can edit the parameters in `configs/infer_config.yaml` or `configs/model_structure.yaml` and run the commands in step 7-8 above. 

Type `python $atacworks/scripts/main.py infer --help` for an explanation of the parameters.

If you are using your own model instead of the one provided, edit `configs/infer_config.yaml` to supply the path to your model under `weights_path`, in place of `model.pth.tar`.

## Step 8: Format peak calls

Delete peaks that are shorter than 20 bp in leangth, and format peak calls in BED format with coverage statistics and summit calls:

```
python $atacworks/scripts/peaksummary.py \
    --peakbw output_latest/NK_inferred.peaks.bw \
    --trackbw output_latest/NK_inferred.track.bw \
    --prefix output_latest/NK_inferred.peak_calls \
    --minlen 20
```
This produces a file `output_latest/NK_inferred.peak_calls.bed` with 8 columns:
1. chromosome
2. start position of peak
3. end position of peak
4. length of peak (bp)
5. Mean coverage over peak
6. Maximum coverage in peak
7. Position of summit (relative to start)
8. Position of summit (absolute)

For more information type `python $atacworks/scripts/peaksummary.py --help`


## References
(1) Lal, A., Chiang, Z.D., Yakovenko, N., Duarte, F.M., Israeli, J. and Buenrostro, J.D., 2019. AtacWorks: A deep convolutional neural network toolkit for epigenomics. BioRxiv, p.829481. (https://www.biorxiv.org/content/10.1101/829481v1)


## Appendix 1: Output the peak probabilities in inference instead of peak calls

The model predicts the probability of every position on the genome being part of a peak. In the above command, we take a cutoff of 0.5, and output the positions of regions where the probability is greater than 0.5. To output the probability for every base in the genome without any cutoff, we use the following command:
```
python $atacworks/main.py infer \
    --files NK.50_cells.h5 \
    --sizes_file $atacworks/example/reference/hg19.auto.sizes \
    --config configs/infer_config.yaml \
    --config_mparams configs/model_structure.yaml
    --infer_threshold None
```
The inference results will be saved in the folder `output_latest`. This folder will contain the same 4 files described in Step 7. However, `NK_inferred.peaks.bedGraph` and `NK_inferred.peaks.bw` will contain the probability of being part of a peak, for every position in the genome. This command is significantly slower, and the `NK_inferred.peaks.bedGraph` file produced by this command is larger than the file produced in Step 7.

The above command is useful in the following situations:
1. To calculate AUPRC or AUROC metrics.
2. If you are not sure what probability threshold to use for peak calling and want to try multiple thresholds.
3. If you wish to use the MACS2 subcommand `macs2 bdgpeakcall` for peak calling.

To call peaks from the probability track generated by this command, you can use `macs2 callpeak` from MACS2 (link) with the following command:
```
macs2 bdgpeakcall -i output_latest/inferred.peaks.bedGraph -o output_latest/inferred.peaks.narrowPeak -c 0.5
```
Where `0.5` is the probability threshold to call peaks. Note that the summit calls and peak sizes generated by this procedure will be slightly different from those produced by steps 7-8.
