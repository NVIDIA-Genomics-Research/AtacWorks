##Introduction

In this tutorial we use a pre-trained AtacWorks model to denoise and call peaks from low-coverage aggregate single-cell ATAC-seq data. We use the dsc-ATAC-seq dataset presented in (1), section (refer to page number, section). This dataset consists of single-cell ATAC-seq data from several types of human blood cells.

We selected 2400 NK cells from this dataset - this is our ‘clean’, high-coverage dataset. We then randomly sampled 50 of these 2400 NK cells. Here's what the ATAC-seq signal from 50 cells and 2400 cells looks like, for a region on chromosome 10:

<insert picture>

Compared to the 'clean' signal from 2400 cells, the aggregated ATAC-Seq profile of these 50 cells is noisy. The Pearson correlation between the 50-cell track and the 2400-cell track is only <insert correlation> on Chromosome 10. Because the signal is noisy, peak calls calculated by MACS2 on this data are also inaccurate; the AUPRC of peak calling from the noisy data is only <insert AUPRC> on Chromosome 10.

As reported in our paper, we trained an AtacWorks model to learn a mapping from 50-cell signal to 2400-cell signals and peak calls. In other words, given a noisy ATAC-seq signal from 50 cells, this model learned what the signal would look like - and where the peaks would be called - if we had sequenced 2400 cells. This model was trained on data from Monocytes and B cells, so it has not encountered data from NK cells. Likewise, Chromosome 10, which we will use for testing here, was also not used in training the model.

If you want to train your own AtacWorks model instead of using the model reported in the paper, refer to Tutorial 1 (link).


##Step 1: Set parameters

```
atacworks=<path to atacworks>
```

##Step 2: Download model

Download a pre-trained deep learning model (model.pth.tar) trained with dsc-ATAC-seq data from Monocytes and B cells. This model was reported and used in the AtacWorks paper (1).
```
aws s3 cp s3://atacworks-paper/dsc_atac_blood_cell_denoising_experiments/models/model.pth.tar ./models/model.pth.tar
```

##Step 3: Download config files

We also need to download the 'configs' directory containing config files for this experiment. The config files describe the parameters of the experiment, including the structure of the deep learning model.
```
aws s3 cp s3://atacworks-paper/dsc_atac_blood_cell_denoising_experiments/configs/ ./configs --recursive
```

##Step 4: Download the test dsc-ATAC-seq signal from 50 NK cells (~ 1M reads), in bigWig format

```
aws s3 cp s3://atacworks-paper/dsc_atac_blood_cell_denoising_experiments/test_data/noisy_data/dsc.1.NK.50.cutsites.smoothed.200.bw ./
```

##Step 5: Download genomic intervals spanning chromosome 10

The model we downloaded takes the input ATAC-seq signal in genomic intervals spanning 50,000 bp. The genomic positions of the intervals to use are supplied to the model in BED format. 
```
aws s3 cp s3://atacworks-paper/dsc_atac_blood_cell_denoising_experiments/intervals/hg19.50000.genome_intervals.bed ./intervals/hg19.50000.genome_intervals.bed
```

##Step 6: Read test data over the selected intervals, and save in .h5 format

We supply to `bw2h5.py` the bigWig file containing the noisy ATAC-seq signal, and the BED file containing the intervals on which to apply the model. This script reads the ATAC-seq signal within each supplied interval and saves it to a .h5 file.

```
python $atacworks/bw2h5.py \
           --noisybw dsc.1.NK.50.cutsites.smoothed.200.bw \
           --intervals intervals/hg19.50000.genome_intervals.bed \
           --prefix NK.50_cells \
           --pad 5000 \
           --batch_size 2000 \
           --nolabel
```
This creates a file `NK.50_cells.h5`, which contains the noisy ATAC-seq signal to be fed to the pre-trained model.

##Step 7: Inference on chromosome 10 producing denoised track and peak calls - for fast, binary peak calls

```
python $atacworks/main.py --infer \
    --infer_files NK.50_cells.h5 \
    --sizes $atacworks/example/reference/hg19.auto.sizes \
    --config configs/config_params.yaml \
    --config_mparams configs/model_structure.yaml \
    --infer_threshold 0.5
```

The inference results will be saved in the folder `output_latest`. This folder will contain four files: 
1. `NK_inferred.track.bedGraph` 
1. `NK_inferred.track.bw` 
3. `NK_inferred.peaks.bedGraph`. 
4. `NK_inferred.peaks.bw`

`NK_inferred.track.bedGraph` and `NK_inferred.track.bw` contain the denoised ATAC-seq track. `NK_inferred.peaks.bedGraph` and `NK_inferred.peaks.bw` contain the positions in the genome that are designated as peaks (the model predicts that the probability of these positions being part of a peak is at least 0.5)

##Step 8:

Delete peaks that are shorter than 50 bp in leangth, and format peak calls in BED format with coverage statistics and summit calls:

```
python $atacworks/peaksummary.py \
    --peakbw output_latest/NK_inferred.peaks.bw \
    --trackbw output_latest/NK_inferred.track.bw \
    --prefix output_latest/NK_inferred.peak_calls \
    --minlen 50
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

##References
(1) Lal, A., Chiang, Z.D., Yakovenko, N., Duarte, F.M., Israeli, J. and Buenrostro, J.D., 2019. AtacWorks: A deep convolutional neural network toolkit for epigenomics. BioRxiv, p.829481. (https://www.biorxiv.org/content/10.1101/829481v1)


##Appendix 1: Customize the inference command using config files

To change any of the parameters for inference with the deep learning model, you can edit the parameters in `configs/config_params.yaml` or `configs/model_structure.yaml` and run the commands in step 7-8 above. See the documentation in these files for an explanation of the parameters. 

If you are using your own model instead of the one provided, edit `configs/config_params.yaml` to supply the path to your model under `weights_path`, in place of `model.pth.tar`.


##Appendix 2: Output the peak probabilities in inference instead of peak calls

The model predicts the probability of every position on the genome being part of a peak. In the above command, we take a cutoff of 0.5, and output the positions of regions where the probability is greater than 0.5. To output the probability for every base in the genome without any cutoff, we use the following command:
```
python $atacworks/main.py --infer \
    --config configs/config_params.yaml \
    --config_mparams configs/model_structure.yaml
```
The inference results will be saved in the folder `output_latest`. This folder will contain the same 4 files described in Step 7. However, `NK_inferred.peaks.bedGraph` and `NK_inferred.peaks.bw` will contain the probability of being part of a peak, for every position in the genome. The `NK_inferred.peaks.bedGraph` file produced by this command is much larger than the file produced in Step 7.

The above command is useful in the following situations:
1. To calculate AUPRC or AUROC metrics.
2. If you are not sure what probability threshold to use for peak calling and want to try multiple thresholds.
3. If you wish to use the MACS2 subcommand `macs2 bdgpeakcall` for peak calling.

To call peaks from the probability track generated by this command, you can use `macs2 callpeak` from MACS2 (link) with the following command:
```
macs2 bdgpeakcall -i output_latest/inferred.peaks.bedGraph -o output_latest/inferred.peaks.narrowPeak -c 0.5
```
Where `0.5` is the probability threshold to call peaks. Note that the summit calls and peak sizes generated by this procedure will be slightly different from those produced by steps 7-8.

##Appendix 3: Visualize results

TBA

