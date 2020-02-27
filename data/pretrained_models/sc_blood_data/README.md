To use pretrained models in this directory for inference:

python <path to repo>/main.py --infer \
    --infer_files <test data in .h5 file> \
    --intervals_file <path to repo>/data/pretrained_models/24000.genome_intervals.bed \ 
    --sizes_file <path to repo>/data/reference/hg19.chrom.sizes \
    --infer_threshold 0.5 \
    --weights_path <path to repo>/data/pretrained_models/sc_blood_data/500.20000.CD4.resnet.5.2.12.1.201.0904.pth.tar \
    --out_home <output directory> \
    --label inference --result_fname output \
    --model resnet --nblocks 5 --nblocks_cla 2 --nfilt 12 --nfilt_cla 12 \
    --width 201  --width_cla 201 --dil 1 --dil_cla 1 \
    --task both --num_workers 0 --gen_bigwig
