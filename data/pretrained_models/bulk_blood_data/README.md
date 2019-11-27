To use pretrained models in this directory for inference:

python <path to repo>/main.py --infer \
--infer_files <test data in .h5 file> \
--intervals_file <BED file of equal, non-overlapping genomic intervals to infer on> \ 
--sizes_file <path to repo>/example/reference/hg19.chrom.sizes \
--infer_threshold 0.5 \
--weights_path <path to repo>/data/pretrained_models/bulk_blood_data/5000000.7cell.resnet.5.2.15.8.50.0803.pth.tar \
--out_home <output directory> \
--label inference --result_fname output \
--model resnet --nblocks 5 --nblocks_cla 2 --nfilt 15 --nfilt_cla 15 \
 --width 50  --width_cla 50 --dil 8 --dil_cla 10 \
 --task both --num_workers 0 --gen_bigwig
