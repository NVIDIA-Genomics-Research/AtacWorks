To use pretrained models in this directory for inference:

python <path to repository>/main.py --infer \
    --infer_files <test data in .h5 file> \
    --weights_path <path to repository>/data/pretrained_models/bulk_blood_data/5000000.7cell.resnet.5.2.15.8.50.0803.pth.tar\
    --out_home <output directory> --label <output label> --result_fname <output file name> \
    --model resnet --nblocks 5 --nblocksc 2 --nfilt 15 --width 50 --dil 8 --task both
