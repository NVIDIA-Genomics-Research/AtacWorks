To use pretrained models in this directory for inference:

python <path to repository>/main.py --infer \
    --infer_files <test data in .h5 file> \
    --weights_path <path to repository>/data/pretrained_models/sc_blood_data/500.20000.CD4.resnet.5.2.12.1.201.0904.pth.tar\
    --out_home <output directory> --label <output label> --result_fname <output file name> \
    --model resnet --nblocks 5 --nblocksc 2 --nfilt 12 --width 201 --dil 1 --task both
