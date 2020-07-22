python train.py -b 64 -patch 2 -mul 1 -w 20 -n 1 -s 100 -lr 1e-3 -net resnet50 -pre 1 -pre_layers 158 128 71 32 -pre_step 50 50 50 300 -m 151 -ngpu 1 --single_type 1
python train.py -b 64 -patch 2 -mul 1 -w 20 -n 1 -s 100 -lr 1e-3 -net resnet50 -pre 1 -pre_layers 158 128 71 32 -pre_step 50 50 50 300 -m 151 -ngpu 1 --single_type 2
