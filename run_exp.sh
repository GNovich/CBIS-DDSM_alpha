python train.py -b 64 -patch 2 -mul 1 -w 20 -n 1 -s 100 -lr 1e-3 -net resnet50 -pre 1 -pre_layers 158 128 71 32 -pre_step 50 50 50 300 -m 150 -ngpu 1
python train.py -b 64 -patch 2 -mul 1 -w 20 -n 1 -s 100 -lr 1e-3 -net resnet50 -pre 1 -pre_layers 158 128 71 32 -pre_step 15 50 50 300 -m 15 115 -ngpu 1
python train.py -b 64 -patch 2 -mul 1 -w 20 -n 1 -s 100 -lr 1e-3 -net resnet50 -pre 1 -pre_layers 158 128 71 32 2 -pre_step 50 50 50 50 250 -m 150  -ngpu 1
python train.py -b 64 -patch 2 -mul 1 -w 20 -n 1 -s 100 -lr 1e-3 -net resnet50 -pre 1 -pre_layers 158 128 71 32 2 -pre_step 50 50 50 50 250 -m 150 200 -ngpu 1
