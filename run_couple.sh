#python train_shape.py -b 128 -patch 2 -mul 1 -w 20 -n 2 -s 100 -lr 1e-3 -net resnet18 -pre 1 -pre_layers 59 -pre_step 50 -m 150 -ngpu 2 -no_bkg 1 --shape_only 1 --n_shapes 4 --logdir four_shapes
python train_shape.py -b 128 -patch 2 -mul 1 -w 20 -n 2 -s 100 -lr 1e-3 -net resnet18 -pre 1 -pre_layers 59 -pre_step 50 -m 150 -p 1 -a 0.01 -ngpu 2 -no_bkg 1 --shape_only 1 --n_shapes 4 --logdir four_shapes
python train_shape.py -b 128 -patch 2 -mul 1 -w 20 -n 2 -s 100 -lr 1e-3 -net resnet18 -pre 1 -pre_layers 59 -pre_step 50 -m 150 -p 1 -a 0.05 -ngpu 2 -no_bkg 1 --shape_only 1 --n_shapes 4 --logdir four_shapes
python train_shape.py -b 128 -patch 2 -mul 1 -w 20 -n 2 -s 100 -lr 1e-3 -net resnet18 -pre 1 -pre_layers 59 -pre_step 50 -m 150 -p 1 -a 0.075 -ngpu 2 -no_bkg 1 --shape_only 1 --n_shapes 4 --logdir four_shapes
python train_shape.py -b 128 -patch 2 -mul 1 -w 20 -n 2 -s 100 -lr 1e-3 -net resnet18 -pre 1 -pre_layers 59 -pre_step 50 -m 150 -p 1 -a 0.1 -ngpu 2 -no_bkg 1 --shape_only 1 --n_shapes 4 --logdir four_shapes
#python train.py -b 64 -patch 2 -mul 1 -w 20 -n 2 -s 100 -lr 1e-3 -net resnet50 -pre 1 -pre_layers 158 128 71 32 -pre_step 50 50 50 300 -m 150 -ngpu 2 --type_only 1
#python train.py -b 64 -patch 2 -mul 1 -w 20 -n 2 -s 100 -lr 1e-3 -net resnet50 -pre 1 -pre_layers 158 128 71 32 -pre_step 50 50 50 300 -m 150 -p 1 -a 0.01 -ngpu 2 --type_only 1
#python train.py -b 64 -patch 2 -mul 1 -w 20 -n 2 -s 100 -lr 1e-3 -net resnet50 -pre 1 -pre_layers 158 128 71 32 -pre_step 50 50 50 300 -m 150 -p 1 -a 0.05 -ngpu 2 --type_only 1
#python train.py -b 64 -patch 2 -mul 1 -w 20 -n 2 -s 100 -lr 1e-3 -net resnet50 -pre 1 -pre_layers 158 128 71 32 -pre_step 50 50 50 300 -m 150 -p 1 -a 0.075 -ngpu 2 --type_only 1
#python train.py -b 64 -patch 2 -mul 1 -w 20 -n 2 -s 100 -lr 1e-3 -net resnet50 -pre 1 -pre_layers 158 128 71 32 -pre_step 50 50 50 300 -m 150 -p 1 -a 0.1 -ngpu 2 --type_only 1
