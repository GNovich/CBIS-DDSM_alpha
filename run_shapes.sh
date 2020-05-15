#python train_shape.py -b 128 -patch 2 -mul 1 -w 20 -n 2 -s 100 -lr 1e-3 -net resnet18 -pre 1 -pre_layers 59 -pre_step 50  -ngpu 2 -no_bkg 1 --shape_only 1 --n_shapes 4 --logdir four_shapes
python train_shape.py -b 128 -patch 2 -mul 1 -w 20 -n 2 -s 100 -lr 1e-3 -net resnet18 -pre 1 -pre_layers 59 -pre_step 50 -p 1 -a .75  -ngpu 2 -no_bkg 1 --shape_only 1 --n_shapes 4 --logdir four_shapes
python train_shape.py -b 128 -patch 2 -mul 1 -w 20 -n 2 -s 100 -lr 1e-3 -net resnet18 -pre 1 -pre_layers 59 -pre_step 50 -p 1 -a .9  -ngpu 2 -no_bkg 1 --shape_only 1 --n_shapes 4 --logdir four_shapes
python train_shape.py -b 128 -patch 2 -mul 1 -w 20 -n 2 -s 100 -lr 1e-3 -net resnet18 -pre 1 -pre_layers 59 -pre_step 50 -p 1 -a .95  -ngpu 2 -no_bkg 1 --shape_only 1 --n_shapes 4 --logdir four_shapes
python train_shape.py -b 128 -patch 2 -mul 1 -w 20 -n 2 -s 100 -lr 1e-3 -net resnet18 -pre 1 -pre_layers 59 -pre_step 50 -p 1 -a .99  -ngpu 2 -no_bkg 1 --shape_only 1 --n_shapes 4 --logdir four_shapes


#python train_shape.py -b 128 -patch 2 -mul 1 -w 20 -n 2 -s 100 -lr 1e-3 -net resnet18 -pre 1 -pre_layers 59 -pre_step 50 -m 150 -ngpu 2 -no_bkg 1
#python train_shape.py -b 128 -patch 2 -mul 1 -w 20 -n 2 -s 100 -lr 1e-3 -net resnet18 -pre 1 -pre_layers 59 -pre_step 50 -p 1 -a .75 -m 150 -ngpu 2 -no_bkg 1
#python train_shape.py -b 128 -patch 2 -mul 1 -w 20 -n 2 -s 100 -lr 1e-3 -net resnet18 -pre 1 -pre_layers 59 -pre_step 50 -p 1 -a .90 -m 150 -ngpu 2 -no_bkg 1
#python train_shape.py -b 128 -patch 2 -mul 1 -w 20 -n 2 -s 100 -lr 1e-3 -net resnet18 -pre 1 -pre_layers 59 -pre_step 50 -p 1 -a .95 -m 150 -ngpu 2 -no_bkg 1
#python train_shape.py -b 128 -patch 2 -mul 1 -w 20 -n 2 -s 100 -lr 1e-3 -net resnet18 -pre 1 -pre_layers 59 -pre_step 50 -p 1 -a .99 -m 150 -ngpu 2 -no_bkg 1

#python train_shape.py -e 100 -b 128 -patch 2 -mul 1 -w 20 -n 2 -s 100 -lr 1e-3 -net resnet18 -p 1 -a .01 -m 150 -ngpu 2 -no_bkg 1
#python train_shape.py -e 100 -b 128 -patch 2 -mul 1 -w 20 -n 2 -s 100 -lr 1e-3 -net resnet18 -p 1 -a .05 -m 150 -ngpu 2 -no_bkg 1
#python train_shape.py -e 100 -b 128 -patch 2 -mul 1 -w 20 -n 2 -s 100 -lr 1e-3 -net resnet18 -p 1 -a .1 -m 150 -ngpu 2 -no_bkg 1
#python train_shape.py -e 100 -b 128 -patch 2 -mul 1 -w 20 -n 2 -s 100 -lr 1e-3 -net resnet18 -p 1 -a .2 -m 150 -ngpu 2 -no_bkg 1
#python train_shape.py -e 100 -b 128 -patch 2 -mul 1 -w 20 -n 2 -s 100 -lr 1e-3 -net resnet18 -p 1 -a .25 -m 150 -ngpu 2 -no_bkg 1
