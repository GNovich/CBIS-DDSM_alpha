#python train_shape.py -b 128 -patch 2 -mul 1 -w 20 -n 2 -s 100 -lr 1e-3 -net resnet18 -pre 1 -pre_layers 59 -pre_step 50 -m 150 -ngpu 2 -no_bkg 1
#python train_shape.py -b 128 -patch 2 -mul 1 -w 20 -n 2 -s 100 -lr 1e-3 -net resnet18 -pre 1 -pre_layers 59 -pre_step 50 -p 1 -a .75 -m 150 -ngpu 2 -no_bkg 1
#python train_shape.py -b 128 -patch 2 -mul 1 -w 20 -n 2 -s 100 -lr 1e-3 -net resnet18 -pre 1 -pre_layers 59 -pre_step 50 -p 1 -a .90 -m 150 -ngpu 2 -no_bkg 1
#python train_shape.py -b 128 -patch 2 -mul 1 -w 20 -n 2 -s 100 -lr 1e-3 -net resnet18 -pre 1 -pre_layers 59 -pre_step 50 -p 1 -a .95 -m 150 -ngpu 2 -no_bkg 1
#python train_shape.py -b 128 -patch 2 -mul 1 -w 20 -n 2 -s 100 -lr 1e-3 -net resnet18 -pre 1 -pre_layers 59 -pre_step 50 -p 1 -a .99 -m 150 -ngpu 2 -no_bkg 1
python train_shape.py -e 50 -b 128 -patch 2 -mul 1 -w 20 -n 2 -s 100 -lr 1e-3 -net resnet18 -m 150 -ngpu 2 -no_bkg 1
python train_shape.py -e 50 -b 128 -patch 2 -mul 1 -w 20 -n 2 -s 100 -lr 1e-3 -net resnet18 -p 1 -a .75 -m 150 -ngpu 2 -no_bkg 1
python train_shape.py -e 50 -b 128 -patch 2 -mul 1 -w 20 -n 2 -s 100 -lr 1e-3 -net resnet18 -p 1 -a .9 -m 150 -ngpu 2 -no_bkg 1
python train_shape.py -e 50 -b 128 -patch 2 -mul 1 -w 20 -n 2 -s 100 -lr 1e-3 -net resnet18 -p 1 -a .95 -m 150 -ngpu 2 -no_bkg 1
python train_shape.py -e 50 -b 128 -patch 2 -mul 1 -w 20 -n 2 -s 100 -lr 1e-3 -net resnet18 -p 1 -a .99 -m 150 -ngpu 2 -no_bkg 1
