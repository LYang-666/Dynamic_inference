#!/usr/bin/env sh

PYTHON="/home/li/.conda/envs/pytorch/bin/python"

############ directory to save result #############
DATE=`date +%Y-%m-%d`

if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save
    mkdir ./dataset
fi

############ Configurations ###############
model=ac_noise_resnet20
dataset=cifar10
epochs=100
batch_size=256
optimizer=SGD

# add more labels as additional info into the saving path
label_info=


$PYTHON subnet_search.py --dataset ${dataset} \
    --data_path ./dataset/   \
    --arch ${model} --save_path ./save/resnet20/sub_search_l1_ft50 \
    --epochs ${epochs} --learning_rate 0.01\
    --optimizer ${optimizer} \
	--schedule 80 120 160    --gammas 0.1 0.1 0.5    \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 3 \
    --print_freq 100  \
    --resume  ./save/resnet20/base_noise/2020-02-20/cifar10_aw_noise_resnet20_200_/model_best.pth.tar \
    --acc
    
  


  