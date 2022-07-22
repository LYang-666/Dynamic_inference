#!/usr/bin/env sh

PYTHON="/home/li/.conda/envs/pytorch/bin/python"
imagenet_path="/opt/imagenet/imagenet_compressed"

############ directory to save result #############
DATE=`date +%Y-%m-%d`

if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save
    mkdir ./dataset
fi

############ Configurations ###############
model=ens_vgg11_bn
dataset=imagenet
epochs=100
batch_size=256
optimizer=SGD

# add more labels as additional info into the saving path
label_info=


CUDA_VISIBLE_DEVICES=0,1,2,3 $PYTHON search_vgg.py --dataset ${dataset} \
    --data_path ${imagenet_path}   \
    --arch ${model} --save_path ./save/subnet_search/vgg11/l1_group8_ft40 \
    --epochs ${epochs} --learning_rate 0.001\
    --optimizer ${optimizer} \
	--schedule 80 120 160    --gammas 0.1 0.1 0.5    \
    --batch_size ${batch_size} --workers 4 --ngpu 4 \
    --print_freq 1000  \
    --acc \
    # --resume  ./save/vgg/base_noise_vgg11/model_best.pth.tar \
    # --acc
    
  


  