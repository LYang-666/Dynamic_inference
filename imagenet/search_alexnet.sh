#!/usr/bin/env sh
imagenet_path="/dataset/imagenet"

############ directory to save result #############
DATE=`date +%Y-%m-%d`

if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save
    mkdir ./dataset
fi

############ Configurations ###############
model=search_alexnet
dataset=imagenet
epochs=100
batch_size=256
optimizer=SGD

# add more labels as additional info into the saving path
label_info=


CUDA_VISIBLE_DEVICES=0,1,2,3 python search_alexnet_new.py --dataset ${dataset} \
    --data_path ${imagenet_path}   \
    --arch ${model} --save_path ./save/subnet_search/alexnet/l1_group8_ft0_conv_fc \
    --epochs ${epochs} --learning_rate 0.005\
    --optimizer ${optimizer} \
	--schedule 80 120 160    --gammas 0.1 0.1 0.5    \
    --batch_size ${batch_size} --workers 4 --ngpu 4 \
    --print_freq 1000  \
    --acc \

    
  


  