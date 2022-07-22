#!/usr/bin/env sh

imagenet_path="/dataset/imagenet"

############ directory to save result #############
DATE=`date +%Y-%m-%d`

if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save
    mkdir ./dataset
fi

############ Configurations ###############
model=search_mobilenetv1
dataset=imagenet
epochs=100
batch_size=512
optimizer=SGD

# add more labels as additional info into the saving path
label_info=


CUDA_VISIBLE_DEVICES=0,1,2,3 python subnet_search_mobilenetv1.py --dataset ${dataset} \
    --data_path ${imagenet_path}   \
    --arch ${model} --save_path ./save/subnet_search/mobilenetv1/test \
    --epochs ${epochs} --learning_rate 0.01\
    --optimizer ${optimizer} \
	--schedule 80 120 160    --gammas 0.1 0.1 0.5    \
    --batch_size ${batch_size} --workers 4 --ngpu 4\
    --print_freq 100  \
    --resume ./save/inject_noise_mobilenetv1/base_noise/model_best.pth.tar \
    --acc
    # --resume  ./save/2020-03-23/cifar10_search_noise_mobilenet_350_base_noise/checkpoint.pth.tar  \
    # --acc
    
  


  