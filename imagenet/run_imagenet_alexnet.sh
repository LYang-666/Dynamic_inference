imagenet_path="/opt/imagenet/imagenet_compressed"

############ directory to save result #############
DATE=`date +%Y-%m-%d`

if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save
    mkdir ./save/${DATE}/
fi

############ Configurations ###############
model=ens_alexnet
dataset=imagenet
epochs=27
batch_size=256
optimizer=SGD
# add more labels as additional info into the saving path
label_info=test

CUDA_VISIBLE_DEVICES=0,1,2,3 $PYTHON main.py --dataset ${dataset} \
    --data_path ${imagenet_path}   \
    --arch ${model} --save_path ./save/ens_alexnet/test \
    --epochs ${epochs} --learning_rate 0.020312343892326293 \
    --optimizer ${optimizer} \
    --lr_scheduler linear_decaying \
	--schedule 30 60 90  --gammas 0.1 0.1 0.1 \
    --batch_size ${batch_size} --workers 16 --ngpu 4  \
    --print_freq 100 --decay 0.0005 \
    --USNN \
    --resume ./save/ens_alexnet/l1_group8_conv_fc_continue/checkpoint.pth.tar \
    --evaluate \
 
 