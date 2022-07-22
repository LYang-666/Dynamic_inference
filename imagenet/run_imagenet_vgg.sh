PYTHON="/home/li/.conda/envs/pytorch/bin/python"
imagenet_path="/opt/imagenet/imagenet_compressed"

############ directory to save result #############
DATE=`date +%Y-%m-%d`

if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save
    mkdir ./save/${DATE}/
fi

############ Configurations ###############
model=ens_vgg11_bn
dataset=imagenet
epochs=64
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=test

CUDA_VISIBLE_DEVICES=0,1,2,3 $PYTHON main.py --dataset ${dataset} \
    --data_path ${imagenet_path}   \
    --arch ${model} --save_path ./save/ens_vgg/test \
    --epochs ${epochs} --learning_rate 0.01 \
    --optimizer ${optimizer} \
    --lr_scheduler linear_decaying \
	--schedule 30 60 90  --gammas 0.1 0.1 0.1 \
    --batch_size ${batch_size} --workers 16 --ngpu 4  \
    --print_freq 200 --decay 0.0005 \
    --SUNN \
    # --resume ./save/ens_resnet/pni_d_group4_ft20/model_best.pth.tar \
    # --evaluate \
 
 