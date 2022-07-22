PYTHON="/home/li/.conda/envs/pytorch/bin/python"
imagenet_path="/opt/imagenet/imagenet_compressed"

############ directory to save result #############
DATE=`date +%Y-%m-%d`

if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save
    mkdir ./save/${DATE}/
fi

############ Configurations ###############
model=ens_resnet18
dataset=imagenet
epochs=64
batch_size=512
optimizer=SGD
# add more labels as additional info into the saving path
label_info=test

CUDA_VISIBLE_DEVICES=0,1,2,3 $PYTHON main.py --dataset ${dataset} \
    --data_path ${imagenet_path}   \
    --arch ${model} --save_path ./save/ens_mobilenetv1/test \
    --epochs ${epochs} --learning_rate 0.05 \
    --optimizer ${optimizer} \
    --lr_scheduler cosine_decaying \
	--schedule 30 60 90  --gammas 0.1 0.1 0.1 \
    --batch_size ${batch_size} --workers 16 --ngpu 4  \
    --print_freq 100 --decay 0.00004 \
    --Indiv \
    --evaluate \

    # --resume ./save/ens_mobilenetv1/l1_group4_continue_0.001/checkpoint.pth.tar \

    # --evaluate \
    # --resume ./save/2020-02-27/imagenet_nas_noise_resnet18_100_pni_group4/model_best.pth.tar \
    # --evaluate \
 