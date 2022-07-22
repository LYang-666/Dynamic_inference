imagenet_path="/dataset/imagenet"

############ directory to save result #############
DATE=`date +%Y-%m-%d`

if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save
    mkdir ./save/${DATE}/
fi

############ Configurations ###############
model=search_mobilenetv1
dataset=imagenet
epochs=30
batch_size=256
optimizer=SGD
# add more labels as additional info into the saving path
label_info=inject_noise

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --dataset ${dataset} \
    --data_path ${imagenet_path}   \
    --arch ${model} --save_path ./save/inject_noise_mobilenetv1/base_noise \
    --epochs ${epochs} --learning_rate 0.05 \
    --optimizer ${optimizer} \
    --decay 4e-5 \
    --lr_scheduler cosine_decaying \
	--schedule 30 60 90  --gammas 0.1 0.1 0.1 \
    --batch_size ${batch_size} --workers 16 --ngpu 4  \
    --print_freq 100 \
    --Indiv \
    # --evaluate \
 
 