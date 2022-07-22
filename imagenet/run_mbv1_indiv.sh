#!/usr/bin/env sh


############ directory to save result #############
DATE=`date +%Y-%m-%d`

save_path='./save'

if [ ! -d "$DIRECTORY" ]; then
    mkdir ${save_path}
    mkdir ${save_path}/${DATE}/
fi

############ Configurations ###############
model=search_mobilenetv1
dataset=imagenet
epochs=20
batch_size=512
optimizer=SGD

pretrained_model=

# add more labels as additional info into the saving path
# label_info=mix_train_4comb_0_init
label_info=base_noise

python main.py --dataset ${dataset} \
    --data_path ./dataset/   \
    --arch ${model} --save_path ./save/mobilenetv1/base_noise \
    --epochs ${epochs} --learning_rate 0.01 \
    --optimizer ${optimizer} \
	--schedule 150 250 350  --gammas 0.1 0.1 0.1 \
    --batch_size ${batch_size} \
    --workers 4 --ngpu 3 \
    --print_freq 100 --decay 0.0005 \
    --Indiv \

