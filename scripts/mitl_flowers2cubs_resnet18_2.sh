#!/bin/sh
JOB='mitl_flowers2cubs_resnet18_2'
LOGDIR='/home/antonio/logs-pytorch'
DATASET='cubs'
DATA_DIR='/home/antonio/Data/Datasets/cubs_cropped/'
SB='/home/antonio/logs-mitl/finetune_resnet18_flowers_2_flowers_resnet18_multisource_90_0.010000/flowers_resnet18_multisource_90_0.010000.pth'
MODEL='resnet18_multisource'
PRETRAINED=True
LR=0.01
NUM_EPOCHS=90
STEP=30

mkdir "$LOGDIR"
python mitl/main.py $JOB --logdir=$LOGDIR --dataset_name=$DATASET \
--dataset_dir=$DATA_DIR --model_name=$MODEL --side_branches=$SB \
--pretrained=$PRETRAINED --learning_rate=$LR \
--num_epochs=$NUM_EPOCHS --step_size=$STEP \
2>&1 |tee "$LOGDIR"/"$JOB".txt
