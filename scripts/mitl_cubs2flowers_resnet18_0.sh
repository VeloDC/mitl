#!/bin/sh
JOB='mitl_cubs2flowers_resnet18_0'
LOGDIR='/home/antonio/logs-pytorch'
DATASET='flowers'
DATA_DIR='/home/antonio/Data/Datasets/flowers/'
SB='/home/antonio/logs-mitl/finetune_resnet18_cubs_1_cubs_resnet18_multisource_60_0.010000/cubs_resnet18_multisource_60_0.010000.pth'
MODEL='resnet18_multisource'
PRETRAINED=True
LR=0.01
NUM_EPOCHS=30
STEP=10

mkdir "$LOGDIR"
python mitl/main.py $JOB --logdir=$LOGDIR --dataset_name=$DATASET \
--dataset_dir=$DATA_DIR --model_name=$MODEL --side_branches=$SB \
--pretrained=$PRETRAINED --learning_rate=$LR \
--num_epochs=$NUM_EPOCHS --step_size=$STEP \
2>&1 |tee "$LOGDIR"/"$JOB".txt
