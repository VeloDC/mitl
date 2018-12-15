#!/bin/sh
JOB='finetune_resnet18_flowers_5'
LOGDIR='/home/antonio/logs-pytorch'
DATASET='flowers'
DATA_DIR='/home/antonio/Data/Datasets/flowers/'
MODEL='resnet18_multisource'
PRETRAINED=True
LR=0.005
NUM_EPOCHS=90
STEP=30

mkdir "$LOGDIR"
python mitl/main.py $JOB --logdir=$LOGDIR --dataset_name=$DATASET \
--dataset_dir=$DATA_DIR --model_name=$MODEL \
--pretrained=$PRETRAINED --learning_rate=$LR \
--num_epochs=$NUM_EPOCHS --step_size=$STEP \
2>&1 |tee "$LOGDIR"/"$JOB".txt