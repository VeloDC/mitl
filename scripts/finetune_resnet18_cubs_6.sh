#!/bin/sh
JOB='finetune_resnet18_cubs_6'
LOGDIR='/home/antonio/logs-pytorch'
DATASET='cubs'
DATA_DIR='/home/antonio/Data/Datasets/cubs_cropped/'
MODEL='resnet18_multisource'
PRETRAINED=True
LR=0.001
NUM_EPOCHS=30
STEP=10

mkdir "$LOGDIR"
python mitl/main.py $JOB --logdir=$LOGDIR --dataset_name=$DATASET \
--dataset_dir=$DATA_DIR --model_name=$MODEL \
--pretrained=$PRETRAINED --learning_rate=$LR \
--num_epochs=$NUM_EPOCHS --step_size=$STEP \
2>&1 |tee "$LOGDIR"/"$JOB".txt
