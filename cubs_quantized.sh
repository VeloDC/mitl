#!/bin/sh
JOB_NAME='cubs_quantized_resnet18'
LOGDIR='./logs/'$JOB_NAME
DATA_DIR='/home/antonio/Data/Datasets/cubs_cropped'
MASK='QuantizedConv2d' # uno tra QuantizedConv2d o MaskedConv2d, oppure togliere per finetuning standard
CLASSES=200
LR=0.0001
EPOCHS=30
STEP=10
GAMMA=0.1
mkdir "$LOGDIR"
python train.py $JOB_NAME $DATA_DIR --num_classes=$CLASSES --mask=$MASK --adam --lr=$LR --num_epochs=$EPOCHS --step_size=$STEP --gamma=$GAMMA 2>&1 |tee outputs/"$NAME".txt
