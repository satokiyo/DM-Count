#!/bin/bash
datasetname=segmentation
project=segmentation
datadir=/media/prostate/20210315_LOWB/mmsegmentation/data/mydataset/0510_sample
lr=1e-3
weight_decay=1e-5
resume=''
max_epoch=100
val_start=0
val_epoch=1
batch_size=8
device='0'
num_workers=4
input_size=512
crop_size=512
downsample_ratio=1
activation=sigmoid # Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and **None**.

visdom_server=http://localhost # if None, nothing will be sent to server.
neptune_tag=('me' 'seg')


## change
activation=identity # Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and **None**.
neptune_tag=(${neptune_tag[@]} 'vgg19_bn' 'unet++')
encoder_name=vgg19_bn # vgg19_bn \
batch_size=2

python3 train.py \
--project $project \
--neptune-tag ${neptune_tag[@]} \
--dataset $datasetname \
--data-dir $datadir \
--device 0 \
--lr $lr \
--max-epoch $max_epoch \
--val-epoch $val_epoch \
--val-start $val_start \
--batch-size $batch_size \
--num-workers $num_workers \
--input-size $input_size \
--crop-size $crop_size \
--visdom-server $visdom_server \
--encoder_name $encoder_name \
--classes 4 \
--scale_pyramid_module 1 \
--use_attention_branch 0 \
--use_albumentation 1 \
--downsample-ratio $downsample_ratio \
--activation $activation \
--weight-decay $weight_decay

