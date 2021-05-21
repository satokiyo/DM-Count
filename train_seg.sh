#!/bin/bash
datasetname=segmentation
project=segmentation
datadir=/media/prostate/20210315_LOWB/mmsegmentation/data/mydataset/0510_sample
lr=1e-3
weight_decay=1e-5
resume=''
max_epoch=500
val_start=0
val_epoch=1
batch_size=8
device='0'
num_workers=4
input_size=512
crop_size=512
downsample_ratio=1
activation=identity # Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and **None**.
deep_supervision=1
cfg=/media/prostate/20210315_LOWB/DM-Count_modify/DM-Count/models/hrnet_seg_ocr/experiments/test.yaml

visdom_server=http://localhost # if None, nothing will be sent to server.
neptune_tag=('me' 'seg')


### change # hrnet ocr
#activation=identity # Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and **None**.
#neptune_tag=(${neptune_tag[@]} 'hrnet_seg_ocr')
#encoder_name=hrnet_seg_ocr # vgg19_bn hrnet_seg hrnet_seg_ocr
#batch_size=2
#deep_supervision=0
#
#python3 train.py \
#--project $project \
#--neptune-tag ${neptune_tag[@]} \
#--dataset $datasetname \
#--data-dir $datadir \
#--device 0 \
#--lr $lr \
#--max-epoch $max_epoch \
#--val-epoch $val_epoch \
#--val-start $val_start \
#--batch-size $batch_size \
#--num-workers $num_workers \
#--input-size $input_size \
#--crop-size $crop_size \
#--visdom-server $visdom_server \
#--encoder_name $encoder_name \
#--classes 4 \
#--scale_pyramid_module 1 \
#--use_attention_branch 0 \
#--use_albumentation 1 \
#--downsample-ratio $downsample_ratio \
#--activation $activation \
#--deep_supervision $deep_supervision \
#--cfg $cfg \
#--weight-decay $weight_decay




## change
activation=identity # Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and **None**.
neptune_tag=(${neptune_tag[@]} 'vgg19_bn' 'unet' 'deep_supervision')
encoder_name=vgg19_bn # vgg19_bn hrnet_seg hrnet_seg_ocr
batch_size=4
deep_supervision=1

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
--deep_supervision $deep_supervision \
--cfg $cfg \
--weight-decay $weight_decay




## change
activation=identity # Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and **None**.
neptune_tag=(${neptune_tag[@]} 'vgg19_bn' 'unet')
encoder_name=vgg19_bn # vgg19_bn hrnet_seg hrnet_seg_ocr
batch_size=4
deep_supervision=0

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
--deep_supervision $deep_supervision \
--cfg $cfg \
--weight-decay $weight_decay

