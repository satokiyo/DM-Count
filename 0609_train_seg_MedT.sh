#!/bin/bash
datasetname=segmentation
project=segmentation2
datadir=/media/prostate/20210331_PDL1/data/annotation/segmentation_gt/20210524forSatou_PDL1_x10_512x512_0531_annotate
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
crop_size=256
downsample_ratio=1
activation=identity # Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and **None**.
deep_supervision=0
cfg=/media/prostate/20210331_PDL1/segmentation/DM-Count/models/hrnet_seg_ocr/experiments/test.yaml
loss=ce

visdom_server=http://localhost # if None, nothing will be sent to server.
neptune_tag=('me' 'seg')

neptune_tag=('MedT')
#encoder_name=coplenet # vgg19_bn hrnet_seg hrnet_seg_ocr  dpn98 resnet152 vgg19_bn timm-resnest50d efficientnet-b5 timm-resnest50d_4s2x40d vgg19_bn mobilenet_v2 timm-efficientnet-lite4 timm-skresnext50_32x4d se_resnext50_32x4d timm-efficientnet-b6 se_resnext101_32x4d xception 
batch_size=1
#deep_supervision=1
use_albumentation=1
use_copy_paste=1
downsample_ratio=1 # batch_size 4
#use_ocr=0
max_epoch=80
loss=nrdice

python3 train_MedT.py \
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
--classes 4 \
--use_albumentation $use_albumentation \
--use_copy_paste $use_copy_paste \
--downsample-ratio $downsample_ratio \
--activation $activation \
--cfg $cfg \
--loss $loss \
--weight-decay $weight_decay


#--encoder_name $encoder_name \
#--scale_pyramid_module 1 \
#--use_attention_branch 0 \
#--deep_supervision $deep_supervision \
#--use_ocr $use_ocr \