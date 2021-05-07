#!/bin/bash
datasetname=shb # qnrf, sha, shb or nwpu
#datadir=/media/HDD2/count_dataset/ShanghaiTech/part_B #  <path to dataset>
datadir=/media/HDD2/count_dataset/ShanghaiTech/part_B_100images
lr=1e-5
weight_decay=1e-4
resume=''
max_epoch=800
val_start=0
val_epoch=1
#batch_size=4 理想
batch_size=2
device='0'
num_workers=4
crop_size=512
#crop_size=256
wot=0.1 # help='weight on OT loss')
wtv=0.01 # help='weight on TV loss')
reg=10.0
num-of-iter-in-ot=100
norm-cood=0 # help='whether to norm cood when computing distance')
visdom_server=http://localhost # if None, nothing will be sent to server.


# spm module affect (relatively) bad and cost high. attention slightly better (or equal to) normal execution with no cost.
python3 train.py \
--dataset $datasetname \
--data-dir $datadir \
--device 0 \
--lr $lr \
--max-epoch $max_epoch \
--val-epoch $val_epoch \
--val-start $val_start \
--batch-size $batch_size \
--num-workers $num_workers \
--crop-size $crop_size \
--visdom-server $visdom_server \
--encoder_name vgg19_bn \
--classes 1 \
--scale_pyramid_module 0 \
--use_attention_branch 0 \
--use_albumentation 1 \
--project test

#--encoder_name vgg19_bn \

#python3 train.py \
#--dataset $datasetname \
#--data-dir $datadir \
#--device 0 \
#--lr $lr \
#--max-epoch $max_epoch \
#--val-epoch $val_epoch \
#--val-start $val_start \
#--batch-size $batch_size \
#--num-workers $num_workers \
#--crop-size $crop_size \
#--visdom-server $visdom_server \
#--encoder_name timm-resnest50d_4s2x40d \
#--classes 1 \
#--scale_pyramid_module 0 \
#--use_attention_branch 0 \
#--use_albumentation 1



#python3 train.py \
#--dataset $datasetname \
#--data-dir $datadir \
#--device 0 \
#--lr $lr \
#--max-epoch $max_epoch \
#--val-epoch $val_epoch \
#--val-start $val_start \
#--batch-size $batch_size \
#--num-workers $num_workers \
#--crop-size $crop_size \
#--visdom-server $visdom_server \
#--encoder_name resnet50 \
#--classes 1 \
#--scale_pyramid_module 0 \
#--use_attention_branch 0 \
#--use_albumentation 1
#
#
#python3 train.py \
#--dataset $datasetname \
#--data-dir $datadir \
#--device 0 \
#--lr $lr \
#--max-epoch $max_epoch \
#--val-epoch $val_epoch \
#--val-start $val_start \
#--batch-size $batch_size \
#--num-workers $num_workers \
#--crop-size $crop_size \
#--visdom-server $visdom_server \
#--encoder_name resnet50 \
#--classes 1 \
#--scale_pyramid_module 0 \
#--use_attention_branch 0 \
#--use_albumentation 0
#
#
#python3 train.py \
#--dataset $datasetname \
#--data-dir $datadir \
#--device 0 \
#--lr $lr \
#--max-epoch $max_epoch \
#--val-epoch $val_epoch \
#--val-start $val_start \
#--batch-size $batch_size \
#--num-workers $num_workers \
#--crop-size $crop_size \
#--visdom-server $visdom_server \
#--encoder_name timm-resnest50d_4s2x40d \
#--classes 1 \
#--scale_pyramid_module 0 \
#--use_attention_branch 0 \
#--use_albumentation 1
#
#
#python3 train.py \
#--dataset $datasetname \
#--data-dir $datadir \
#--device 0 \
#--lr $lr \
#--max-epoch $max_epoch \
#--val-epoch $val_epoch \
#--val-start $val_start \
#--batch-size $batch_size \
#--num-workers $num_workers \
#--crop-size $crop_size \
#--visdom-server $visdom_server \
#--encoder_name timm-resnest50d_4s2x40d \
#--classes 1 \
#--scale_pyramid_module 0 \
#--use_attention_branch 0 \
#--use_albumentation 0
#
#
#
#python3 train.py \
#--dataset $datasetname \
#--data-dir $datadir \
#--device 0 \
#--lr $lr \
#--max-epoch $max_epoch \
#--val-epoch $val_epoch \
#--val-start $val_start \
#--batch-size $batch_size \
#--num-workers $num_workers \
#--crop-size $crop_size \
#--visdom-server $visdom_server \
#--encoder_name mobilenet_v2 \
#--classes 1 \
#--scale_pyramid_module 0 \
#--use_attention_branch 0 \
#--use_albumentation 1
#
#
#python3 train.py \
#--dataset $datasetname \
#--data-dir $datadir \
#--device 0 \
#--lr $lr \
#--max-epoch $max_epoch \
#--val-epoch $val_epoch \
#--val-start $val_start \
#--batch-size $batch_size \
#--num-workers $num_workers \
#--crop-size $crop_size \
#--visdom-server $visdom_server \
#--encoder_name mobilenet_v2 \
#--classes 1 \
#--scale_pyramid_module 0 \
#--use_attention_branch 0 \
#--use_albumentation 0
#


