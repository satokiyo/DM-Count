#!/bin/bash
datasetname=cell # qnrf, sha, shb or nwpu
datadir=/media/prostate/20210331_PDL1/nuclei_detection/YOLO/darknet/cfg/task/datasets
#lr=1e-5
#lr=1e-3
lr=5e-4
weight_decay=1e-4
resume=''
#max_epoch=300
max_epoch=300
val_start=0
val_epoch=1
#batch_size=4 理想
batch_size=8
device='0'
num_workers=4
#input_size=512
input_size=256
#crop_size=512
crop_size=256
wot=0.1 # help='weight on OT loss')
wtv=0.01 # help='weight on TV loss')
reg=10.0
#num-of-iter-in-ot=100
iter_ot=300
norm-cood=0 # help='whether to norm cood when computing distance')
visdom_server=http://localhost # if None, nothing will be sent to server.
downsample_ratio=1 #1
project=cell-down
neptune_tag=('run-organization' 'me')
cfg=/media/prostate/20210315_LOWB/DM-Count_modify/DM-Count/models/hrnet/experiments/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml


## change
downsample_ratio=1 #1
lr=5e-4
weight_decay=0.0001
neptune_tag=(${neptune_tag[@]} 'vgg19_bn' )
encoder_name=vgg19_bn # vgg19_bn \
reg=1
wot=0.1
wtv=0.7
batch_size=4

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
--input-size $input_size \
--crop-size $crop_size \
--visdom-server $visdom_server \
--encoder_name $encoder_name \
--classes 1 \
--scale_pyramid_module 1 \
--use_attention_branch 0 \
--use_albumentation 1 \
--downsample-ratio $downsample_ratio \
--t_0 35 \
--t_mult 1 \
--weight-decay $weight_decay \
--neptune-tag ${neptune_tag[@]} \
--reg $reg \
--num-of-iter-in-ot $iter_ot \
--cfg $cfg \
--wot $wot \
--wtv $wtv \
--project $project

