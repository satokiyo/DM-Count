#!/bin/bash
datasetname=cell # qnrf, sha, shb or nwpu
project=cell-down
datadir=/media/prostate/20210331_PDL1/nuclei_detection/YOLO/darknet/cfg/task/datasets
lr=5e-4
weight_decay=1e-5
resume=''
max_epoch=300
val_start=0
val_epoch=1
batch_size=8
device='0'
num_workers=4
input_size=512
crop_size=512
#input_size=256
#crop_size=256
downsample_ratio=1 #1
activation=identity # Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and **None**.
deep_supervision=0
cfg=./models/hrnet/experiments/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml
wot=0.1 # help='weight on OT loss')
wtv=0.01 # help='weight on TV loss')
reg=10.0
iter_ot=300
norm-cood=0 # help='whether to norm cood when computing distance')
visdom_server=http://localhost # if None, nothing will be sent to server.
neptune_tag=('me' 'detect')
t_0=10
t_mult=1

lr=5e-4
neptune_tag=(${neptune_tag[@]} 'se_resnext50_32x4d' 'unet' 'spm_drop_rate_0.2' 'without_can' 'iter150' 'deep_supervision' 'ssl')
encoder_name=se_resnext50_32x4d # vgg19_bn \
#encoder_name=vgg19_bn # vgg19_bn \
reg=1
wot=0.1
wtv=0.7
batch_size=2 # 4 if deep_supervision, 2 if use_ssl, 4 if normal
deep_supervision=1
use_albumentation=1
downsample_ratio=1 # batch_size 4
max_epoch=40
use_ssl=1
data_dir_ul=/media/prostate/20210331_PDL1/data/ROI/20210428_forSatou2/dako準拠染色_日本スライド/semi_supervised_seg_x40_512x512/train/dako準拠染色_日本スライド
batch_size_ul=2
unsupervised_w=30
rampup_ends=0.2
iter_ot=150


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
--classes 1 \
--scale_pyramid_module 1 \
--use_attention_branch 0 \
--use_albumentation $use_albumentation \
--downsample-ratio $downsample_ratio \
--activation $activation \
--deep_supervision $deep_supervision \
--cfg $cfg \
--reg $reg \
--num-of-iter-in-ot $iter_ot \
--wot $wot \
--wtv $wtv \
--t_0 $t_0 \
--t_mult $t_mult \
--weight-decay $weight_decay \
--use_ssl $use_ssl \
--data_dir_ul $data_dir_ul \
--batch-size-ul $batch_size_ul \
--rampup_ends $rampup_ends \
--unsupervised_w $unsupervised_w


