#!/bin/bash
datasetname=segmentation
project=test
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
crop_size=512
downsample_ratio=1
activation=identity # Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and **None**.
deep_supervision=1
cfg=/media/prostate/20210315_LOWB/DM-Count_modify/DM-Count/models/hrnet_seg_ocr/experiments/test.yaml
loss=ce

visdom_server=http://localhost # if None, nothing will be sent to server.
neptune_tag=('me' 'seg')




## change se resnext
activation=identity # Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and **None**.
neptune_tag=(${neptune_tag[@]} 'se_resnext50_32x4d' 'unet' 'deep_supervision')
#neptune_tag=(${neptune_tag[@]} 'vgg19_bn' 'unet' 'deep_supervision' 'copy_paste' 'ocr')
encoder_name=se_resnext50_32x4d # vgg19_bn hrnet_seg hrnet_seg_ocr
batch_size=2
deep_supervision=1
use_albumentation=1
use_copy_paste=0
#downsample_ratio=2 # batch_size 6
downsample_ratio=1 # batch_size 4
use_ocr=0
max_epoch=25
#loss=ce
#loss=dice
#loss=focal
#loss=jaccard
#loss=combo # bug? not work
#loss=lovasz # loss not reduce
#loss=softce 
loss=nrdice
#loss=abCE
data_dir_ul=/media/prostate/20210331_PDL1/data/ROI/20210428_forSatou2/dako準拠染色_日本スライド/semi_supervised_seg_x10_512x512/train/dako準拠染色_日本スライド
use_ssl=1
batch_size_ul=2
unsupervised_w=30
rampup_ends=0.2


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
--use_albumentation $use_albumentation \
--use_copy_paste $use_copy_paste \
--downsample-ratio $downsample_ratio \
--activation $activation \
--deep_supervision $deep_supervision \
--use_ocr $use_ocr \
--cfg $cfg \
--loss $loss \
--weight-decay $weight_decay \
--use_ssl $use_ssl \
--data_dir_ul $data_dir_ul \
--batch-size-ul $batch_size_ul \
--rampup_ends $rampup_ends \
--unsupervised_w $unsupervised_w

