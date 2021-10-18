#!/bin/bash
datasetname=segmentation
#project=segmentation2
#datadir=/media/prostate/20210331_pdl1/data/annotation/segmentation_gt/20210524forsatou_pdl1_x10_512x512_0531_annotate
weight_decay=1e-5
resume=''
val_start=0
val_epoch=1
device='0'
num_workers=4
input_size=512
crop_size=512
downsample_ratio=1
activation=identity # available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and **none**.
cfg=/media/prostate/20210331_pdl1/segmentation/dm-count/models/hrnet_seg_ocr/experiments/test.yaml

visdom_server=http://localhost # if none, nothing will be sent to server.
visdom_port=8990
neptune_tag=('seg')
amp=1


## change
project=segmentation-dako1st-202108
datadir=/media/hdd2/20210331_pdl1/data/annotation/segmentation_gt/dako1st_202108_x10_512x512
datadir_ul=/media/hdd2/20210331_pdl1/data/annotation/segmentation_gt/dako1st_202108_x10_512x512/images/training/2021-05-21-15.36.53 # dummy
classes=7
deep_supervision=1
use_albumentation=1
use_copy_paste=0
downsample_ratio=1 # batch_size 4
use_ocr=0
#loss=ce
#loss=dice
#loss=focal
#loss=jaccard
#loss=combo # bug? not work
#loss=lovasz # loss not reduce
#loss=softce 
loss=nrdice
#loss=abCE
datadir_ul=/media/prostate/20210331_PDL1/data/ROI/20210428_forSatou2/dako準拠染色_日本スライド/semi_supervised_seg_x10_512x512/train/dako準拠染色_日本スライド
use_ssl=1
unsupervised_w=30
rampup_ends=0.2


# change se_resnext50_32x4d 512x512
datadir=/media/HDD2/20210331_PDL1/data/annotation/segmentation_gt/dako1st_202108_x10_512x512_increase_rate2
#datadir=/media/HDD2/20210331_PDL1/data/annotation/segmentation_gt/dako1st_202108_x10_512x512
datadir_ul=/media/HDD2/20210331_PDL1/data/ROI/0721pdl1CHN_40xRename/semi_supervised_seg_x10_512x512/train/PDL1
input_size=512
crop_size=512
#lr_max=0.00025
lr_max=0.00015
lr_min=0.0001
max_epoch=20
encoder_name=se_resnext50_32x4d
batch_size=3
batch_size_ul=3
weight_decay=4e-5
neptune_tag=('seg' 'nrdice_for_unsup_loss')
use_copy_paste=0
use_ssl=1
neptune_tag=(${neptune_tag[@]} $encoder_name 'unet' 'deep_supervision-'$deep_supervision 'copy_paste-'$use_copy_paste 'loss-'$loss 'use_ssl-'$use_ssl)

python3 train.py \
--project $project \
--neptune-tag ${neptune_tag[@]} \
--dataset $datasetname \
--datadir $datadir \
--datadir_ul $datadir_ul \
--device 0 \
--lr-max $lr_max \
--lr-min $lr_min \
--max-epoch $max_epoch \
--val-epoch $val_epoch \
--val-start $val_start \
--batch-size $batch_size \
--batch-size-ul $batch_size_ul \
--num-workers $num_workers \
--input-size $input_size \
--crop-size $crop_size \
--visdom-server $visdom_server \
--encoder_name $encoder_name \
--classes $classes \
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
--visdom-port $visdom_port \
--weight-decay $weight_decay \
--amp $amp \
--rampup_ends $rampup_ends \
--unsupervised_w $unsupervised_w \
--use_ssl $use_ssl

#--resume /media/prostate/20210331_PDL1/segmentation/DM-Count/ckpts/encoder-se_resnext50_32x4d_input-512_downrate-1_scale_pyramid_module-1_attention_branch-0_albumentation-1_copy_paste-0_deep_supervision-1_ocr-0_ssl-1/0826-151549/9_ckpt.tar





## change se_resnext50_32x4d 768x768
#datadir=/media/HDD2/20210331_PDL1/data/annotation/segmentation_gt/dako1st_202108_x10_768x768_increase_rate2
#datadir_ul=/media/HDD2/20210331_PDL1/data/ROI/0721pdl1CHN_40xRename/semi_supervised_seg_x10_768x768/train/PDL1
#input_size=768
#crop_size=768
##lr_max=0.00025
#lr_max=0.00015
#lr_min=0.0001
#max_epoch=20
#encoder_name=se_resnext50_32x4d
#batch_size=1
#batch_size_ul=1
#weight_decay=4e-5
#neptune_tag=('seg')
#use_copy_paste=0
#use_ssl=1
#neptune_tag=(${neptune_tag[@]} $encoder_name 'unet' 'deep_supervision-'$deep_supervision 'copy_paste-'$use_copy_paste 'loss-'$loss 'use_ssl-'$use_ssl)
#
#python3 train.py \
#--project $project \
#--neptune-tag ${neptune_tag[@]} \
#--dataset $datasetname \
#--datadir $datadir \
#--datadir_ul $datadir_ul \
#--device 0 \
#--lr-max $lr_max \
#--lr-min $lr_min \
#--max-epoch $max_epoch \
#--val-epoch $val_epoch \
#--val-start $val_start \
#--batch-size $batch_size \
#--batch-size-ul $batch_size_ul \
#--num-workers $num_workers \
#--input-size $input_size \
#--crop-size $crop_size \
#--visdom-server $visdom_server \
#--encoder_name $encoder_name \
#--classes $classes \
#--scale_pyramid_module 1 \
#--use_attention_branch 0 \
#--use_albumentation $use_albumentation \
#--use_copy_paste $use_copy_paste \
#--downsample-ratio $downsample_ratio \
#--activation $activation \
#--deep_supervision $deep_supervision \
#--use_ocr $use_ocr \
#--cfg $cfg \
#--loss $loss \
#--visdom-port $visdom_port \
#--weight-decay $weight_decay \
#--amp $amp \
#--rampup_ends $rampup_ends \
#--unsupervised_w $unsupervised_w \
#--use_ssl $use_ssl

