#!/bin/bash
datasetname=segmentation
project=segmentation2
datadir=/media/prostate/20210331_pdl1/data/annotation/segmentation_gt/20210524forsatou_pdl1_x10_512x512_0531_annotate
weight_decay=1e-5
resume=''
max_epoch=500
val_start=0
val_epoch=1
batch_size=8
device='0'
num_workers=12
input_size=512
crop_size=512
downsample_ratio=1
activation=identity # available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and **none**.
deep_supervision=1
cfg=/media/prostate/20210331_pdl1/segmentation/dm-count/models/hrnet_seg_ocr/experiments/test.yaml

visdom_server=http://localhost # if none, nothing will be sent to server.
visdom_port=8990
neptune_tag=('seg')


## change
neptune_project=colon-segmentation
neptune_user=satokiyo
neptune_api_token=eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwMTAxZjFkZS1jODNmLTQ2MWQtYWJhYi1kZTM5OGQ3NWYyZDAifQ==
 
classes=6
deep_supervision=1
use_albumentation=1
downsample_ratio=1 
use_ocr=0
loss=nrdice
amp=1

# change se_resnext50_32x4d
datadir=/media/HDD2/20210607_colon_segmentation/20211015_ReLearning_seg/segmentation_annotation_tool_USEareasize_USEtag/make_semseg_annotation/sample_OUTPUT/colon_x10_660x660_annotate_permit1percent_inc3/1101_CEE_CEE2_COT_COE_sato_modify
#datadir=/media/HDD2/20210607_colon_segmentation/20211015_ReLearning_seg/segmentation_annotation_tool_USEareasize_USEtag/make_semseg_annotation/sample_OUTPUT/colon_x10_700x700_annotate_permit1percent_inc3/1101_CEE_CEE2_COT_COE_sato_modify
datadir_ul=${datadir}/images/training/CEE0014401 # dummy
input_size=660
crop_size=660
#input_size=700
#crop_size=700

#lr_max=0.0025
lr_max=0.00015
lr_min=0.0001
max_epoch=50
encoder_name=se_resnext50_32x4d
batch_size=5
#encoder_name=efficientnetv2_m 
#batch_size=18
#encoder_name=efficientnetv2_s
#batch_size=25
weight_decay=4e-5
neptune_tag=('seg')
use_copy_paste=0
deep_supervision=1
neptune_tag=(${neptune_tag[@]} $neptune_project $encoder_name 'unet' 'input_size-'$input_size 'deep_supervision-'$deep_supervision 'copy_paste-'$use_copy_paste 'loss-'$loss)

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
--amp $amp \
--visdom-port $visdom_port \
--weight-decay $weight_decay \
--neptune_project $neptune_project \
--neptune_user $neptune_user \
--neptune_api_token $neptune_api_token \
--resume /media/HDD2/20210607_colon_segmentation/20211015_ReLearning_seg/DM-Count/ckpts/encoder-se_resnext50_32x4d_input-660_downrate-1_scale_pyramid_module-1_attention_branch-0_albumentation-1_copy_paste-0_deep_supervision-1_ocr-0_ssl-0/1102-185943/12_ckpt.tar

#--resume /media/HDD2/20210607_colon_segmentation/20211015_ReLearning_seg/DM-Count/ckpts/encoder-se_resnext50_32x4d_input-660_downrate-1_scale_pyramid_module-1_attention_branch-0_albumentation-1_copy_paste-0_deep_supervision-1_ocr-0_ssl-0/1102-130116/0_ckpt.tar

#--resume /media/HDD2/20210607_colon_segmentation/20211015_ReLearning_seg/DM-Count/ckpts/encoder-se_resnext50_32x4d_input-512_downrate-1_scale_pyramid_module-1_attention_branch-0_albumentation-1_copy_paste-0_deep_supervision-1_ocr-0_ssl-0/1019-224016/5_ckpt.tar

#--resume /media/HDD2/20210607_colon_segmentation/20211015_ReLearning_seg/DM-Count/ckpts/encoder-se_resnext50_32x4d_input-768_downrate-1_scale_pyramid_module-1_attention_branch-0_albumentation-1_copy_paste-0_deep_supervision-1_ocr-0_ssl-0/1022-194842/25_ckpt.tar

## change efficientnetv2_m
#lr_max=0.003
#lr_min=0.0005
#max_epoch=100
#encoder_name=efficientnetv2_m 
#batch_size=10
#weight_decay=4e-5
#neptune_tag=('seg')
#neptune_tag=(${neptune_tag[@]} $encoder_name 'unet' 'deep_supervision-'$deep_supervision 'copy_paste-'$use_copy_paste 'loss-'$loss)
