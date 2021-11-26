#!/bin/bash
datadir=/media/prostate/20210331_PDL1/nuclei_detection/YOLO/darknet/cfg/task/datasets
weight_decay=1e-5
val_start=0
val_epoch=1
device='0'
num_workers=4
activation=identity # Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and **None**.
cfg=./models/hrnet/experiments/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml
norm-cood=0 # help='whether to norm cood when computing distance')
visdom_server=http://localhost # if None, nothing will be sent to server.
visdom_port=8991
t_0=10
t_mult=1
reg=1
wot=0.1
wtv=0.7
unsupervised_w=30
rampup_ends=0.2
iter_ot=150
amp=0

## change
datasetname=cell # qnrf, sha, shb, nwpu or cell
project=detection-dako1st-202108
classes=1
deep_supervision=1
use_albumentation=1
downsample_ratio=1 # batch_size 4

# change se_resnext50_32x4d 512x512 yolo pseudo-label
#datadir=/media/HDD2/20210331_PDL1/data/annotation/yolo_gt/dako1st_202108_x10_512x512_yolo_pseudo_label_contain_cancer_only/images
datadir=/media/HDD2/20210331_PDL1/data/annotation/yolo_gt/dako1st_202108_x10_512x512_yolo_pseudo_label/images
datadir_ul=/media/HDD2/20210331_PDL1/data/ROI/0721pdl1CHN_40xRename/semi_supervised_detection_x20_512x512/train/PDL1
input_size=512
crop_size=512
resize=512
use_ssl=0
lr_max=0.0018
lr_min=0.0013
max_epoch=30
encoder_name=se_resnext50_32x4d
batch_size=4
batch_size_ul=2
weight_decay=4e-5
neptune_tag=('detection')
neptune_tag=(${neptune_tag[@]} $encoder_name 'unet' 'deep_supervision-'$deep_supervision 'input_size-'$input_size 'resize-'$resize 'ssl-'$use_ssl )

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
#--num-workers $num_workers \
#--input-size $input_size \
#--crop-size $crop_size \
#--resize $resize \
#--visdom-server $visdom_server \
#--visdom-port $visdom_port \
#--encoder_name $encoder_name \
#--classes $classes \
#--scale_pyramid_module 1 \
#--use_attention_branch 0 \
#--use_albumentation $use_albumentation \
#--downsample-ratio $downsample_ratio \
#--activation $activation \
#--deep_supervision $deep_supervision \
#--cfg $cfg \
#--amp $amp \
#--weight-decay $weight_decay \
#--reg $reg \
#--num-of-iter-in-ot $iter_ot \
#--wot $wot \
#--wtv $wtv \
#--t_0 $t_0 \
#--t_mult $t_mult \
#--use_ssl $use_ssl \
#--batch-size-ul $batch_size_ul \
#--rampup_ends $rampup_ends \
#--unsupervised_w $unsupervised_w
#
##--resume /media/prostate/20210331_PDL1/nuclei_detection/DM-Count/ckpts/encoder-se_resnext50_32x4d_input-512_wot-0.1_wtv-0.7_reg-1.0_nIter-150_normCood-0/0714-195107/best_model_6.pth \


datadir=/media/HDD2/20210331_PDL1/data/annotation/yolo_gt/dako1st_202108_x10_512x512_yolo_pseudo_label/images
datadir_ul=/media/HDD2/20210331_PDL1/data/ROI/0721pdl1CHN_40xRename/semi_supervised_detection_x20_512x512/train/PDL1
input_size=512
crop_size=512
resize=512
max_epoch=30
deep_supervision=1
use_ssl=1
#lr_max=0.0018
#lr_max=0.0000378
lr_max=0.001
#lr_min=0.0013
lr_min=0.00001
batch_size=2
batch_size_ul=2
amp=1
unsupervised_w=5
#unsupervised_w=100
neptune_tag=('detection' 'not_only_cancer_all' 'unsupervised_w_5')
neptune_tag=(${neptune_tag[@]} $encoder_name 'unet' 'deep_supervision-'$deep_supervision 'input_size-'$input_size 'resize-'$resize 'ssl-'$use_ssl )

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
--resize $resize \
--visdom-server $visdom_server \
--visdom-port $visdom_port \
--encoder_name $encoder_name \
--classes $classes \
--scale_pyramid_module 1 \
--use_attention_branch 0 \
--use_albumentation $use_albumentation \
--downsample-ratio $downsample_ratio \
--activation $activation \
--deep_supervision $deep_supervision \
--cfg $cfg \
--amp $amp \
--weight-decay $weight_decay \
--reg $reg \
--num-of-iter-in-ot $iter_ot \
--wot $wot \
--wtv $wtv \
--t_0 $t_0 \
--t_mult $t_mult \
--use_ssl $use_ssl \
--batch-size-ul $batch_size_ul \
--rampup_ends $rampup_ends \
--unsupervised_w $unsupervised_w



### ssl
#deep_supervision=1
#use_ssl=1
#lr_max=0.0005
#lr_min=0.0005
#use_ssl=1
#batch_size=2
#batch_size_ul=2
#neptune_tag=('detection')
#neptune_tag=(${neptune_tag[@]} $encoder_name 'unet' 'deep_supervision-'$deep_supervision 'input_size-'$input_size 'resize-'$resize 'ssl-'$use_ssl )
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
#--num-workers $num_workers \
#--input-size $input_size \
#--crop-size $crop_size \
#--resize $resize \
#--visdom-server $visdom_server \
#--visdom-port $visdom_port \
#--encoder_name $encoder_name \
#--classes $classes \
#--scale_pyramid_module 1 \
#--use_attention_branch 0 \
#--use_albumentation $use_albumentation \
#--downsample-ratio $downsample_ratio \
#--activation $activation \
#--deep_supervision $deep_supervision \
#--cfg $cfg \
#--amp $amp \
#--weight-decay $weight_decay \
#--reg $reg \
#--num-of-iter-in-ot $iter_ot \
#--wot $wot \
#--wtv $wtv \
#--t_0 $t_0 \
#--t_mult $t_mult \
#--use_ssl $use_ssl \
#--batch-size-ul $batch_size_ul \
#--rampup_ends $rampup_ends \
#--unsupervised_w $unsupervised_w




## change se_resnext50_32x4d 512x512 yolo pseudo-label
#use_ssl=0
#lr_max=0.0018
#lr_min=0.0013
#max_epoch=30
#encoder_name=se_resnext50_32x4d
#batch_size=4
#batch_size_ul=2
#weight_decay=4e-5
#neptune_tag=('detection' 'w_deep_sup=10')
#neptune_tag=(${neptune_tag[@]} $encoder_name 'unet' 'deep_supervision-'$deep_supervision 'input_size-'$input_size 'resize-'$resize 'ssl-'$use_ssl )
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
#--num-workers $num_workers \
#--input-size $input_size \
#--crop-size $crop_size \
#--resize $resize \
#--visdom-server $visdom_server \
#--visdom-port $visdom_port \
#--encoder_name $encoder_name \
#--classes $classes \
#--scale_pyramid_module 1 \
#--use_attention_branch 0 \
#--use_albumentation $use_albumentation \
#--downsample-ratio $downsample_ratio \
#--activation $activation \
#--deep_supervision $deep_supervision \
#--cfg $cfg \
#--amp $amp \
#--weight-decay $weight_decay \
#--reg $reg \
#--num-of-iter-in-ot $iter_ot \
#--wot $wot \
#--wtv $wtv \
#--t_0 $t_0 \
#--t_mult $t_mult \
#--use_ssl $use_ssl \
#--batch-size-ul $batch_size_ul \
#--rampup_ends $rampup_ends \
#--unsupervised_w $unsupervised_w




#


## use all data 
## 学習データを3000より多くする
#deep_supervision=1
#use_ssl=0
#batch_size=7
#neptune_tag=('detection' 'all_data')
#neptune_tag=(${neptune_tag[@]} $encoder_name 'unet' 'deep_supervision-'$deep_supervision 'input_size-'$input_size 'resize-'$resize 'ssl-'$use_ssl )
#datadir=/media/HDD2/20210331_PDL1/data/annotation/yolo_gt/dako1st_202108_x10_512x512_yolo_pseudo_label/images
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
#--num-workers $num_workers \
#--input-size $input_size \
#--crop-size $crop_size \
#--resize $resize \
#--visdom-server $visdom_server \
#--visdom-port $visdom_port \
#--encoder_name $encoder_name \
#--classes $classes \
#--scale_pyramid_module 1 \
#--use_attention_branch 0 \
#--use_albumentation $use_albumentation \
#--downsample-ratio $downsample_ratio \
#--activation $activation \
#--deep_supervision $deep_supervision \
#--cfg $cfg \
#--amp $amp \
#--weight-decay $weight_decay \
#--reg $reg \
#--num-of-iter-in-ot $iter_ot \
#--wot $wot \
#--wtv $wtv \
#--t_0 $t_0 \
#--t_mult $t_mult \
#--use_ssl $use_ssl \
#--batch-size-ul $batch_size_ul \
#--rampup_ends $rampup_ends \
#--unsupervised_w $unsupervised_w

