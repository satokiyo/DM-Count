#!/bin/bash
datasetname=classification
datadir=/media/HDD2/20210728_colon_classification/data/ROI/x10_330x330/train
lr=1e-3
weight_decay=1e-5
max_epoch=25
val_start=0
val_epoch=1
batch_size=30
device='0'
num_workers=4
input_size=330
crop_size=330
activation=identity # Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and **None**.

visdom_server=http://localhost # if None, nothing will be sent to server.
visdom_port=8990
neptune_workspace_name='satokiyo'
neptune_project_name='colon-classification'
neptune_api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwMTAxZjFkZS1jODNmLTQ2MWQtYWJhYi1kZTM5OGQ3NWYyZDAifQ=='
neptune_tag=('colon' 'classification')


# 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x4d', 'resnext101_32x8d', 'resnext101_32x16d', 'resnext101_32x32d'
# 'resnext101_32x48d', 'dpn68', 'dpn68b', 'dpn92', 'dpn98', 'dpn107', 'dpn131', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 
# 'senet154', 'se_resnet50', 'se_resnet101', 'se_resnet152', 'se_resnext50_32x4d', 'se_resnext101_32x4d', 'densenet121', 'densenet169', 'densenet201', 'densenet161', 
# 'inceptionresnetv2', 'inceptionv4', 'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6',
# 'efficientnet-b7', 'mobilenet_v2', 'xception', 'timm-efficientnet-b0', 'timm-efficientnet-b1', 'timm-efficientnet-b2', 'timm-efficientnet-b3', 'timm-efficientnet-b4', 
# 'timm-efficientnet-b5', 'timm-efficientnet-b6', 'timm-efficientnet-b7', 'timm-efficientnet-b8', 'timm-efficientnet-l2', 'timm-tf_efficientnet_lite0', 
# 'timm-tf_efficientnet_lite1', 'timm-tf_efficientnet_lite2', 'timm-tf_efficientnet_lite3', 'timm-tf_efficientnet_lite4', 'efficientnetv2_s', 'efficientnetv2_m', 
# 'efficientnetv2_l', 'timm-resnest14d', 'timm-resnest26d', 'timm-resnest50d', 'timm-resnest101e', 'timm-resnest200e', 'timm-resnest269e', 'timm-resnest50d_4s2x40d',
# 'timm-resnest50d_1s4x24d', 'timm-res2net50_26w_4s', 'timm-res2net101_26w_4s', 'timm-res2net50_26w_6s', 'timm-res2net50_26w_8s', 'timm-res2net50_48w_2s', 
# 'timm-res2net50_14w_8s', 'timm-res2next50', 'timm-regnetx_002', 'timm-regnetx_004', 'timm-regnetx_006', 'timm-regnetx_008', 'timm-regnetx_016', 'timm-regnetx_032',
# 'timm-regnetx_040', 'timm-regnetx_064', 'timm-regnetx_080', 'timm-regnetx_120', 'timm-regnetx_160', 'timm-regnetx_320', 'timm-regnety_002', 'timm-regnety_004', 
# 'timm-regnety_006', 'timm-regnety_008', 'timm-regnety_016', 'timm-regnety_032', 'timm-regnety_040', 'timm-regnety_064', 'timm-regnety_080', 'timm-regnety_120', 
# 'timm-regnety_160', 'timm-regnety_320', 'timm-skresnet18', 'timm-skresnet34', 'timm-skresnext50_32x4d'

#encoder_name=mobilenet_v2 
#encoder_name=efficientnet-b3
#encoder_name=se_resnext50_32x4d
#encoder_name=timm-resnest50d_1s4x24d
encoder_name=efficientnetv2_l

label_smooth=1
use_albumentation=1
classes=4
#flag_csv=('/media/HDD2/20210728_colon_classification/data/ndpi/segmentationデータ利用状況/CEE/flag.csv' 
#          '/media/HDD2/20210728_colon_classification/data/ndpi/segmentationデータ利用状況/COT/flag.csv' 
#          '/media/HDD2/20210728_colon_classification/data/ndpi/segmentationデータ利用状況/COE/flag.csv')
flag_csv=('/media/HDD2/20210728_colon_classification/data/ndpi/segmentationデータ利用状況/merged/flag.csv')

neptune_tag=("${neptune_tag[@]}" ${encoder_name} albumentation-${use_albumentation} label_smooth-${label_smooth})
resume=''


#python3 train.py \
#--neptune_workspace_name $neptune_workspace_name \
#--neptune_project_name $neptune_project_name \
#--neptune_api_token $neptune_api_token \
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
#--visdom-port $visdom_port \
#--encoder_name $encoder_name \
#--classes $classes \
#--use_albumentation $use_albumentation \
#--activation $activation \
#--flag_csv ${flag_csv[@]} \
#--label_smooth $label_smooth \
#--weight-decay $weight_decay 
#
#
##-----------------------------#
## change condition
##-----------------------------#
#encoder_name=efficientnetv2_l
#use_albumentation=0
#label_smooth=1
#neptune_tag=('colon' 'classification')
#neptune_tag=("${neptune_tag[@]}" ${encoder_name} albumentation-${use_albumentation} label_smooth-${label_smooth})
#
#python3 train.py \
#--neptune_workspace_name $neptune_workspace_name \
#--neptune_project_name $neptune_project_name \
#--neptune_api_token $neptune_api_token \
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
#--visdom-port $visdom_port \
#--encoder_name $encoder_name \
#--classes $classes \
#--use_albumentation $use_albumentation \
#--activation $activation \
#--flag_csv ${flag_csv[@]} \
#--label_smooth $label_smooth \
#--weight-decay $weight_decay 
#
#
#
#
##-----------------------------#
## change condition
##-----------------------------#
#encoder_name=efficientnetv2_l
#use_albumentation=1
#label_smooth=0
#neptune_tag=('colon' 'classification')
#neptune_tag=("${neptune_tag[@]}" ${encoder_name} albumentation-${use_albumentation} label_smooth-${label_smooth})
#
#python3 train.py \
#--neptune_workspace_name $neptune_workspace_name \
#--neptune_project_name $neptune_project_name \
#--neptune_api_token $neptune_api_token \
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
#--visdom-port $visdom_port \
#--encoder_name $encoder_name \
#--classes $classes \
#--use_albumentation $use_albumentation \
#--activation $activation \
#--flag_csv ${flag_csv[@]} \
#--label_smooth $label_smooth \
#--weight-decay $weight_decay 
#
#
#
#
#
#-----------------------------#
# change condition
#-----------------------------#
encoder_name=efficientnetv2_m
use_albumentation=1
label_smooth=1
lr=0.009
#lr=1e-3
max_epoch=100
batch_size=64
neptune_tag=('colon' 'classification' 'LRmodified')
neptune_tag=("${neptune_tag[@]}" ${encoder_name} albumentation-${use_albumentation} label_smooth-${label_smooth})

python3 train.py \
--neptune_workspace_name $neptune_workspace_name \
--neptune_project_name $neptune_project_name \
--neptune_api_token $neptune_api_token \
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
--visdom-port $visdom_port \
--encoder_name $encoder_name \
--classes $classes \
--use_albumentation $use_albumentation \
--activation $activation \
--flag_csv ${flag_csv[@]} \
--label_smooth $label_smooth \
--weight-decay $weight_decay 





##-----------------------------#
## change condition
##-----------------------------#
#encoder_name=efficientnetv2_s
#use_albumentation=1
#label_smooth=1
#neptune_tag=('colon' 'classification')
#neptune_tag=("${neptune_tag[@]}" ${encoder_name} albumentation-${use_albumentation} label_smooth-${label_smooth})
#
#python3 train.py \
#--neptune_workspace_name $neptune_workspace_name \
#--neptune_project_name $neptune_project_name \
#--neptune_api_token $neptune_api_token \
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
#--visdom-port $visdom_port \
#--encoder_name $encoder_name \
#--classes $classes \
#--use_albumentation $use_albumentation \
#--activation $activation \
#--flag_csv ${flag_csv[@]} \
#--label_smooth $label_smooth \
#--weight-decay $weight_decay 
#
#
#
#
#
#
##-----------------------------#
## change condition
##-----------------------------#
#encoder_name=mobilenet_v2 
#use_albumentation=1
#label_smooth=1
#neptune_tag=('colon' 'classification')
#neptune_tag=("${neptune_tag[@]}" ${encoder_name} albumentation-${use_albumentation} label_smooth-${label_smooth})
#
#python3 train.py \
#--neptune_workspace_name $neptune_workspace_name \
#--neptune_project_name $neptune_project_name \
#--neptune_api_token $neptune_api_token \
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
#--visdom-port $visdom_port \
#--encoder_name $encoder_name \
#--classes $classes \
#--use_albumentation $use_albumentation \
#--activation $activation \
#--flag_csv ${flag_csv[@]} \
#--label_smooth $label_smooth \
#--weight-decay $weight_decay 
#


##-----------------------------#
## change condition
##-----------------------------#
#batch_size=20
#encoder_name=se_resnext50_32x4d
#use_albumentation=1
#label_smooth=1
#neptune_tag=('colon' 'classification')
#neptune_tag=("${neptune_tag[@]}" ${encoder_name} albumentation-${use_albumentation} label_smooth-${label_smooth})
#
#python3 train.py \
#--neptune_workspace_name $neptune_workspace_name \
#--neptune_project_name $neptune_project_name \
#--neptune_api_token $neptune_api_token \
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
#--visdom-port $visdom_port \
#--encoder_name $encoder_name \
#--classes $classes \
#--use_albumentation $use_albumentation \
#--activation $activation \
#--flag_csv ${flag_csv[@]} \
#--label_smooth $label_smooth \
#--weight-decay $weight_decay 





##-----------------------------#
## change condition
##-----------------------------#
#batch_size=20
#encoder_name=efficientnet-b3
#use_albumentation=1
#label_smooth=1
#neptune_tag=('colon' 'classification')
#neptune_tag=("${neptune_tag[@]}" ${encoder_name} albumentation-${use_albumentation} label_smooth-${label_smooth})
#
#python3 train.py \
#--neptune_workspace_name $neptune_workspace_name \
#--neptune_project_name $neptune_project_name \
#--neptune_api_token $neptune_api_token \
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
#--visdom-port $visdom_port \
#--encoder_name $encoder_name \
#--classes $classes \
#--use_albumentation $use_albumentation \
#--activation $activation \
#--flag_csv ${flag_csv[@]} \
#--label_smooth $label_smooth \
#--weight-decay $weight_decay 
#
#
#