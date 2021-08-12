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
crop_size=512
downsample_ratio=1
activation=identity # Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and **None**.
deep_supervision=1
cfg=/media/prostate/20210331_PDL1/segmentation/DM-Count/models/hrnet_seg_ocr/experiments/test.yaml
loss=ce

visdom_server=http://localhost # if None, nothing will be sent to server.
neptune_tag=('me' 'seg')


### change # hrnet ocr
#activation=identity # Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and **None**.
#neptune_tag=(${neptune_tag[@]} 'hrnet_seg_ocr')
#encoder_name=hrnet_seg_ocr # vgg19_bn hrnet_seg hrnet_seg_ocr
#batch_size=2
#deep_supervision=0
#use_albumentation=1
#use_copy_paste=1
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
#--use_albumentation $use_albumentation \
#--use_copy_paste $use_copy_paste \
#--downsample-ratio $downsample_ratio \
#--activation $activation \
#--deep_supervision $deep_supervision \
#--cfg $cfg \
#--loss $loss \
#--weight-decay $weight_decay



### change se resnext
#activation=identity # Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and **None**.
#neptune_tag=('se_resnext50_32x4d' 'unet' 'deep_supervision')
##neptune_tag=(${neptune_tag[@]} 'vgg19_bn' 'unet' 'deep_supervision' 'copy_paste' 'ocr')
#encoder_name=se_resnext50_32x4d # vgg19_bn hrnet_seg hrnet_seg_ocr
#batch_size=6
#deep_supervision=1
#use_albumentation=1
#use_copy_paste=1
##downsample_ratio=2 # batch_size 6
#downsample_ratio=1 # batch_size 4
#use_ocr=0
#max_epoch=80
#loss=ce
##loss=dice
##loss=focal
##loss=jaccard
##loss=combo # bug? not work
##loss=lovasz # loss not reduce
##loss=softce 
##loss=nrdice
#
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
#--use_albumentation $use_albumentation \
#--use_copy_paste $use_copy_paste \
#--downsample-ratio $downsample_ratio \
#--activation $activation \
#--deep_supervision $deep_supervision \
#--use_ocr $use_ocr \
#--cfg $cfg \
#--loss $loss \
#--weight-decay $weight_decay
#

## change  coplenet
#activation=identity # Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and **None**.
##neptune_tag=(${neptune_tag[@]} 'vgg19_bn' 'unet' 'deep_supervision')
#neptune_tag=(${neptune_tag[@]} 'coplenet' 'deep_supervision' 'copy_paste')
##neptune_tag=(${neptune_tag[@]} 'coplenet' 'deep_supervision' 'copy_paste' 'ocr')
#encoder_name=coplenet # vgg19_bn hrnet_seg hrnet_seg_ocr
#batch_size=2
#deep_supervision=1
#use_albumentation=1
#use_copy_paste=1
#downsample_ratio=1 # batch_size 6
#use_ocr=0
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
#--use_albumentation $use_albumentation \
#--use_copy_paste $use_copy_paste \
#--downsample-ratio $downsample_ratio \
#--activation $activation \
#--deep_supervision $deep_supervision \
#--use_ocr $use_ocr \
#--cfg $cfg \
#--loss $loss \
#--weight-decay $weight_decay



### change se resnext
#activation=identity # Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and **None**.
#neptune_tag=('se_resnext50_32x4d' 'unet' 'deep_supervision')
##neptune_tag=(${neptune_tag[@]} 'vgg19_bn' 'unet' 'deep_supervision' 'copy_paste' 'ocr')
#encoder_name=se_resnext50_32x4d # vgg19_bn hrnet_seg hrnet_seg_ocr
#batch_size=6
#deep_supervision=1
#use_albumentation=1
#use_copy_paste=0
##downsample_ratio=2 # batch_size 6
#downsample_ratio=1 # batch_size 4
#use_ocr=0
#max_epoch=80
#loss=ce
##loss=dice
##loss=focal
##loss=jaccard
##loss=combo # bug? not work
##loss=lovasz # loss not reduce
##loss=softce 
##loss=nrdice
#
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
#--use_albumentation $use_albumentation \
#--use_copy_paste $use_copy_paste \
#--downsample-ratio $downsample_ratio \
#--activation $activation \
#--deep_supervision $deep_supervision \
#--use_ocr $use_ocr \
#--cfg $cfg \
#--loss $loss \
#--weight-decay $weight_decay
#
#
### change se resnext
#activation=identity # Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and **None**.
#neptune_tag=('vgg19_bn' 'unet' 'deep_supervision')
##neptune_tag=(${neptune_tag[@]} 'vgg19_bn' 'unet' 'deep_supervision' 'copy_paste' 'ocr')
#encoder_name=vgg19_bn # vgg19_bn hrnet_seg hrnet_seg_ocr  dpn98 resnet152 vgg19_bn timm-resnest50d efficientnet-b5 timm-resnest50d_4s2x40d vgg19_bn mobilenet_v2 timm-efficientnet-lite4 timm-skresnext50_32x4d se_resnext50_32x4d timm-efficientnet-b6 se_resnext101_32x4d xception 
#batch_size=5
#deep_supervision=1
#use_albumentation=1
#use_copy_paste=0
##downsample_ratio=2 # batch_size 6
#downsample_ratio=1 # batch_size 4
#use_ocr=0
#max_epoch=80
#loss=ce
##loss=dice
##loss=focal
##loss=jaccard
##loss=combo # bug? not work
##loss=lovasz # loss not reduce
##loss=softce 
##loss=nrdice
#
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
#--use_albumentation $use_albumentation \
#--use_copy_paste $use_copy_paste \
#--downsample-ratio $downsample_ratio \
#--activation $activation \
#--deep_supervision $deep_supervision \
#--use_ocr $use_ocr \
#--cfg $cfg \
#--loss $loss \
#--weight-decay $weight_decay
#
#
#
#
### change se resnext
#activation=identity # Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and **None**.
#neptune_tag=('hrnet_seg_ocr' 'deep_supervision')
##neptune_tag=(${neptune_tag[@]} 'vgg19_bn' 'unet' 'deep_supervision' 'copy_paste' 'ocr')
#encoder_name=hrnet_seg_ocr # vgg19_bn hrnet_seg hrnet_seg_ocr  dpn98 resnet152 vgg19_bn timm-resnest50d efficientnet-b5 timm-resnest50d_4s2x40d vgg19_bn mobilenet_v2 timm-efficientnet-lite4 timm-skresnext50_32x4d se_resnext50_32x4d timm-efficientnet-b6 se_resnext101_32x4d xception 
#batch_size=4
#deep_supervision=1
#use_albumentation=1
#use_copy_paste=0
##downsample_ratio=2 # batch_size 6
#downsample_ratio=1 # batch_size 4
#use_ocr=0
#max_epoch=80
#loss=ce
##loss=dice
##loss=focal
##loss=jaccard
##loss=combo # bug? not work
##loss=lovasz # loss not reduce
##loss=softce 
##loss=nrdice
#
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
#--use_albumentation $use_albumentation \
#--use_copy_paste $use_copy_paste \
#--downsample-ratio $downsample_ratio \
#--activation $activation \
#--deep_supervision $deep_supervision \
#--use_ocr $use_ocr \
#--cfg $cfg \
#--loss $loss \
#--weight-decay $weight_decay







### change se resnext
#activation=identity # Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and **None**.
#neptune_tag=('se_resnext50_32x4d' 'unet' 'deep_supervision')
##neptune_tag=(${neptune_tag[@]} 'vgg19_bn' 'unet' 'deep_supervision' 'copy_paste' 'ocr')
#encoder_name=se_resnext50_32x4d # vgg19_bn hrnet_seg hrnet_seg_ocr
#batch_size=4
#deep_supervision=1
#use_albumentation=1
#use_copy_paste=1
##downsample_ratio=2 # batch_size 6
#downsample_ratio=1 # batch_size 4
#use_ocr=0
#max_epoch=80
#loss=ce
##loss=dice
##loss=focal
##loss=jaccard
##loss=combo # bug? not work
##loss=lovasz # loss not reduce
##loss=softce 
##loss=nrdice
#
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
#--use_albumentation $use_albumentation \
#--use_copy_paste $use_copy_paste \
#--downsample-ratio $downsample_ratio \
#--activation $activation \
#--deep_supervision $deep_supervision \
#--use_ocr $use_ocr \
#--cfg $cfg \
#--loss $loss \
#--weight-decay $weight_decay
#
#
#
#
### change se resnext
#activation=identity # Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and **None**.
#neptune_tag=('se_resnext50_32x4d' 'unet' 'deep_supervision')
##neptune_tag=(${neptune_tag[@]} 'vgg19_bn' 'unet' 'deep_supervision' 'copy_paste' 'ocr')
#encoder_name=se_resnext50_32x4d # vgg19_bn hrnet_seg hrnet_seg_ocr
#batch_size=4
#deep_supervision=1
#use_albumentation=1
#use_copy_paste=1
##downsample_ratio=2 # batch_size 6
#downsample_ratio=1 # batch_size 4
#use_ocr=0
#max_epoch=80
##loss=ce
##loss=dice
##loss=focal
##loss=jaccard
##loss=combo # bug? not work
##loss=lovasz # loss not reduce
##loss=softce 
#loss=nrdice
#
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
#--use_albumentation $use_albumentation \
#--use_copy_paste $use_copy_paste \
#--downsample-ratio $downsample_ratio \
#--activation $activation \
#--deep_supervision $deep_supervision \
#--use_ocr $use_ocr \
#--cfg $cfg \
#--loss $loss \
#--weight-decay $weight_decay
#
#
#
#
### change se resnext
#activation=identity # Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and **None**.
#neptune_tag=('se_resnext50_32x4d' 'unet' 'deep_supervision')
##neptune_tag=(${neptune_tag[@]} 'vgg19_bn' 'unet' 'deep_supervision' 'copy_paste' 'ocr')
#encoder_name=se_resnext50_32x4d # vgg19_bn hrnet_seg hrnet_seg_ocr
#batch_size=4
#deep_supervision=1
#use_albumentation=1
#use_copy_paste=1
##downsample_ratio=2 # batch_size 6
#downsample_ratio=1 # batch_size 4
#use_ocr=0
#max_epoch=80
##loss=ce
##loss=dice
#loss=focal
##loss=jaccard
##loss=combo # bug? not work
##loss=lovasz # loss not reduce
##loss=softce 
##loss=nrdice
#
#
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
#--use_albumentation $use_albumentation \
#--use_copy_paste $use_copy_paste \
#--downsample-ratio $downsample_ratio \
#--activation $activation \
#--deep_supervision $deep_supervision \
#--use_ocr $use_ocr \
#--cfg $cfg \
#--loss $loss \
#--weight-decay $weight_decay
#
#
#
#
#
#
#
## change  coplenet
#activation=identity # Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and **None**.
#neptune_tag=('coplenet' 'deep_supervision')
##neptune_tag=(${neptune_tag[@]} 'vgg19_bn' 'unet' 'deep_supervision' 'copy_paste' 'ocr')
#encoder_name=coplenet # vgg19_bn hrnet_seg hrnet_seg_ocr  dpn98 resnet152 vgg19_bn timm-resnest50d efficientnet-b5 timm-resnest50d_4s2x40d vgg19_bn mobilenet_v2 timm-efficientnet-lite4 timm-skresnext50_32x4d se_resnext50_32x4d timm-efficientnet-b6 se_resnext101_32x4d xception 
#batch_size=6
#deep_supervision=1
#use_albumentation=1
#use_copy_paste=1
#downsample_ratio=1 # batch_size 4
#use_ocr=0
#max_epoch=80
#loss=nrdice
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
#--use_albumentation $use_albumentation \
#--use_copy_paste $use_copy_paste \
#--downsample-ratio $downsample_ratio \
#--activation $activation \
#--deep_supervision $deep_supervision \
#--use_ocr $use_ocr \
#--cfg $cfg \
#--loss $loss \
#--weight-decay $weight_decay



### change se resnest
#activation=identity # Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and **None**.
#neptune_tag=('timm-resnest50d_4s2x40d' 'unet' 'deep_supervision')
##neptune_tag=(${neptune_tag[@]} 'vgg19_bn' 'unet' 'deep_supervision' 'copy_paste' 'ocr')
#encoder_name=timm-resnest50d_4s2x40d # vgg19_bn hrnet_seg hrnet_seg_ocr  dpn98 resnet152 vgg19_bn timm-resnest50d efficientnet-b5 timm-resnest50d_4s2x40d vgg19_bn mobilenet_v2 timm-efficientnet-lite4 timm-skresnext50_32x4d se_resnext50_32x4d timm-efficientnet-b6 se_resnext101_32x4d xception 
#batch_size=2
#deep_supervision=1
#use_albumentation=1
#use_copy_paste=0
##downsample_ratio=2 # batch_size 6
#downsample_ratio=1 # batch_size 4
#use_ocr=0
#max_epoch=80
#loss=ce
##loss=dice
##loss=focal
##loss=jaccard
##loss=combo # bug? not work
##loss=lovasz # loss not reduce
##loss=softce 
##loss=nrdice
#
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
#--use_albumentation $use_albumentation \
#--use_copy_paste $use_copy_paste \
#--downsample-ratio $downsample_ratio \
#--activation $activation \
#--deep_supervision $deep_supervision \
#--use_ocr $use_ocr \
#--cfg $cfg \
#--loss $loss \
#--weight-decay $weight_decay



#activation=identity # Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and **None**.
#neptune_tag=('vgg19_bn' 'unet' 'deep_supervision')
##neptune_tag=(${neptune_tag[@]} 'vgg19_bn' 'unet' 'deep_supervision' 'copy_paste' 'ocr')
#encoder_name=vgg19_bn # vgg19_bn hrnet_seg hrnet_seg_ocr  dpn98 resnet152 vgg19_bn timm-resnest50d efficientnet-b5 timm-resnest50d_4s2x40d vgg19_bn mobilenet_v2 timm-efficientnet-lite4 timm-skresnext50_32x4d se_resnext50_32x4d timm-efficientnet-b6 se_resnext101_32x4d xception 
#batch_size=5
#deep_supervision=1
#use_albumentation=1
#use_copy_paste=1
##downsample_ratio=2 # batch_size 6
#downsample_ratio=1 # batch_size 4
#use_ocr=0
#max_epoch=120
##loss=ce
##loss=dice
##loss=focal
##loss=jaccard
##loss=combo # bug? not work
##loss=lovasz # loss not reduce
##loss=softce 
#loss=nrdice
#
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
#--use_albumentation $use_albumentation \
#--use_copy_paste $use_copy_paste \
#--downsample-ratio $downsample_ratio \
#--activation $activation \
#--deep_supervision $deep_supervision \
#--use_ocr $use_ocr \
#--cfg $cfg \
#--loss $loss \
#--weight-decay $weight_decay
#
#
#activation=identity # Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and **None**.
#neptune_tag=('vgg19_bn' 'unet' 'deep_supervision')
##neptune_tag=(${neptune_tag[@]} 'vgg19_bn' 'unet' 'deep_supervision' 'copy_paste' 'ocr')
#encoder_name=vgg19_bn # vgg19_bn hrnet_seg hrnet_seg_ocr  dpn98 resnet152 vgg19_bn timm-resnest50d efficientnet-b5 timm-resnest50d_4s2x40d vgg19_bn mobilenet_v2 timm-efficientnet-lite4 timm-skresnext50_32x4d se_resnext50_32x4d timm-efficientnet-b6 se_resnext101_32x4d xception 
#batch_size=4
#deep_supervision=1
#use_albumentation=1
#use_copy_paste=1
##downsample_ratio=2 # batch_size 6
#downsample_ratio=1 # batch_size 4
#use_ocr=1
#max_epoch=120
##loss=ce
##loss=dice
##loss=focal
##loss=jaccard
##loss=combo # bug? not work
##loss=lovasz # loss not reduce
##loss=softce 
#loss=nrdice
#
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
#--use_albumentation $use_albumentation \
#--use_copy_paste $use_copy_paste \
#--downsample-ratio $downsample_ratio \
#--activation $activation \
#--deep_supervision $deep_supervision \
#--use_ocr $use_ocr \
#--cfg $cfg \
#--loss $loss \
#--weight-decay $weight_decay
#


#
#activation=identity # Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and **None**.
#neptune_tag=('se_resnext50_32x4d' 'unet' 'deep_supervision' 'ocr' 'copy_paste')
##neptune_tag=('timm-resnest101e' 'unet' 'deep_supervision' 'ocr' 'copy_paste')
##neptune_tag=(${neptune_tag[@]} 'vgg19_bn' 'unet' 'deep_supervision' 'copy_paste' 'ocr')
#encoder_name=se_resnext50_32x4d # vgg19_bn hrnet_seg hrnet_seg_ocr  dpn98 resnet152 vgg19_bn timm-resnest50d efficientnet-b5 timm-resnest50d_4s2x40d vgg19_bn mobilenet_v2 timm-efficientnet-lite4 timm-skresnext50_32x4d se_resnext50_32x4d timm-efficientnet-b6 se_resnext101_32x4d xception 
#batch_size=4
#deep_supervision=1
#use_albumentation=1
#use_copy_paste=1
##downsample_ratio=2 # batch_size 6
#downsample_ratio=1 # batch_size 4
#use_ocr=1
#max_epoch=80
##loss=ce
##loss=dice
##loss=focal
##loss=jaccard
##loss=combo # bug? not work
##loss=lovasz # loss not reduce
##loss=softce 
#loss=nrdice
#
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
#--use_albumentation $use_albumentation \
#--use_copy_paste $use_copy_paste \
#--downsample-ratio $downsample_ratio \
#--activation $activation \
#--deep_supervision $deep_supervision \
#--use_ocr $use_ocr \
#--cfg $cfg \
#--loss $loss \
#--weight-decay $weight_decay
#
#
#



#activation=identity # Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and **None**.
#neptune_tag=('timm-resnest50d_1s4x24d' 'unet' 'deep_supervision' 'ocr' 'copy_paste')
##neptune_tag=('timm-resnest101e' 'unet' 'deep_supervision' 'ocr' 'copy_paste')
##neptune_tag=(${neptune_tag[@]} 'vgg19_bn' 'unet' 'deep_supervision' 'copy_paste' 'ocr')
#encoder_name=timm-resnest50d_1s4x24d # vgg19_bn hrnet_seg hrnet_seg_ocr  dpn98 resnet152 vgg19_bn timm-resnest50d efficientnet-b5 timm-resnest50d_4s2x40d vgg19_bn mobilenet_v2 timm-efficientnet-lite4 timm-skresnext50_32x4d se_resnext50_32x4d timm-efficientnet-b6 se_resnext101_32x4d xception 
#batch_size=4
#deep_supervision=1
#use_albumentation=1
#use_copy_paste=1
##downsample_ratio=2 # batch_size 6
#downsample_ratio=1 # batch_size 4
#use_ocr=1
#max_epoch=80
##loss=ce
##loss=dice
##loss=focal
##loss=jaccard
##loss=combo # bug? not work
##loss=lovasz # loss not reduce
##loss=softce 
#loss=nrdice
#
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
#--use_albumentation $use_albumentation \
#--use_copy_paste $use_copy_paste \
#--downsample-ratio $downsample_ratio \
#--activation $activation \
#--deep_supervision $deep_supervision \
#--use_ocr $use_ocr \
#--cfg $cfg \
#--loss $loss \
#--weight-decay $weight_decay



## efficientnetv2_s ocr
#activation=identity # Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and **None**.
#neptune_tag=('efficientnetv2_s' 'unet' 'deep_supervision' 'ocr' 'copy_paste')
##neptune_tag=(${neptune_tag[@]} 'vgg19_bn' 'unet' 'deep_supervision' 'copy_paste' 'ocr')
#encoder_name=efficientnetv2_s # vgg19_bn hrnet_seg hrnet_seg_ocr  dpn98 resnet152 vgg19_bn timm-resnest50d efficientnet-b5 timm-resnest50d_4s2x40d vgg19_bn mobilenet_v2 timm-efficientnet-lite4 timm-skresnext50_32x4d se_resnext50_32x4d timm-efficientnet-b6 se_resnext101_32x4d xception 
#batch_size=9
#deep_supervision=1
#use_albumentation=1
#use_copy_paste=1
##downsample_ratio=2 # batch_size 6
#downsample_ratio=1 # batch_size 4
#use_ocr=1
#max_epoch=80
##loss=ce
##loss=dice
##loss=focal
##loss=jaccard
##loss=combo # bug? not work
##loss=lovasz # loss not reduce
##loss=softce 
#loss=nrdice
#
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
#--use_albumentation $use_albumentation \
#--use_copy_paste $use_copy_paste \
#--downsample-ratio $downsample_ratio \
#--activation $activation \
#--deep_supervision $deep_supervision \
#--use_ocr $use_ocr \
#--cfg $cfg \
#--loss $loss \
#--weight-decay $weight_decay
#


## efficientnetv2_s ocr
#activation=identity # Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and **None**.
#neptune_tag=('efficientnetv2_m' 'unet' 'deep_supervision' 'copy_paste')
##neptune_tag=(${neptune_tag[@]} 'vgg19_bn' 'unet' 'deep_supervision' 'copy_paste' 'ocr')
#encoder_name=efficientnetv2_m # vgg19_bn hrnet_seg hrnet_seg_ocr  dpn98 resnet152 vgg19_bn timm-resnest50d efficientnet-b5 timm-resnest50d_4s2x40d vgg19_bn mobilenet_v2 timm-efficientnet-lite4 timm-skresnext50_32x4d se_resnext50_32x4d timm-efficientnet-b6 se_resnext101_32x4d xception 
#batch_size=8
#deep_supervision=1
#use_albumentation=1
#use_copy_paste=1
##downsample_ratio=2 # batch_size 6
#downsample_ratio=1 # batch_size 4
#use_ocr=0
#max_epoch=80
##loss=ce
##loss=dice
##loss=focal
##loss=jaccard
##loss=combo # bug? not work
##loss=lovasz # loss not reduce
##loss=softce 
#loss=nrdice
#
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
#--use_albumentation $use_albumentation \
#--use_copy_paste $use_copy_paste \
#--downsample-ratio $downsample_ratio \
#--activation $activation \
#--deep_supervision $deep_supervision \
#--use_ocr $use_ocr \
#--cfg $cfg \
#--loss $loss \
#--weight-decay $weight_decay
#



activation=identity # Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and **None**.
neptune_tag=('efficientnetv2_m' 'unet' 'deep_supervision' 'copy_paste')
#neptune_tag=(${neptune_tag[@]} 'vgg19_bn' 'unet' 'deep_supervision' 'copy_paste' 'ocr')
encoder_name=efficientnetv2_m # vgg19_bn hrnet_seg hrnet_seg_ocr  dpn98 resnet152 vgg19_bn timm-resnest50d efficientnet-b5 timm-resnest50d_4s2x40d vgg19_bn mobilenet_v2 timm-efficientnet-lite4 timm-skresnext50_32x4d se_resnext50_32x4d timm-efficientnet-b6 se_resnext101_32x4d xception 
batch_size=4
deep_supervision=1
use_albumentation=1
use_copy_paste=1
#downsample_ratio=2 # batch_size 6
downsample_ratio=1 # batch_size 4
use_ocr=0
max_epoch=80
#loss=ce
#loss=dice
#loss=focal
#loss=jaccard
#loss=combo # bug? not work
#loss=lovasz # loss not reduce
#loss=softce 
loss=nrdice


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
--weight-decay $weight_decay


