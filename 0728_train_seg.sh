#!/bin/bash
datasetname=classification
datadir=/media/HDD2/20210728_colon_classification/data/ROI/x10_330x330/train
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
loss=ce

visdom_server=http://localhost # if None, nothing will be sent to server.
neptune_workspace_name='satokiyo'
neptune_project_name='colon_classification'
neptune_api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwMTAxZjFkZS1jODNmLTQ2MWQtYWJhYi1kZTM5OGQ3NWYyZDAifQ=='
neptune_tag=('colon' 'classification')


### change se resnext

encoder_name=mobilenet_v2 # vgg19_bn hrnet_seg hrnet_seg_ocr
use_albumentation=1
use_copy_paste=1
#loss=ce
#loss=dice
#loss=focal
#loss=jaccard
#loss=combo # bug? not work
#loss=lovasz # loss not reduce
#loss=softce 
loss=nrdice
#loss=abCE

#neptune_tag+=${encoder_name}
#neptune_tag+=albumentation-${use_albumentation}
#neptune_tag+=copy_paste-${} loss-${loss} max_epoch-${max_epoch})
#neptune_tag+=${encoder_name} albumentation-${use_albumentation} copy_paste-${} loss-${loss} max_epoch-${max_epoch})
#neptune_tag+=${encoder_name} albumentation-${use_albumentation} copy_paste-${} loss-${loss} max_epoch-${max_epoch})
neptune_tag=("${neptune_tag[@]}" ${encoder_name} albumentation-${use_albumentation} copy_paste-${use_copy_paste} loss-${loss} max_epoch-${max_epoch})

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
--encoder_name $encoder_name \
--classes 4 \
--use_albumentation $use_albumentation \
--use_copy_paste $use_copy_paste \
--activation $activation \
--loss $loss \
--weight-decay $weight_decay \

