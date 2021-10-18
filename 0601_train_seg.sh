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
num_workers=4
input_size=512
crop_size=512
downsample_ratio=1
activation=identity # available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and **none**.
deep_supervision=1
cfg=/media/prostate/20210331_pdl1/segmentation/dm-count/models/hrnet_seg_ocr/experiments/test.yaml
loss=ce
amp=1

visdom_server=http://localhost # if none, nothing will be sent to server.
visdom_port=8990
neptune_tag=('seg')


## change
project=segmentation-dako1st-202108
datadir=/media/HDD2/20210331_PDL1/data/annotation/segmentation_gt/dako1st_202108_x10_512x512
datadir_ul=/media/HDD2/20210331_PDL1/data/annotation/segmentation_gt/dako1st_202108_x10_512x512/images/training/2021-05-21-15.36.53 # dummy
classes=7
deep_supervision=1
use_albumentation=1
use_copy_paste=1
downsample_ratio=1 # batch_size 4
use_ocr=0
max_epoch=50
#loss=ce
#loss=dice
#loss=focal
#loss=jaccard
#loss=combo # bug? not work
#loss=lovasz # loss not reduce
#loss=softce 
loss=nrdice


## change efficientnetv2_m
#lr_max=0.003
#lr_min=0.0005
#max_epoch=100
#encoder_name=efficientnetv2_m 
#batch_size=10
#weight_decay=4e-5
#neptune_tag=('seg')
#neptune_tag=(${neptune_tag[@]} $encoder_name 'unet' 'deep_supervision-'$deep_supervision 'copy_paste-'$use_copy_paste 'loss-'$loss)

## change se_resnext50_32x4d
#lr_max=0.00025
#lr_min=0.0001
#max_epoch=50
#encoder_name=se_resnext50_32x4d
#batch_size=4
#weight_decay=4e-5
#neptune_tag=('seg')
#use_copy_paste=1
#neptune_tag=(${neptune_tag[@]} $encoder_name 'unet' 'deep_supervision-'$deep_supervision 'copy_paste-'$use_copy_paste 'loss-'$loss)
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
#--amp $amp \
#--visdom-port $visdom_port \
#--weight-decay $weight_decay 
#
#
## change se_resnext50_32x4d 512x512
#datadir=/media/HDD2/20210331_PDL1/data/annotation/segmentation_gt/dako1st_202108_x10_512x512_increase_rate2
#datadir_ul=${datadir}/images/training/2021-05-21-15.36.53 # dummy
#input_size=512
#crop_size=512
#
##lr_max=0.00025
#lr_max=0.00015
#lr_min=0.0001
#max_epoch=50
#encoder_name=se_resnext50_32x4d
#batch_size=9
#weight_decay=4e-5
#neptune_tag=('seg')
#use_copy_paste=0
#neptune_tag=(${neptune_tag[@]} $encoder_name 'unet' 'deep_supervision-'$deep_supervision 'copy_paste-'$use_copy_paste 'loss-'$loss)
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
#--amp $amp \
#--visdom-port $visdom_port \
#--weight-decay $weight_decay 



# change se_resnext50_32x4d 768x768
datadir=/media/HDD2/20210331_PDL1/data/annotation/segmentation_gt/dako1st_202108_x10_768x768_increase_rate2
datadir_ul=${datadir}/images/training/2021-05-21-15.36.53 # dummy
input_size=768
crop_size=768

#lr_max=0.00025
lr_max=0.00015
lr_min=0.0001
max_epoch=50
encoder_name=se_resnext50_32x4d
batch_size=2
weight_decay=4e-5
neptune_tag=('seg')
use_copy_paste=0
neptune_tag=(${neptune_tag[@]} $encoder_name 'unet' 'deep_supervision-'$deep_supervision 'copy_paste-'$use_copy_paste 'loss-'$loss)

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
--weight-decay $weight_decay

#--resume /media/prostate/20210331_PDL1/segmentation/DM-Count/ckpts/encoder-se_resnext50_32x4d_input-768_downrate-1_scale_pyramid_module-1_attention_branch-0_albumentation-1_copy_paste-0_deep_supervision-1_ocr-0/0825-124534/17_ckpt.tar

## change se_resnext50_32x4d 768x768 reduce on plateau
#datadir=/media/HDD2/20210331_PDL1/data/annotation/segmentation_gt/dako1st_202108_x10_768x768_increase_rate2
#datadir_ul=${datadir}/images/training/2021-05-21-15.36.53 # dummy
#
#lr_max=0.0001
#lr_min=0.000001
#max_epoch=50
#encoder_name=se_resnext50_32x4d
#batch_size=4
#weight_decay=4e-5
#neptune_tag=('seg')
#use_copy_paste=1
#neptune_tag=(${neptune_tag[@]} $encoder_name 'unet' 'deep_supervision-'$deep_supervision 'copy_paste-'$use_copy_paste 'loss-'$loss)
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
#--amp $amp \
#--visdom-port $visdom_port \
#--weight-decay $weight_decay 



## todo 
# 512は確定にする。label smoothも不要とする。
# 512 without SSL -> check if 512 better performance is due to ema and roi_increase_rate2 or SSL. (now)
# -> then, check segmentation overlay for TMU -> リンパ球のあたりや癌のあたりが、未知データの東医大のデータに対しても、SSLより上手くいっているならSSLにする必要ないと判断出来る
# -> 結果を比較したところ、sslで学習した方が、癌の周囲のリンパ球など綺麗に出ている。癌に混ざったリンパ球と、腺癌の判定を気管支にしないこと。しかし扁平上皮癌や非癌腺管については若干、教師ありの方がいい。
# -> 教師ありでエポックを長くするorfinetuneするで試す。とりあえずfinetune。SSLで流し続けた場合のlossも見たい
# calculate TPS using SSL 512x512 model (帰るとき)
# SSL学習済モデルのエンコーダ凍結→デコーダだけsupervisedで学習?(帰った後)
# SSL with nrdice loss for unsup loss (not mse loss)(上が終わったら、かつSSLが単純な教師ありより良いと分かったら)

# compress .h5 (too large...)
# check onnx convertion
# detection model learning data
# ema + reduce on plateau + early stopping
# cancer / not cancer heatmap
# ndpi2roi slow -> c++ implementation? or parallel exec?


