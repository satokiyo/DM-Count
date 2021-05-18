#!/bin/bash
#model_path=/media/prostate/20210315_LOWB/DM-Count_modify/DM-Count/ckpts/encoder-resnet50_input-512_/0518-074833/best_model_3.pth
model_path=/media/prostate/20210315_LOWB/DM-Count_modify/DM-Count/ckpts/encoder-vgg19_bn_input-512_/0518-041904/best_model_4.pth
data_path=/media/prostate/20210315_LOWB/mmsegmentation/data/mydataset/0510_sample
dataset=segmentation
pred_density_map_path=/media/prostate/20210315_LOWB/DM-Count_modify/DM-Count/ckpts/tmp/segmentation/0518/unet_vgg19
encoder_name=vgg19_bn # dpn98 resnet152 vgg19_bn timm-resnest50d efficientnet-b5 timm-resnest50d_4s2x40d vgg19_bn mobilenet_v2 timm-efficientnet-lite4 timm-skresnext50_32x4d se_resnext50_32x4d timm-efficientnet-b6 se_resnext101_32x4d xception 
crop_size=512
downsample_ratio=1
activation=identity # Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and **None**.
test_type=val
#test_type=test_no_gt
#test_type=val_no_gt 


python3 test_seg.py \
--model-path $model_path \
--data-path $data_path \
--dataset $dataset \
--crop-size $crop_size  \
--test-type $test_type \
--encoder_name $encoder_name \
--classes 4 \
--scale_pyramid_module 1 \
--use_attention_branch 0 \
--downsample-ratio $downsample_ratio \
--activation $activation \
--pred-density-map-path $pred_density_map_path




