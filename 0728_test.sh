#!/bin/bash
model_path=/media/HDD2/20210728_colon_classification/DM-Count/ckpts/encoder-efficientnetv2_m_albumentation-1_label_smooth-1/0809-151611/best_model_1.pth
data_dir=/media/HDD2/20210728_colon_classification/data/ROI/x10_330x330/train
out_dir=/media/HDD2/20210728_colon_classification/20210812_forward
datasetname=classification
encoder_name=efficientnetv2_m #se_resnext50_32x4d # dpn98 resnet152 vgg19_bn timm-resnest50d efficientnet-b5 timm-resnest50d_4s2x40d vgg19_bn mobilenet_v2 timm-efficientnet-lite4 timm-skresnext50_32x4d se_resnext50_32x4d timm-efficientnet-b6 se_resnext101_32x4d xception 
input_size=330
crop_size=330
batch_size=64
activation=identity # Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and **None**.
classes=4
#flag_csv=('/media/HDD2/20210728_colon_classification/data/ndpi/segmentationデータ利用状況/CEE/flag.csv' 
#          '/media/HDD2/20210728_colon_classification/data/ndpi/segmentationデータ利用状況/COT/flag.csv' 
#          '/media/HDD2/20210728_colon_classification/data/ndpi/segmentationデータ利用状況/COE/flag.csv')
flag_csv=('/media/HDD2/20210728_colon_classification/data/ndpi/segmentationデータ利用状況/merged/flag.csv')
gpu=1
test_type=test


python3 test.py \
--model-path $model_path \
--data-dir $data_dir \
--out-dir $out_dir \
--datasetname $datasetname \
--input-size $input_size  \
--crop-size $crop_size  \
--batch-size $batch_size \
--encoder_name $encoder_name \
--classes $classes \
--flag_csv ${flag_csv[@]} \
--resume $model_path \
--test-type $test_type \
--gpu $gpu 




