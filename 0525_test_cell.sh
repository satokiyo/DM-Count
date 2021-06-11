#!/bin/bash
model_path=/media/prostate/20210315_LOWB/DM-Count_modify/DM-Count/ckpts/encoder-vgg19_bn_input-512_wot-0.1_wtv-0.7_reg-1.0_nIter-300_normCood-0/0505-111316/best_model_10.pth

#data_path=/media/prostate/20210331_PDL1/nuclei_detection/ndpi2roi_new/out/0525_PDL1/whole/train/PDL1
#data_path=/media/prostate/20210331_PDL1/nuclei_detection/ndpi2roi_new/out/0525_PDL1/cancer/train/PDL1
#data_path=/media/prostate/20210331_PDL1/nuclei_detection/ndpi2roi_new/out/0525_PDL1/cancer_seg_x20_1024x1024/train/PDL1
data_path=/media/prostate/20210331_PDL1/nuclei_detection/ndpi2roi_new/out/0525_PDL1/cancer_seg_x20_768x768/train/PDL1
dataset=cell #<dataset name: qnrf, sha, shb or nwpu>\
#pred_density_map_path=/media/prostate/20210331_PDL1/nuclei_detection/20210525_dmcount_forward/whole
#pred_density_map_path=/media/prostate/20210331_PDL1/nuclei_detection/20210525_dmcount_forward/cancer
#pred_density_map_path=/media/prostate/20210331_PDL1/nuclei_detection/20210525_dmcount_forward/cancer_seg_x20_1024x1024
pred_density_map_path=/media/prostate/20210331_PDL1/nuclei_detection/20210525_dmcount_forward/cancer_seg_x20_768x768
test_type=test_no_gt # 'val' 'val_no_gt' 'test_no_gt'
encoder_name=vgg19_bn # dpn98 resnet152 vgg19_bn timm-resnest50d efficientnet-b5 timm-resnest50d_4s2x40d vgg19_bn mobilenet_v2 timm-efficientnet-lite4 timm-skresnext50_32x4d se_resnext50_32x4d timm-efficientnet-b6 se_resnext101_32x4d xception 
downsample_ratio=1
input_size=768
crop_size=768
cfg=/media/prostate/20210315_LOWB/DM-Count_modify/DM-Count/models/hrnet/experiments/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml


python3 test.py \
--model-path $model_path \
--data-path $data_path \
--dataset $dataset \
--input-size $input_size \
--crop-size $crop_size  \
--test-type $test_type \
--encoder_name $encoder_name \
--classes 1 \
--scale_pyramid_module 1 \
--use_attention_branch 0 \
--downsample-ratio $downsample_ratio \
--cfg $cfg \
--pred-density-map-path $pred_density_map_path




