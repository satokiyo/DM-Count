#!/bin/bash
#model_path=/media/prostate/20210315_LOWB/DM-Count_modify/DM-Count/ckpts/encoder-se_resnext50_32x4d_input-512_downrate-1_scale_pyramid_module-1_attention_branch-0_albumentation-1_copy_paste-1_deep_supervision-1_ocr-0/0524-211419/best_model_13.pth
#model_path=/media/prostate/20210315_LOWB/DM-Count_modify/DM-Count/ckpts/encoder-vgg19_bn_input-512_/0521-123619/best_model_31.pth
# ce loss + copy-paste
#model_path=/media/prostate/20210331_PDL1/segmentation/DM-Count/ckpts/encoder-se_resnext50_32x4d_input-512_downrate-1_scale_pyramid_module-1_attention_branch-0_albumentation-1_copy_paste-1_deep_supervision-1_ocr-0/0602-192246/best_model_5.pth
# nrdice loss + copy-paste
model_path=/media/prostate/20210331_PDL1/segmentation/DM-Count/ckpts/encoder-se_resnext50_32x4d_input-512_downrate-1_scale_pyramid_module-1_attention_branch-0_albumentation-1_copy_paste-1_deep_supervision-1_ocr-0/0603-021057/best_model_5.pth
## focal loss + copy-paste
#model_path=/media/prostate/20210331_PDL1/segmentation/DM-Count/ckpts/encoder-se_resnext50_32x4d_input-512_downrate-1_scale_pyramid_module-1_attention_branch-0_albumentation-1_copy_paste-1_deep_supervision-1_ocr-0/0603-090853/best_model_3.pth
## ce loss
#model_path=/media/prostate/20210331_PDL1/segmentation/DM-Count/ckpts/encoder-se_resnext50_32x4d_input-512_downrate-1_scale_pyramid_module-1_attention_branch-0_albumentation-1_copy_paste-0_deep_supervision-1_ocr-0/0602-000726/best_model_3.pth

#data_path=/media/prostate/20210315_LOWB/mmsegmentation/data/mydataset/0510_sample
#data_path=/media/prostate/20210331_PDL1/nuclei_detection/ndpi2roi_new/out/0525_PDL1/cancer_seg_x20_1024x1024/train/PDL1
#data_path=/media/prostate/20210331_PDL1/data/ROI/0525_PDL1/cancer_seg_x10_1024x1024/train/PDL1
#data_path=/media/prostate/20210331_PDL1/data/ROI/20210524forSatou/cancer_seg_x10_1024x1024/train/PDL1
#data_path=/media/prostate/20210331_PDL1/data/ROI/20210524forSatou/semi_supervised_seg_x10_512x512/train/PDL1
#data_path=/media/prostate/20210331_PDL1/data/ROI/20210524forSatou/semi_supervised_seg_x10_2048x2048/train/PDL1
#data_path=/media/prostate/20210331_PDL1/data/ROI/20210524forSatou/semi_supervised_seg_x10_1536x1536/train/PDL1
#data_path=/media/prostate/20210331_PDL1/data/ROI/20210524forSatou/test_seg_x10_2048x2048/train/PDL1
#data_path=/media/prostate/20210331_PDL1/data/ROI/20210428_forSatou2/dako準拠染色_日本スライド/test_seg_x10_2048x2048/train/dako準拠染色_日本スライド
data_path=/media/prostate/20210331_PDL1/data/ROI/20210405_forSatou/test_seg_x10_2048x2048/train/20210405_forSatou

dataset=segmentation
#pred_density_map_path=/media/prostate/20210315_LOWB/DM-Count_modify/DM-Count/ckpts/tmp/segmentation/0518/unet_vgg19_deepsupervision_copy_paste
#pred_density_map_path=/media/prostate/20210331_PDL1/segmentation/20210603_segmentation_forward/x10_2048x2048_se-resnext50_crossentropy_copy-paste
#pred_density_map_path=/media/prostate/20210331_PDL1/segmentation/20210603_segmentation_forward/x10_2048x2048_se-resnext50_nrdice_copy-paste
#pred_density_map_path=/media/prostate/20210331_PDL1/segmentation/20210603_segmentation_forward/x10_2048x2048_se-resnext50_focal_copy-paste
#pred_density_map_path=/media/prostate/20210331_PDL1/segmentation/20210603_segmentation_forward/x10_2048x2048_se-resnext50_crossentropy
#pred_density_map_path=/media/prostate/20210331_PDL1/segmentation/20210603_segmentation_forward/x10_1536x1536_se-resnext50_crossentropy_copy-paste
#pred_density_map_path=/media/prostate/20210331_PDL1/segmentation/20210603_segmentation_forward/x10_2048x2048_se-resnext50_nrdice_copy-paste
#pred_density_map_path=/media/prostate/20210331_PDL1/segmentation/20210603_segmentation_forward/20210428_forSatou2/dako準拠染色_日本スライド/x10_2048x2048_se-resnext50_nrdice_copy-paste
pred_density_map_path=/media/prostate/20210331_PDL1/segmentation/20210603_segmentation_forward/20210405_forSatou/x10_2048x2048_se-resnext50_nrdice_copy-paste

#encoder_name=vgg19_bn # dpn98 resnet152 vgg19_bn timm-resnest50d efficientnet-b5 timm-resnest50d_4s2x40d vgg19_bn mobilenet_v2 timm-efficientnet-lite4 timm-skresnext50_32x4d se_resnext50_32x4d timm-efficientnet-b6 se_resnext101_32x4d xception 
encoder_name=se_resnext50_32x4d
#encoder_name=hrnet_seg_ocr # dpn98 resnet152 vgg19_bn timm-resnest50d efficientnet-b5 timm-resnest50d_4s2x40d vgg19_bn mobilenet_v2 timm-efficientnet-lite4 timm-skresnext50_32x4d se_resnext50_32x4d timm-efficientnet-b6 se_resnext101_32x4d xception 

input-size=2048
crop_size=2048
downsample_ratio=1
activation=identity # Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and **None**.
deep_supervision=1
#deep_supervision=0
cfg=/media/prostate/20210315_LOWB/DM-Count_modify/DM-Count/models/hrnet_seg_ocr/experiments/test.yaml
#test_type=val
test_type=test_no_gt
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
--deep_supervision $deep_supervision \
--cfg $cfg \
--pred-density-map-path $pred_density_map_path




