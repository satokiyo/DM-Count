#!/bin/bash
#model_path=/media/prostate/20210315_LOWB/DM-Count_modify/DM-Count/ckpts/encoder-vgg19_bn_input-512_wot-0.1_wtv-0.01_reg-10.0_nIter-100_normCood-0/0420-233633/best_model_2.pth
#model_path=/media/prostate/20210315_LOWB/DM-Count_modify/DM-Count/ckpts/encoder-vgg19_bn_input-512_wot-0.1_wtv-0.01_reg-10.0_nIter-100_normCood-0/0421-141049/best_model_5.pth
#model_path=/media/prostate/20210315_LOWB/DM-Count_modify/DM-Count/ckpts/encoder-vgg19_bn_input-512_wot-0.1_wtv-0.01_reg-10.0_nIter-100_normCood-0/0421-012301/best_model_1.pth
#down2
#model_path=/media/prostate/20210315_LOWB/DM-Count_modify/DM-Count/ckpts/encoder-vgg19_bn_input-512_wot-0.1_wtv-0.01_reg-10.0_nIter-100_normCood-0/0424-183215/best_model_9.pth
#model_path=/media/prostate/20210315_LOWB/DM-Count_modify/DM-Count/ckpts/encoder-vgg19_bn_input-512_wot-0.1_wtv-0.01_reg-10.0_nIter-100_normCood-0/0424-210910/best_model_9.pth
#model_path=/media/prostate/20210315_LOWB/DM-Count_modify/DM-Count/ckpts/encoder-vgg19_bn_input-512_wot-0.1_wtv-0.01_reg-10.0_nIter-100_normCood-0/0426-120819/best_model_4.pth
# resize 256
#model_path=/media/prostate/20210315_LOWB/DM-Count_modify/DM-Count/ckpts/encoder-vgg19_bn_input-512_wot-0.1_wtv-0.01_reg-10.0_nIter-100_normCood-0/0427-104406/best_model_9.pth
#model_path=/media/prostate/20210315_LOWB/DM-Count_modify/DM-Count/ckpts/encoder-vgg19_bn_input-512_wot-0.1_wtv-0.01_reg-10.0_nIter-100_normCood-0/0428-155953/best_model_7.pth
#model_path=/media/prostate/20210315_LOWB/DM-Count_modify/DM-Count/ckpts/encoder-vgg19_bn_input-512_wot-0.1_wtv-0.01_reg-10.0_nIter-300_normCood-0/0428-224552/best_model_9.pth

#model_path=/media/prostate/20210315_LOWB/DM-Count_modify/DM-Count/ckpts/encoder-vgg19_bn_input-512_wot-0.1_wtv-0.01_reg-1.0_nIter-300_normCood-0/0429-012910/best_model_6.pth

#model_path=/media/prostate/20210315_LOWB/DM-Count_modify/DM-Count/ckpts/encoder-vgg19_bn_input-512_wot-0.1_wtv-0.01_reg-0.1_nIter-300_normCood-0/0429-041247/best_model_9.pth
#model_path=/media/prostate/20210315_LOWB/DM-Count_modify/DM-Count/ckpts/encoder-vgg19_bn_input-512_wot-0.1_wtv-0.01_reg-0.01_nIter-300_normCood-0/0429-065516/best_model_5.pth
#model_path=/media/prostate/20210315_LOWB/DM-Count_modify/DM-Count/ckpts/encoder-vgg19_bn_input-512_wot-0.1_wtv-0.01_reg-100.0_nIter-300_normCood-0/0429-093939/best_model_7.pth
#model_path=/media/prostate/20210315_LOWB/DM-Count_modify/DM-Count/ckpts/encoder-vgg19_bn_input-512_wot-0.1_wtv-0.1_reg-10.0_nIter-300_normCood-0/0429-122111/best_model_9.pth
#model_path=/media/prostate/20210315_LOWB/DM-Count_modify/DM-Count/ckpts/encoder-vgg19_bn_input-512_wot-1.0_wtv-0.01_reg-10.0_nIter-300_normCood-0/0429-150519/best_model_9.pth
#model_path=/media/prostate/20210315_LOWB/DM-Count_modify/DM-Count/ckpts/encoder-vgg19_bn_input-512_wot-1.0_wtv-0.1_reg-10.0_nIter-300_normCood-0/0429-174912/best_model_7.pth
#model_path=/media/prostate/20210315_LOWB/DM-Count_modify/DM-Count/ckpts/encoder-vgg19_bn_input-512_wot-0.01_wtv-0.001_reg-10.0_nIter-300_normCood-0/0429-203405/best_model_11.pth
#model_path=/media/prostate/20210315_LOWB/DM-Count_modify/DM-Count/ckpts/encoder-vgg19_bn_input-512_wot-0.1_wtv-0.1_reg-0.5_nIter-300_normCood-0/0501-114117/best_model_18.pth
#model_path=/media/prostate/20210315_LOWB/DM-Count_modify/DM-Count/ckpts/encoder-vgg19_bn_input-512_wot-0.1_wtv-0.1_reg-1.0_nIter-300_normCood-0/0501-142718/best_model_14.pth
#model_path=/media/prostate/20210315_LOWB/DM-Count_modify/DM-Count/ckpts/encoder-vgg19_bn_input-512_wot-0.1_wtv-0.1_reg-2.0_nIter-300_normCood-0/0501-171310/best_model_7.pth
#model_path=/media/prostate/20210315_LOWB/DM-Count_modify/DM-Count/ckpts/encoder-vgg19_bn_input-512_wot-0.1_wtv-0.1_reg-3.0_nIter-300_normCood-0/0501-195959/best_model_8.pth
#model_path=/media/prostate/20210315_LOWB/DM-Count_modify/DM-Count/ckpts/encoder-vgg19_bn_input-512_wot-0.1_wtv-0.1_reg-4.0_nIter-300_normCood-0/0501-224637/best_model_12.pth
#model_path=/media/prostate/20210315_LOWB/DM-Count_modify/DM-Count/ckpts/encoder-vgg19_bn_input-512_wot-0.1_wtv-0.1_reg-5.0_nIter-300_normCood-0/0502-013352/best_model_9.pth
#model_path=/media/prostate/20210315_LOWB/DM-Count_modify/DM-Count/ckpts/encoder-vgg19_bn_input-512_wot-0.1_wtv-0.05_reg-1.0_nIter-300_normCood-0/0502-042022/best_model_13.pth
#model_path=/media/prostate/20210315_LOWB/DM-Count_modify/DM-Count/ckpts/encoder-vgg19_bn_input-512_wot-0.1_wtv-0.5_reg-1.0_nIter-300_normCood-0/0502-070505/best_model_13.pth
#model_path=/media/prostate/20210315_LOWB/DM-Count_modify/DM-Count/ckpts/encoder-vgg19_bn_input-512_wot-0.1_wtv-0.7_reg-1.0_nIter-300_normCood-0/0502-221410/best_model_13.pth
#model_path=/media/prostate/20210315_LOWB/DM-Count_modify/DM-Count/ckpts/encoder-vgg19_bn_input-512_wot-0.1_wtv-0.9_reg-1.0_nIter-300_normCood-0/0503-010041/best_model_8.pth
model_path=/media/prostate/20210315_LOWB/DM-Count_modify/DM-Count/ckpts/encoder-vgg19_bn_input-512_wot-0.1_wtv-0.7_reg-1.0_nIter-300_normCood-0/0505-111316/best_model_10.pth
# kiunet
#model_path=/media/prostate/20210315_LOWB/DM-Count_modify/DM-Count/ckpts/encoder-vgg19_bn_input-256_wot-0.1_wtv-0.01_reg-10.0_nIter-100_normCood-0/0425-034922/best_model_5.pth
# hrnet
#model_path=/media/prostate/20210315_LOWB/DM-Count_modify/DM-Count/ckpts/encoder-vgg19_bn_input-512_wot-0.1_wtv-0.01_reg-10.0_nIter-300_normCood-0/0430-203224/best_model_6.pth
#model_path=/media/prostate/20210315_LOWB/DM-Count_modify/DM-Count/ckpts/encoder-seg_hrnet_input-512_wot-0.1_wtv-0.01_reg-5.0_nIter-300_normCood-0/0501-032854/best_model_6.pth
#model_path=/media/prostate/20210315_LOWB/DM-Count_modify/DM-Count/ckpts/encoder-seg_hrnet_input-512_wot-0.1_wtv-10.0_reg-10.0_nIter-300_normCood-0/0501-082511/best_model_4.pth
#model_path=/media/prostate/20210315_LOWB/DM-Count_modify/DM-Count/ckpts/encoder-seg_hrnet_input-512_wot-0.1_wtv-0.01_reg-10.0_nIter-300_normCood-0/0501-003846/best_model_8.pth

data_path=/media/prostate/20210331_PDL1/nuclei_detection/YOLO/darknet/cfg/task/datasets
dataset=cell #<dataset name: qnrf, sha, shb or nwpu>\
pred_density_map_path=/media/prostate/20210315_LOWB/DM-Count_modify/DM-Count/ckpts/tmp/0514_BN_DAB_thres92_val_vgg19_res512_CEL1_332_reg1_wot0.1_wtv0.7
#test_type=test_no_gt # 'val' 'val_with_gt' 'test_no_gt'
#test_type=val_with_gt # 'val' 'val_with_gt' 'test_no_gt'
test_type=val # 'val' 'val_with_gt' 'test_no_gt'
encoder_name=vgg19_bn # dpn98 resnet152 vgg19_bn timm-resnest50d efficientnet-b5 timm-resnest50d_4s2x40d vgg19_bn mobilenet_v2 timm-efficientnet-lite4 timm-skresnext50_32x4d se_resnext50_32x4d timm-efficientnet-b6 se_resnext101_32x4d xception 
downsample_ratio=1
#crop_size=512
crop_size=256
cfg=/media/prostate/20210315_LOWB/DM-Count_modify/DM-Count/models/hrnet/experiments/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml


python3 test.py \
--model-path $model_path \
--data-path $data_path \
--dataset $dataset \
--crop-size $crop_size  \
--test-type $test_type \
--encoder_name $encoder_name \
--classes 1 \
--scale_pyramid_module 1 \
--use_attention_branch 0 \
--downsample-ratio $downsample_ratio \
--cfg $cfg \
--pred-density-map-path $pred_density_map_path




