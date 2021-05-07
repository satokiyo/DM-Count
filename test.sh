#!/bin/bash
model_path=/media/prostate/20210315_LOWB/DM-Count/DM-Count/ckpts/input-512_wot-0.1_wtv-0.01_reg-10.0_nIter-100_normCood-0/best_model_5.pth

data_path=/media/HDD2/count_dataset/ShanghaiTech/part_B #  <path to dataset>
dataset=shb #<dataset name: qnrf, sha, shb or nwpu>\
crop_size=512
#crop_size=256
pred_density_map_path=/media/prostate/20210315_LOWB/DM-Count/DM-Count/ckpts/input-256_wot-0.1_wtv-0.01_reg-10.0_nIter-100_normCood-0/tmp

python3 test.py \
--model-path $model_path \
--data-path $data_path \
--dataset $dataset \
--crop-size $crop_size  \
--pred-density-map-path $pred_density_map_path

