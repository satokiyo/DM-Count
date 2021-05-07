#!/bin/bash
#datasetname=shb # qnrf, sha, shb or nwpu
datasetname=cell
#datadir=/media/HDD2/count_dataset/ShanghaiTech/part_B #  <path to dataset>
#datadir=/media/HDD2/count_dataset/ShanghaiTech/part_B_100images
datadir=/media/prostate/20210331_PDL1/YOLO/darknet/cfg/task/datasets
#batch_size=4 理想
batch_size=1
device='0'
num_workers=4
crop_size=512


python3 train.py \
--dataset $datasetname \
--data-dir $datadir \
--device 0 \
--batch-size $batch_size \
--num-workers $num_workers \
--crop-size $crop_size \
--encoder_name vgg19_bn \
--classes 1 \
--use_albumentation 1 \
--data_check 1