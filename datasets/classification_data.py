from PIL import Image
import torch.utils.data as data
import os
from glob import glob
import torch
import torchvision.transforms.functional as F
from torchvision import transforms
import random
import numpy as np
import pandas as pd
import albumentations as A
from .augment_and_mix import augment_and_mix
from pathlib import Path


def random_crop(im_h, im_w, crop_h, crop_w):
    res_h = im_h - crop_h
    res_w = im_w - crop_w
    i = random.randint(0, res_h)
    j = random.randint(0, res_w)
    return i, j, crop_h, crop_w

class Base(data.Dataset):
    def __init__(self, root_path_img, crop_size=330):

        self.root_path_img = root_path_img
        self.c_size = crop_size
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass

    def train_transform(self, img, label):
        pass


class ClassificationDataset(Base):
    def __init__(self, root_path_img, crop_size=330, method='train', flag_csv=None, use_albumentation=0, balanced_over_samping=False, sample_fraction=None):
        super().__init__(root_path_img, crop_size)
        self.method = method
        if method not in ['train', 'val', 'test']:
            raise Exception("not implement")

        self.im_list = sorted(list(Path(self.root_path_img).glob("**/*.jpg")))
        self.label_list = [int(p.stem.rsplit("_")[-1]) for p in self.im_list]

        if flag_csv is None:
            "flag csv not specified. exit"
            exit()

        use_columns = ['filename_head', 'train_flag', 'val_flag', 'test_flag', 'unuse_flag']
        df_flag_csv_concat = pd.DataFrame(columns=use_columns)

        for csv in flag_csv:
            df_csv = pd.read_csv(csv, encoding="utf-8")
            df_csv = df_csv.loc[:, use_columns]
            df_flag_csv_concat = pd.concat([df_flag_csv_concat, df_csv])

        # check if duplicated
        duplicated = df_flag_csv_concat.loc[df_flag_csv_concat.filename_head.duplicated()]
        if not duplicated.empty:
            print("duplicated needle in flag_csv.")
            print(f"{duplicated}")
            print("exit.")
            exit()

        df_flag_csv_concat = df_flag_csv_concat.loc[~(df_flag_csv_concat.unuse_flag == 1)]

        if method == 'train':
            target_slides = df_flag_csv_concat.loc[(df_flag_csv_concat.train_flag == 1), "filename_head"].tolist()
        if method == 'val':
            target_slides = df_flag_csv_concat.loc[(df_flag_csv_concat.val_flag == 1), "filename_head"].tolist()
        if method == 'test':
            target_slides = df_flag_csv_concat.loc[(df_flag_csv_concat.test_flag == 1), "filename_head"].tolist()

        self.im_list = [im for im in self.im_list if im.name.split("_")[0] in target_slides]
        self.label_list = [int(p.stem.rsplit("_")[-1]) for p in self.im_list]
        if (sample_fraction is not None) and (method == 'train'): # 実験用に学習データの一部だけをサンプリングして学習
            print(f"use sample_fraction : {sample_fraction}")
            random.seed(0)
            self.im_list = random.sample(self.im_list, int(len(self.im_list)*sample_fraction))
            self.label_list = [int(p.stem.rsplit("_")[-1]) for p in self.im_list]

        print(f'number of img {method}: {len(self.im_list)}')
        df = pd.DataFrame(self.im_list, columns=["filename_head"])
        df["label"] = df.filename_head.apply(lambda x: int(x.stem.rsplit("_")[-1]))
        vc = df.label.value_counts().sort_index()
        print(vc)

        if balanced_over_samping:
            # Balanced Over sampling
            group = df.groupby('label')
            df_bo_sampled = group.apply(lambda x: x.sample(n=group.size().max(), random_state=0, replace=True)).reset_index(drop=True)
            self.im_list = df_bo_sampled.filename_head.tolist()
            self.label_list = [int(p.stem.rsplit("_")[-1]) for p in self.im_list]
            print(f'number of img balanced over sampled {method}: {len(self.im_list)}')
            vc = df_bo_sampled.label.value_counts().sort_index()
            print(vc)

        self.use_albumentation = use_albumentation
        if use_albumentation:
            self.albumentations = A.Compose([
#                                      A.HorizontalFlip(p=0.5),
#                                      #A.RandomCrop(width=768, height=768, p=0.5),
                                      A.VerticalFlip(p=0.5),
                                      A.Rotate(p=0.5),
#                                      #A.CenterCrop(height=512, width=512, p=0.5),
#                                      A.ShiftScaleRotate(p=0.5),
#                                      A.RandomSizedCrop(min_max_height=(int(crop_size / 2), crop_size), height=crop_size, width=crop_size, p=0.5),
#                                      A.OneOf([
#                                          A.HueSaturationValue(p=0.5), 
#                                          A.RGBShift(p=0.7)
#                                      ], p=1),                          
#                                      A.RandomBrightnessContrast(p=0.5),
#                                      A.CLAHE(p=0.8),
#                                       A.RGBShift(p=0.5),
                                       A.RandomGamma(p=0.5),
                                       A.GaussianBlur(p=0.5),
                                       A.GaussNoise(p=0.5),
#                                      A.Cutout(p=0.5),
                                   ])  

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):
        img_path = self.im_list[item]
        img = Image.open(img_path).convert('RGB')
        if self.label_list:
            label = self.label_list[item]
        if self.method == 'train':
            img, label = self.train_transform(img, label)
            return img.type(self.inputs_dtype), label.type(self.targets_dtype)
        elif self.method == 'val':
            img, label =  self.trans(img), torch.as_tensor(np.array(label))
            return img.type(self.inputs_dtype), label.type(self.targets_dtype)
        elif self.method == 'test':
            img, label =  self.trans(img), torch.as_tensor(np.array(label))
            return img.type(self.inputs_dtype), label.type(self.targets_dtype), str(img_path)
        else:
            raise


    def train_transform(self, img, label):
        wd, ht = img.size
        st_size = 1.0 * min(wd, ht)
        # resize the image to fit the crop size
        if st_size < self.c_size:
            rr = 1.0 * self.c_size / st_size
            wd = round(wd * rr)
            ht = round(ht * rr)
            st_size = 1.0 * min(wd, ht)
            img = img.resize((wd, ht), Image.BICUBIC)
        assert st_size >= self.c_size, print(wd, ht)
        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        img = F.crop(img, i, j, h, w)

        if self.use_albumentation:
            transformed = self.albumentations(image=np.array(img))
            img = transformed['image']
            img = augment_and_mix(np.float32(img)/255, severity=10)
            img = np.uint8((img+1)/2*255)

        if not isinstance(img, (np.ndarray, np.generic)):
            img = np.array(img)

        img = Image.fromarray(img)

        if random.random() > 0.5:
            img = F.hflip(img)

        return self.trans(img), torch.as_tensor(np.array(label))
 