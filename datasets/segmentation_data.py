from PIL import Image
import torch.utils.data as data
import os
from glob import glob
import torch
import torchvision.transforms.functional as F
from torchvision import transforms
import random
import numpy as np
import scipy.io as sio
import albumentations as A
from .augment_and_mix import augment_and_mix
from pathlib import Path
from .copy_paste import copy_paste_class, CopyPaste



def random_crop(im_h, im_w, crop_h, crop_w):
    res_h = im_h - crop_h
    res_w = im_w - crop_w
    i = random.randint(0, res_h)
    j = random.randint(0, res_w)
    return i, j, crop_h, crop_w

class Base(data.Dataset):
    def __init__(self, root_path_img, root_path_ano=None, crop_size=512, downsample_ratio=1):

        self.root_path_img = root_path_img
        self.root_path_ano = root_path_ano
        self.c_size = crop_size
        self.d_ratio = downsample_ratio
        assert self.c_size % self.d_ratio == 0
        self.dc_size = self.c_size // self.d_ratio
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

    def train_transform(self, img, gt):
        pass


@copy_paste_class
class SegDataset(Base):
    def __init__(self, root_path_img, root_path_ano=None, crop_size=512,
                 downsample_ratio=1,
                 method='train',
                 use_albumentation=0,
                 use_copy_paste=0,):
        super().__init__(root_path_img, root_path_ano, crop_size, downsample_ratio)
        self.method = method
        if method not in ['train', 'val', 'val_no_gt', 'test_no_gt']:
            raise Exception("not implement")

        if method in ['train', 'val']: # with gt
            self.im_list = sorted(list(Path(self.root_path_img).glob("**/*.jpg")))
            if root_path_ano is not None:
                self.gt_list = [Path(self.root_path_ano) / p.name.replace("jpg", "png") for p in self.im_list]
            else:
                raise Exception("specify gt dir")
        elif method in ['val_no_gt', 'test_no_gt']: # without gt
            self.im_list = sorted(list(Path(self.root_path_img).glob("**/*.jpg")))
            self.gt_list = None

        print('number of img: {}'.format(len(self.im_list)))

        self.use_copy_paste = use_copy_paste
        if use_copy_paste:
            self.copy_paste_aug = A.Compose([
                                       CopyPaste(blend=True, sigma=1, pct_objects_paste=0.8, p=0.8), #pct_objects_paste is a guess
                                   ])  

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
#                                   keypoint_params=A.KeypointParams(format='xy'))

    def __len__(self):
        return len(self.im_list)

    def load_example(self, item):
#    def __getitem__(self, item):
        img_path = self.im_list[item]
        img = Image.open(img_path).convert('RGB')
        if self.gt_list:
            gt_path = self.gt_list[item]
            gt = Image.open(gt_path) # 1-ch index color image
        
        """
        The y is supposed to have a shape [N, H, W] , (this is because torch’s loss function torch.nn.CrossEntropyLoss expects it, even though it will internally one-hot encode it).
        or pixelwise argmax() to convert from one-hot to index color
        """

        if self.method == 'train':
            img, gt = self.train_transform(img, gt)
            return img.type(self.inputs_dtype), gt.type(self.targets_dtype)
        elif self.method == 'val':
            img, gt =  self.trans(img), torch.as_tensor(np.array(gt))
            return img.type(self.inputs_dtype), gt.type(self.targets_dtype)
        elif self.method in ['val_no_gt', 'test_no_gt']:
            img = self.trans(img)
            return img.type(self.inputs_dtype)
        else:
            raise


    def train_transform(self, img, gt):
        wd, ht = img.size
        st_size = 1.0 * min(wd, ht)
        # resize the image to fit the crop size
        if st_size < self.c_size:
            rr = 1.0 * self.c_size / st_size
            wd = round(wd * rr)
            ht = round(ht * rr)
            st_size = 1.0 * min(wd, ht)
            img = img.resize((wd, ht), Image.BICUBIC)
            gt = gt.resize((wd, ht), Image.BICUBIC)
        assert st_size >= self.c_size, print(wd, ht)
        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        img = F.crop(img, i, j, h, w)
        gt = F.crop(gt, i, j, h, w)

        if self.use_copy_paste:
            transformed = self.copy_paste_aug(image=np.array(img), mask=np.array(gt))
            img, gt = transformed['image'], transformed['mask']
 
        if self.use_albumentation:
            transformed = self.albumentations(image=np.array(img), mask=np.array(gt))
            img, gt = transformed['image'], transformed['mask']
            img = augment_and_mix(np.float32(img)/255, severity=10)
            img = np.uint8((img+1)/2*255)

        img = Image.fromarray(img)
        gt = Image.fromarray(np.uint8(gt))

        if random.random() > 0.5:
            img = F.hflip(img)
            gt = F.hflip(gt)

        return self.trans(img), torch.as_tensor(np.array(gt))
 