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


def random_crop(im_h, im_w, crop_h, crop_w):
    res_h = im_h - crop_h
    res_w = im_w - crop_w
    i = random.randint(0, res_h)
    j = random.randint(0, res_w)
    return i, j, crop_h, crop_w


def cal_innner_area(c_left, c_up, c_right, c_down, bbox):
    inner_left = np.maximum(c_left, bbox[:, 0])
    inner_up = np.maximum(c_up, bbox[:, 1])
    inner_right = np.minimum(c_right, bbox[:, 2])
    inner_down = np.minimum(c_down, bbox[:, 3])
    inner_area = np.maximum(inner_right-inner_left, 0.0) * np.maximum(inner_down-inner_up, 0.0)
    return inner_area


def gen_discrete_map(im_height, im_width, points):
    """
        func: generate the discrete map.
        points: [num_gt, 2], for each row: [width, height]
        """
    discrete_map = np.zeros([im_height, im_width], dtype=np.float32)
    h, w = discrete_map.shape[:2]
    num_gt = points.shape[0]
    if num_gt == 0:
        return discrete_map
    
    # fast create discrete map
    points_np = np.array(points).round().astype(int)
    p_h = np.minimum(points_np[:, 1], np.array([h-1]*num_gt).astype(int))
    p_w = np.minimum(points_np[:, 0], np.array([w-1]*num_gt).astype(int))
    p_index = torch.from_numpy(p_h* im_width + p_w)
    discrete_map = torch.zeros(im_width * im_height).scatter_add_(0, index=p_index, src=torch.ones(im_width*im_height)).view(im_height, im_width).numpy()

    ''' slow method
    for p in points:
        p = np.round(p).astype(int)
        p[0], p[1] = min(h - 1, p[1]), min(w - 1, p[0])
        discrete_map[p[0], p[1]] += 1
    '''
    assert np.sum(discrete_map) == num_gt
    return discrete_map


class Base(data.Dataset):
    def __init__(self, root_path, crop_size, downsample_ratio=8):

        self.root_path = root_path
        self.c_size = crop_size
        self.d_ratio = downsample_ratio
        assert self.c_size % self.d_ratio == 0
        self.dc_size = self.c_size // self.d_ratio
        self.trans = transforms.Compose([
            transforms.ToTensor(),
#            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass

    def train_transform(self, img, keypoints):
        wd, ht = img.size
        st_size = 1.0 * min(wd, ht)
        assert st_size >= self.c_size
        assert len(keypoints) >= 0
        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        img = F.crop(img, i, j, h, w)
        if len(keypoints) > 0:
            keypoints = keypoints - [j, i]
            idx_mask = (keypoints[:, 0] >= 0) * (keypoints[:, 0] <= w) * \
                       (keypoints[:, 1] >= 0) * (keypoints[:, 1] <= h)
            keypoints = keypoints[idx_mask]
        else:
            keypoints = np.empty([0, 2])

        gt_discrete = gen_discrete_map(h, w, keypoints)
        down_w = w // self.d_ratio
        down_h = h // self.d_ratio
        gt_discrete = gt_discrete.reshape([down_h, self.d_ratio, down_w, self.d_ratio]).sum(axis=(1, 3))
        assert np.sum(gt_discrete) == len(keypoints)

        if len(keypoints) > 0:
            if random.random() > 0.5:
                img = F.hflip(img)
                gt_discrete = np.fliplr(gt_discrete)
                keypoints[:, 0] = w - keypoints[:, 0]
        else:
            if random.random() > 0.5:
                img = F.hflip(img)
                gt_discrete = np.fliplr(gt_discrete)
        gt_discrete = np.expand_dims(gt_discrete, 0)

        return self.trans(img), torch.from_numpy(keypoints.copy()).float(), torch.from_numpy(
            gt_discrete.copy()).float()


class Crowd_qnrf(Base):
    def __init__(self, root_path, crop_size,
                 downsample_ratio=8,
                 method='train'):
        super().__init__(root_path, crop_size, downsample_ratio)
        self.method = method
        self.im_list = sorted(glob(os.path.join(self.root_path, '*.jpg')))
        print('number of img: {}'.format(len(self.im_list)))
        if method not in ['train', 'val']:
            raise Exception("not implement")

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):
        img_path = self.im_list[item]
        gd_path = img_path.replace('jpg', 'npy')
        img = Image.open(img_path).convert('RGB')
        if self.method == 'train':
            keypoints = np.load(gd_path)
            return self.train_transform(img, keypoints)
        elif self.method == 'val':
            keypoints = np.load(gd_path)
            img = self.trans(img)
            name = os.path.basename(img_path).split('.')[0]
            return img, len(keypoints), name


class Crowd_nwpu(Base):
    def __init__(self, root_path, crop_size,
                 downsample_ratio=8,
                 method='train'):
        super().__init__(root_path, crop_size, downsample_ratio)
        self.method = method
        self.im_list = sorted(glob(os.path.join(self.root_path, '*.jpg')))
        print('number of img: {}'.format(len(self.im_list)))

        if method not in ['train', 'val', 'test']:
            raise Exception("not implement")

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):
        img_path = self.im_list[item]
        gd_path = img_path.replace('jpg', 'npy')
        img = Image.open(img_path).convert('RGB')
        if self.method == 'train':
            keypoints = np.load(gd_path)
            return self.train_transform(img, keypoints)
        elif self.method == 'val':
            keypoints = np.load(gd_path)
            img = self.trans(img)
            name = os.path.basename(img_path).split('.')[0]
            return img, len(keypoints), name
        elif self.method == 'test':
            img = self.trans(img)
            name = os.path.basename(img_path).split('.')[0]
            return img, name


class Crowd_sh(Base):
    def __init__(self, root_path, crop_size,
                 downsample_ratio=8,
                 method='train',
                 use_albumentation=0):
        super().__init__(root_path, crop_size, downsample_ratio)
        self.method = method
        if method not in ['train', 'val']:
            raise Exception("not implement")

        self.im_list = sorted(glob(os.path.join(self.root_path, 'images', '*.jpg')))
        print('number of img: {}'.format(len(self.im_list)))

        self.use_albumentation = use_albumentation
        if use_albumentation:
            self.albumentations = A.Compose([
#                                      A.HorizontalFlip(p=0.5),
#                                      #A.RandomCrop(width=768, height=768, p=0.5),
                                      A.VerticalFlip(p=0.5),
                                      A.Rotate(p=0.5),
#                                      #A.CenterCrop(height=512, width=512, p=0.5),
                                      A.ShiftScaleRotate(p=0.5),
                                      A.RandomSizedCrop(min_max_height=(256, 512), height=512, width=512, p=0.5),
                                      A.OneOf([
                                          A.HueSaturationValue(p=0.5), 
                                          A.RGBShift(p=0.7)
                                      ], p=1),                          
                                      A.RandomBrightnessContrast(p=0.5),
                                      A.CLAHE(p=0.8),
                                      A.RandomBrightnessContrast(p=0.8),    
                                      A.RandomGamma(p=0.8),
                                      A.GaussianBlur(p=0.5),
                                      A.GaussNoise(p=0.5),
                                      A.Cutout(p=0.5),
                                   ],  
                                   keypoint_params=A.KeypointParams(format='xy'))

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):
        img_path = self.im_list[item]
        name = os.path.basename(img_path).split('.')[0]
        gd_path = os.path.join(self.root_path, 'ground-truth', 'GT_{}.mat'.format(name))
        img = Image.open(img_path).convert('RGB')
        keypoints = sio.loadmat(gd_path)['image_info'][0][0][0][0][0]

        if self.method == 'train':
            return self.train_transform(img, keypoints)
        elif self.method == 'val':
            img = self.trans(img)
            return img, len(keypoints), name

    def train_transform(self, img, keypoints):
        wd, ht = img.size
        st_size = 1.0 * min(wd, ht)
        # resize the image to fit the crop size
        if st_size < self.c_size:
            rr = 1.0 * self.c_size / st_size
            wd = round(wd * rr)
            ht = round(ht * rr)
            st_size = 1.0 * min(wd, ht)
            img = img.resize((wd, ht), Image.BICUBIC)
            keypoints = keypoints * rr
        assert st_size >= self.c_size, print(wd, ht)
        assert len(keypoints) >= 0
        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        img = F.crop(img, i, j, h, w)
        if len(keypoints) > 0:
            keypoints = keypoints - [j, i]
            idx_mask = (keypoints[:, 0] >= 0) * (keypoints[:, 0] <= w) * \
                       (keypoints[:, 1] >= 0) * (keypoints[:, 1] <= h)
            keypoints = keypoints[idx_mask]
        else:
            keypoints = np.empty([0, 2])

        if self.use_albumentation:
#            transformed = self.albumentations(image=np.array(img), keypoints=keypoints)
#            img, keypoints = transformed['image'], transformed['keypoints']
            img = augment_and_mix(np.float32(img)/255, severity=10)
            img = Image.fromarray(np.uint8((img+1)/2*255))
            keypoints = np.array(keypoints)

        gt_discrete = gen_discrete_map(h, w, keypoints)
        down_w = w // self.d_ratio
        down_h = h // self.d_ratio
        gt_discrete = gt_discrete.reshape([down_h, self.d_ratio, down_w, self.d_ratio]).sum(axis=(1, 3))
        assert np.sum(gt_discrete) == len(keypoints)

        if len(keypoints) > 0:
            if random.random() > 0.5:
                img = F.hflip(img)
                gt_discrete = np.fliplr(gt_discrete)
                keypoints[:, 0] = w - keypoints[:, 0] - 1
        else:
            if random.random() > 0.5:
                img = F.hflip(img)
                gt_discrete = np.fliplr(gt_discrete)
        gt_discrete = np.expand_dims(gt_discrete, 0)

        return self.trans(img), torch.from_numpy(keypoints.copy()).float(), torch.from_numpy(
            gt_discrete.copy()).float()


class CellDataset(Base):
    def __init__(self, root_path, crop_size,
                 downsample_ratio=8,
                 method='train',
                 use_albumentation=0):
        super().__init__(root_path, crop_size, downsample_ratio)
        self.method = method
        if method not in ['train', 'val', 'val_with_gt', 'test_no_gt']:
            raise Exception("not implement")

        if method in ['train', 'val', 'val_with_gt']: # with gt
            self.gt_list = sorted(list(Path(self.root_path).glob("**/*.mat")))
            im_list = []
            for gt in self.gt_list:
                img_path = str(gt).split('.')[0] + '.jpg'
                #img_path = str(gt).split('.')[0] + '.jpg_pred.jpg'
                im_list.append(img_path)
            self.im_list = im_list
        elif method ==  'test_no_gt': # without gt
            self.im_list = sorted(list(Path(self.root_path).glob("**/*.jpg")))

        print('number of img: {}'.format(len(self.im_list)))


        self.use_albumentation = use_albumentation
        if use_albumentation:
            self.albumentations = A.Compose([
        # 20210428 tmp
        #                               A.Resize(256,256, p=1.0),
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
                                   ],  
                                   keypoint_params=A.KeypointParams(format='xy'))

        # 20210428 tmp
        #self.resize = A.Compose([
        #                               A.Resize(256,256, p=1.0)],
        #                           keypoint_params=A.KeypointParams(format='xy'))





    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):
        img_path = self.im_list[item]
        name = str(os.path.basename(img_path)).split('.')[0]
        #gd_path = self.gt_list[item]
        gd_path = str(img_path).split('.')[0] + '.mat'
        img = Image.open(img_path).convert('RGB')

        if self.method == 'train':
            keypoints = sio.loadmat(gd_path)['image_info']
            return self.train_transform(img, keypoints)
        elif self.method == 'val':
            keypoints = sio.loadmat(gd_path)['image_info']
            ## 20210428 tmp
            #img, keypoints = self.val_transform(img, keypoints)
            img = self.trans(img)
            #return img, len(keypoints), name
            return img, torch.from_numpy(np.array(keypoints).copy()).float(), name
            #return img, keypoints, name
        elif self.method == 'val_with_gt':
            keypoints = sio.loadmat(gd_path)['image_info']
            img = self.trans(img)
            #return img, keypoints, name
            return img, torch.from_numpy(np.array(keypoints).copy()).float()
        elif self.method == 'test_no_gt':
            img = self.trans(img)
            return img, name
        else:
            raise


    def train_transform(self, img, keypoints):
        wd, ht = img.size
        st_size = 1.0 * min(wd, ht)
        # resize the image to fit the crop size
        if st_size < self.c_size:
            rr = 1.0 * self.c_size / st_size
            wd = round(wd * rr)
            ht = round(ht * rr)
            st_size = 1.0 * min(wd, ht)
            img = img.resize((wd, ht), Image.BICUBIC)
            keypoints = keypoints * rr
        assert st_size >= self.c_size, print(wd, ht)
        assert len(keypoints) >= 0
        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        img = F.crop(img, i, j, h, w)
        if len(keypoints) > 0:
            keypoints = keypoints - [j, i]
            idx_mask = (keypoints[:, 0] >= 0) * (keypoints[:, 0] <= w) * \
                       (keypoints[:, 1] >= 0) * (keypoints[:, 1] <= h)
            keypoints = keypoints[idx_mask]
        else:
            keypoints = np.empty([0, 2])

        keypoints = np.clip(keypoints, None, self.c_size-1e-5) # add

        if self.use_albumentation:
            transformed = self.albumentations(image=np.array(img), keypoints=keypoints)
            img, keypoints = transformed['image'], transformed['keypoints']
#            img = Image.fromarray(np.uint8((img)))
            img = augment_and_mix(np.float32(img)/255, severity=10)
            img = Image.fromarray(np.uint8((img+1)/2*255))
            keypoints = np.array(keypoints)

        ## 20210428 tmp
        #w = int(w/2)
        #h = int(h/2)
        gt_discrete = gen_discrete_map(h, w, keypoints)
        down_w = w // self.d_ratio
        down_h = h // self.d_ratio
        gt_discrete = gt_discrete.reshape([down_h, self.d_ratio, down_w, self.d_ratio]).sum(axis=(1, 3))
        assert np.sum(gt_discrete) == len(keypoints)

        if len(keypoints) > 0:
            if random.random() > 0.5:
                img = F.hflip(img)
                gt_discrete = np.fliplr(gt_discrete)
                keypoints[:, 0] = w - keypoints[:, 0] - 1
        else:
            if random.random() > 0.5:
                img = F.hflip(img)
                gt_discrete = np.fliplr(gt_discrete)
        gt_discrete = np.expand_dims(gt_discrete, 0)

        return self.trans(img), torch.from_numpy(keypoints.copy()).float(), torch.from_numpy(
            gt_discrete.copy()).float()

    ## 20210428 tmp
    #def val_transform(self, img, keypoints):
    #    transformed = self.resize(image=np.array(img), keypoints=keypoints)
    #    img, keypoints = transformed['image'], transformed['keypoints']
    #    return img, torch.from_numpy(np.array(keypoints).copy()).float()

class CellDatasetBL(Base):
    """for BL"""
    def __init__(self, root_path, crop_size,
                 downsample_ratio=8,
                 method='train',
                 use_albumentation=0):
        super().__init__(root_path, crop_size, downsample_ratio)
        self.method = method
        if method not in ['train', 'val']:
            raise Exception("not implement")

        self.gt_list = sorted(list(Path(self.root_path).glob("**/*.mat")))
        im_list = []
        for gt in self.gt_list:
            img_path = str(gt).split('.')[0] + '.jpg'
            im_list.append(img_path)
        #im_list = sorted(list(Path(self.root_path).glob("**/*.jpg")))
        self.im_list = im_list
        print('number of img: {}'.format(len(self.im_list)))

        self.use_albumentation = use_albumentation
        if use_albumentation:
            self.albumentations = A.Compose([
#                                      A.HorizontalFlip(p=0.5),
#                                      #A.RandomCrop(width=768, height=768, p=0.5),
                                      A.VerticalFlip(p=0.5),
                                      A.Rotate(p=0.5),
#                                      #A.CenterCrop(height=512, width=512, p=0.5),
                                      A.ShiftScaleRotate(p=0.5),
                                      A.RandomSizedCrop(min_max_height=(256, 512), height=512, width=512, p=0.5),
                                      A.OneOf([
                                          A.HueSaturationValue(p=0.5), 
                                          A.RGBShift(p=0.7)
                                      ], p=1),                          
                                      A.RandomBrightnessContrast(p=0.5),
                                      A.CLAHE(p=0.8),
                                      A.RandomBrightnessContrast(p=0.8),    
                                      A.RandomGamma(p=0.8),
                                      A.GaussianBlur(p=0.5),
                                      A.GaussNoise(p=0.5),
                                      A.Cutout(p=0.5),
                                   ],  
                                   keypoint_params=A.KeypointParams(format='xy'))

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):
        img_path = self.im_list[item]
        name = str(os.path.basename(img_path)).split('.')[0]
        #gd_path = self.gt_list[item]
        gd_path = str(img_path).split('.')[0] + '.mat'
        img = Image.open(img_path).convert('RGB')
        keypoints = sio.loadmat(gd_path)['image_info']
        #print(sio.loadmat(gd_path)["image_info"])]

        if self.method == 'train':
            return self.train_transform(img, keypoints)
        elif self.method == 'val':
            img = self.trans(img)
            return img, len(keypoints), name

    def train_transform(self, img, keypoints):
        """random crop image patch and find people in it"""
        wd, ht = img.size
        st_size = min(wd, ht)
        assert st_size >= self.c_size
        assert len(keypoints) > 0
        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        img = F.crop(img, i, j, h, w)
        nearest_dis = np.clip(keypoints[:, 2], 4.0, 128.0)

        points_left_up = keypoints[:, :2] - nearest_dis[:, None] / 2.0
        points_right_down = keypoints[:, :2] + nearest_dis[:, None] / 2.0
        bbox = np.concatenate((points_left_up, points_right_down), axis=1)
        inner_area = cal_innner_area(j, i, j+w, i+h, bbox)
        origin_area = nearest_dis * nearest_dis
        ratio = np.clip(1.0 * inner_area / origin_area, 0.0, 1.0)
        mask = (ratio >= 0.3)

        target = ratio[mask]
        keypoints = keypoints[mask]
        keypoints = keypoints[:, :2] - [j, i]  # change coodinate

        if self.use_albumentation:
#            transformed = self.albumentations(image=np.array(img), keypoints=keypoints)
#            img, keypoints = transformed['image'], transformed['keypoints']
            img = augment_and_mix(np.float32(img)/255, severity=10)
            img = Image.fromarray(np.uint8((img+1)/2*255))
            keypoints = np.array(keypoints)

        if len(keypoints) > 0:
            if random.random() > 0.5:
                img = F.hflip(img)
                keypoints[:, 0] = w - keypoints[:, 0]
        else:
            if random.random() > 0.5:
                img = F.hflip(img)
        return self.trans(img), torch.from_numpy(keypoints.copy()).float(), \
               torch.from_numpy(target.copy()).float(), st_size



class Crowd_shBL(Base):
    def __init__(self, root_path, crop_size,
                 downsample_ratio=8,
                 method='train',
                 use_albumentation=0):
        super().__init__(root_path, crop_size, downsample_ratio)
        self.method = method
        if method not in ['train', 'val']:
            raise Exception("not implement")

        self.im_list = sorted(glob(os.path.join(self.root_path, 'images', '*.jpg')))
        print('number of img: {}'.format(len(self.im_list)))

        self.use_albumentation = use_albumentation
        if use_albumentation:
            self.albumentations = A.Compose([
#                                      A.HorizontalFlip(p=0.5),
#                                      #A.RandomCrop(width=768, height=768, p=0.5),
                                      A.VerticalFlip(p=0.5),
                                      A.Rotate(p=0.5),
#                                      #A.CenterCrop(height=512, width=512, p=0.5),
                                      A.ShiftScaleRotate(p=0.5),
                                      A.RandomSizedCrop(min_max_height=(256, 512), height=512, width=512, p=0.5),
                                      A.OneOf([
                                          A.HueSaturationValue(p=0.5), 
                                          A.RGBShift(p=0.7)
                                      ], p=1),                          
                                      A.RandomBrightnessContrast(p=0.5),
                                      A.CLAHE(p=0.8),
                                      A.RandomBrightnessContrast(p=0.8),    
                                      A.RandomGamma(p=0.8),
                                      A.GaussianBlur(p=0.5),
                                      A.GaussNoise(p=0.5),
                                      A.Cutout(p=0.5),
                                   ],  
                                   keypoint_params=A.KeypointParams(format='xy'))

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):
        img_path = self.im_list[item]
        name = os.path.basename(img_path).split('.')[0]
        gd_path = os.path.join(self.root_path, 'ground-truth', 'GT_{}.mat'.format(name))
        img = Image.open(img_path).convert('RGB')
        keypoints = sio.loadmat(gd_path)['image_info'][0][0][0][0][0]

        if self.method == 'train':
            return self.train_transform(img, keypoints)
        elif self.method == 'val':
            img = self.trans(img)
            return img, len(keypoints), name

    def train_transform(self, img, keypoints):
        """random crop image patch and find people in it"""
        wd, ht = img.size
        st_size = min(wd, ht)
        assert st_size >= self.c_size
        assert len(keypoints) > 0
        #keypoints = np.array(keypoints) # ADD
        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        img = F.crop(img, i, j, h, w)
        nearest_dis = np.clip(keypoints[:, 2], 4.0, 128.0)

        points_left_up = keypoints[:, :2] - nearest_dis[:, None] / 2.0
        points_right_down = keypoints[:, :2] + nearest_dis[:, None] / 2.0
        bbox = np.concatenate((points_left_up, points_right_down), axis=1)
        inner_area = cal_innner_area(j, i, j+w, i+h, bbox)
        origin_area = nearest_dis * nearest_dis
        ratio = np.clip(1.0 * inner_area / origin_area, 0.0, 1.0)
        mask = (ratio >= 0.3)

        target = ratio[mask]
        keypoints = keypoints[mask]
        keypoints = keypoints[:, :2] - [j, i]  # change coodinate
#        if len(keypoints) > 0:
#            keypoints = keypoints - [j, i]
#            idx_mask = (keypoints[:, 0] >= 0) * (keypoints[:, 0] <= w) * \
#                       (keypoints[:, 1] >= 0) * (keypoints[:, 1] <= h)
#            keypoints = keypoints[idx_mask]
#        else:
#            keypoints = np.empty([0, 2])




        if self.use_albumentation:
#            transformed = self.albumentations(image=np.array(img), keypoints=keypoints)
#            img, keypoints = transformed['image'], transformed['keypoints']
            img = augment_and_mix(np.float32(img)/255, severity=10)
            img = Image.fromarray(np.uint8((img+1)/2*255))
            keypoints = np.array(keypoints)

        if len(keypoints) > 0:
            if random.random() > 0.5:
                img = F.hflip(img)
                keypoints[:, 0] = w - keypoints[:, 0]
        else:
            if random.random() > 0.5:
                img = F.hflip(img)
        return self.trans(img), torch.from_numpy(keypoints.copy()).float(), \
               torch.from_numpy(target.copy()).float(), st_size

