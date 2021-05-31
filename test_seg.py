import argparse
import torch
import torch.nn.functional as F
import os
import numpy as np
from datasets.segmentation_data import SegDataset
from models.segmentation_model import SegModel
import time
import utils.utils as utils
from PIL import Image
from tqdm import tqdm
import torch.backends.cudnn as cudnn 

start = time.time()

parser = argparse.ArgumentParser(description='Test ')
parser.add_argument('--device', default='0', help='assign device')
parser.add_argument('--crop-size', type=int, default=512,)
parser.add_argument('--model-path', type=str, help='saved model path')
parser.add_argument('--data-path', type=str)
parser.add_argument('--dataset', type=str, default='segmentation')
parser.add_argument('--pred-density-map-path', type=str, default='',  help='save predicted density maps when pred-density-map-path is not empty.')
parser.add_argument('--test-type', type=str, default='val',) # 'val' 'val_no_gt' 'test_no_gt'
parser.add_argument('--encoder_name', type=str, default='vgg19_bn')  # dpn98 resnet152 vgg19_bn timm-resnest50d efficientnet-b5 timm-resnest50d_4s2x40d vgg19_bn mobilenet_v2 timm-efficientnet-lite4 timm-skresnext50_32x4d se_resnext50_32x4d timm-efficientnet-b6 se_resnext101_32x4d xception 
parser.add_argument('--classes', type=int, default=1)
parser.add_argument('--scale_pyramid_module', type=int, default=0,) 
parser.add_argument('--use_attention_branch', type=int, default=0,) 
parser.add_argument('--downsample-ratio', type=int, default=1)
parser.add_argument('--input-size', type=int, default=512) 
parser.add_argument('--cfg', type=str, default='')
parser.add_argument('--activation', type=str,default=None)
parser.add_argument('--deep_supervision', type=int,default=1)
parser.add_argument('--use_ocr', type=int,default=0)
parser.add_argument('opts',
                    help="Modify config options using the command-line",
                    default=None,
                    nargs=argparse.REMAINDER)


args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.device  # set vis gpu
device = torch.device('cuda')

model_path = args.model_path
crop_size = args.crop_size
data_path = args.data_path
downsample_ratio = args.downsample_ratio # for U-Net like architecture

# dataset
if args.dataset.lower() == 'segmentation':
    if args.test_type == 'val':
        dataset = SegDataset(os.path.join(data_path, 'images', 'validation'),
                             root_path_ano=os.path.join(data_path, 'annotations', 'validation'),
                             crop_size=crop_size,
                             downsample_ratio=downsample_ratio,
                             method=args.test_type)
    elif args.test_type == 'val_no_gt':
        dataset = SegDataset(os.path.join(data_path, 'images', 'validation'),
                             root_path_ano=None,
                             crop_size=crop_size,
                             downsample_ratio=downsample_ratio,
                             method=args.test_type)
    elif args.test_type == 'test_no_gt':
        dataset = SegDataset(os.path.join(data_path, 'images', 'test'),
                             root_path_ano=None,
                             crop_size=crop_size,
                             downsample_ratio=downsample_ratio,
                             method=args.test_type)
else:
    raise NotImplementedError


dataloader = torch.utils.data.DataLoader(dataset, 1, shuffle=False,
                                         num_workers=1, pin_memory=True)

if args.pred_density_map_path:
    import cv2
    if not os.path.exists(args.pred_density_map_path):
        os.makedirs(args.pred_density_map_path)

if "hrnet" in args.encoder_name:
    from models.hrnet_seg_ocr import seg_hrnet, seg_hrnet_ocr
    from models.hrnet_seg_ocr.config import config, update_config
    update_config(config, args)
    config = config
    #model = seg_hrnet.get_seg_model(config)
    model = seg_hrnet_ocr.get_seg_model(config)
else:
    model = SegModel(args)
 
    #from models.hrnet.models import seg_hrnet, seg_hrnet_ocr
    #from models.hrnet.config import config
    #from models.hrnet.config import update_config
    #update_config(config, args)
    #model = seg_hrnet.get_seg_model(config)
     
    #from kiunet import kiunet, densekiunet, reskiunet
    #model = kiunet()

model.to(device)
model.load_state_dict(torch.load(model_path, device))
model.eval()
if args.test_type == 'val':
    epoch_mIoU = []
    epoch_IoU_class0 = []
    epoch_IoU_class1 = []
    epoch_IoU_class2 = []
    epoch_IoU_class3 = []
    nums=1
    idx = 0
    for (inputs, masks) in tqdm(dataloader):
        inputs = inputs.to(device)
        assert inputs.size(0) == 1, 'the batch size should equal to 1 in validation mode'
        masks  = masks.squeeze(1).to(device) # squeeze ch(1ch)
        with torch.set_grad_enabled(False):
            if "hrnet" in args.encoder_name and "ocr" in args.encoder_name: # with OCR output
                nums = config.MODEL.NUM_OUTPUTS
                outputs = model(inputs)
            elif args.deep_supervision:
                outputs, _ = model(inputs)
            else:
                outputs = model(inputs)

            pred = outputs
            confusion_matrix = np.zeros((args.classes, args.classes, nums))
            if not isinstance(pred, (list, tuple)):
                pred = [pred]
            size = masks.size()
            for i, x in enumerate(pred):
                x = F.interpolate(
                    input=x, size=size[-2:],
                    mode='bilinear', align_corners=True,
                )
                confusion_matrix[..., i] += utils.get_confusion_matrix(
                    masks,
                    x,
                    size,
                    args.classes,
                    ignore=-1,
                )
            for i in range(nums):
                pos = confusion_matrix[..., i].sum(1)
                res = confusion_matrix[..., i].sum(0)
                tp = np.diag(confusion_matrix[..., i])
                IoU_array = (tp / np.maximum(1.0, pos + res - tp))
                mean_IoU = IoU_array.mean()
                #print('{} {} {}'.format(i, IoU_array, mean_IoU))
            epoch_IoU_class0.append(IoU_array[0])
            epoch_IoU_class1.append(IoU_array[1])
            epoch_IoU_class2.append(IoU_array[2])
            epoch_IoU_class3.append(IoU_array[3])
            epoch_mIoU.append(mean_IoU)

            # save image
            PALETTE = [
                0,0,0,
                0,255,0,
                255,0,0,
                0,0,255
            ]
            if "hrnet" in args.encoder_name and "ocr" in args.encoder_name: # with OCR output
                outputs = outputs[1] # 0:ocr output/1:normal output
            # show images to original size (1枚目の画像だけを表示する)
            vis_img = outputs[0].detach().cpu().numpy()
            # normalize density map values from 0 to 1, then map it to 0-255.
            vis_img = (vis_img - np.min(vis_img)) / np.ptp(vis_img)
            vis_img = (vis_img*255).astype(np.uint8)
            vis_map = np.argmax(vis_img, axis=0)
            vis_map = Image.fromarray(vis_map.astype(np.uint8), mode="P")
            vis_map.putpalette(PALETTE)
            org_img = inputs[0].detach().cpu().numpy().transpose(1,2,0)
            org_img = (org_img - np.min(org_img)) / np.ptp(org_img)
            org_img = (org_img*255).astype(np.uint8)
            if (vis_map.size) != (org_img.shape[:1]):
                vis_map = vis_map.resize(org_img.shape[:2])
            vis_map = np.array(vis_map.convert("RGB"))
 
            # overlay
            overlay = np.uint8((org_img/2) + (vis_map/2))

            cv2.imwrite(os.path.join(args.pred_density_map_path, str(idx) + '_mask.jpg'), vis_map[:,:,::-1])
            cv2.imwrite(os.path.join(args.pred_density_map_path, str(idx) + '_orig.jpg'), org_img[:,:,::-1])
            cv2.imwrite(os.path.join(args.pred_density_map_path, str(idx) + '_overlay.jpg'), overlay[:,:,::-1])

            idx+=1

    iou0 = np.mean(np.array(epoch_IoU_class0))
    iou1 = np.mean(np.array(epoch_IoU_class1))
    iou2 = np.mean(np.array(epoch_IoU_class2))
    iou3 = np.mean(np.array(epoch_IoU_class3))
    miou = np.mean(np.array(epoch_mIoU))
    print('IoU0: {:.2f} IoU1: {:.2f} IoU2: {:.2f} IoU3: {:.2f} mIoU: {:.2f}'
             .format(np.mean(np.array(epoch_IoU_class0)),
                     np.mean(np.array(epoch_IoU_class1)),
                     np.mean(np.array(epoch_IoU_class2)),
                     np.mean(np.array(epoch_IoU_class3)),
                     np.mean(np.array(epoch_mIoU))))


if args.test_type in ['val_no_gt', 'test_no_gt']:
    epoch_mIoU = []
    epoch_IoU_class0 = []
    epoch_IoU_class1 = []
    epoch_IoU_class2 = []
    epoch_IoU_class3 = []
    nums=1
    idx = 0
    #for (inputs, masks) in tqdm(dataloader):
    for (inputs) in tqdm(dataloader):
        inputs = inputs.to(device)
        assert inputs.size(0) == 1, 'the batch size should equal to 1 in validation mode'
#        masks  = masks.squeeze(1).to(device) # squeeze ch(1ch)
        with torch.set_grad_enabled(False):
            if "hrnet" in args.encoder_name and "ocr" in args.encoder_name: # with OCR output
                nums = config.MODEL.NUM_OUTPUTS
                outputs = model(inputs)
            elif args.deep_supervision:
                outputs, _ = model(inputs)
            else:
                outputs = model(inputs)

            pred = outputs
#            confusion_matrix = np.zeros((args.classes, args.classes, nums))
            if not isinstance(pred, (list, tuple)):
                pred = [pred]
#            size = masks.size()
#            for i, x in enumerate(pred):
#                x = F.interpolate(
#                    input=x, size=size[-2:],
#                    mode='bilinear', align_corners=True,
#                )
#                confusion_matrix[..., i] += utils.get_confusion_matrix(
#                    masks,
#                    x,
#                    size,
#                    args.classes,
#                    ignore=-1,
#                )
#            for i in range(nums):
#                pos = confusion_matrix[..., i].sum(1)
#                res = confusion_matrix[..., i].sum(0)
#                tp = np.diag(confusion_matrix[..., i])
#                IoU_array = (tp / np.maximum(1.0, pos + res - tp))
#                mean_IoU = IoU_array.mean()
#                #print('{} {} {}'.format(i, IoU_array, mean_IoU))
#            epoch_IoU_class0.append(IoU_array[0])
#            epoch_IoU_class1.append(IoU_array[1])
#            epoch_IoU_class2.append(IoU_array[2])
#            epoch_IoU_class3.append(IoU_array[3])
#            epoch_mIoU.append(mean_IoU)

            # save image
            PALETTE = [
                0,0,0,
                0,255,0,
                255,0,0,
                0,0,255
            ]
            if "hrnet" in args.encoder_name and "ocr" in args.encoder_name: # with OCR output
                outputs = outputs[1] # 0:ocr output/1:normal output
            # show images to original size (1枚目の画像だけを表示する)
            vis_img = outputs[0].detach().cpu().numpy()
            # normalize density map values from 0 to 1, then map it to 0-255.
            vis_img = (vis_img - np.min(vis_img)) / np.ptp(vis_img)
            vis_img = (vis_img*255).astype(np.uint8)
            vis_map = np.argmax(vis_img, axis=0)
            vis_map = Image.fromarray(vis_map.astype(np.uint8), mode="P")
            vis_map.putpalette(PALETTE)
            org_img = inputs[0].detach().cpu().numpy().transpose(1,2,0)
            org_img = (org_img - np.min(org_img)) / np.ptp(org_img)
            org_img = (org_img*255).astype(np.uint8)
            if (vis_map.size) != (org_img.shape[:1]):
                vis_map = vis_map.resize(org_img.shape[:2])
            vis_map = np.array(vis_map.convert("RGB"))
 
            # overlay
            overlay = np.uint8((org_img/2) + (vis_map/2))

            cv2.imwrite(os.path.join(args.pred_density_map_path, str(idx) + '_mask.jpg'), vis_map[:,:,::-1])
            cv2.imwrite(os.path.join(args.pred_density_map_path, str(idx) + '_orig.jpg'), org_img[:,:,::-1])
            cv2.imwrite(os.path.join(args.pred_density_map_path, str(idx) + '_overlay.jpg'), overlay[:,:,::-1])

            idx+=1

#    iou0 = np.mean(np.array(epoch_IoU_class0))
#    iou1 = np.mean(np.array(epoch_IoU_class1))
#    iou2 = np.mean(np.array(epoch_IoU_class2))
#    iou3 = np.mean(np.array(epoch_IoU_class3))
#    miou = np.mean(np.array(epoch_mIoU))
#    print('IoU0: {:.2f} IoU1: {:.2f} IoU2: {:.2f} IoU3: {:.2f} mIoU: {:.2f}'
#             .format(np.mean(np.array(epoch_IoU_class0)),
#                     np.mean(np.array(epoch_IoU_class1)),
#                     np.mean(np.array(epoch_IoU_class2)),
#                     np.mean(np.array(epoch_IoU_class3)),
#                     np.mean(np.array(epoch_mIoU))))


