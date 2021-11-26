import argparse
import torch
import torch.nn.functional as F
import os
import numpy as np
import pandas as pd
from datasets.classification_data import ClassificationDataset
from models.classification_model import ClassificationModel
import time
import utils.utils as utils
from PIL import Image
from tqdm import tqdm

start = time.time()

parser = argparse.ArgumentParser(description='Test ')
parser.add_argument('--device', default='0', help='assign device')
parser.add_argument('--crop-size', type=int, default=330)
parser.add_argument('--model-path', type=str, help='saved model path')
parser.add_argument('--data-dir', type=str)
parser.add_argument('--datasetname', type=str, default='segmentation')
parser.add_argument('--out-dir', type=str)
parser.add_argument('--test-type', type=str, default='test')
parser.add_argument('--flag-csv', type=str, default='')
parser.add_argument('--encoder_name', type=str, default='vgg19_bn')  # dpn98 resnet152 vgg19_bn timm-resnest50d efficientnet-b5 timm-resnest50d_4s2x40d vgg19_bn mobilenet_v2 timm-efficientnet-lite4 timm-skresnext50_32x4d se_resnext50_32x4d timm-efficientnet-b6 se_resnext101_32x4d xception 
parser.add_argument('--classes', type=int, default=4)
parser.add_argument('--input-size', type=int, default=990) 
parser.add_argument('--activation', type=str,default=None)
parser.add_argument('--flag_csv', type=str, nargs='*')
parser.add_argument('--resume', default='', type=str, help='the path of resume training model')
parser.add_argument('--batch-size', type=int, default=10, help='train batch size')
parser.add_argument('--gpu', type=int, default=1)
parser.add_argument('opts',
                    help="Modify config options using the command-line",
                    default=None,
                    nargs=argparse.REMAINDER)


args = parser.parse_args()

# device
if args.gpu and torch.cuda.is_available():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device  # set vis gpu
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


model_path = args.model_path
crop_size = args.crop_size
data_path = args.data_dir

# dataset
if args.datasetname.lower() == 'classification':
        dataset = ClassificationDataset(data_path, crop_size=crop_size, method=args.test_type, flag_csv=args.flag_csv, balanced_over_samping=False)
else:
    raise NotImplementedError


dataloader = torch.utils.data.DataLoader(dataset, args.batch_size, shuffle=False, num_workers=1, pin_memory=True)

if args.out_dir:
    import cv2
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

model = ClassificationModel(args)
model.load_state_dict(torch.load(model_path, device))
model.to(device)
model.eval()

result_list = []
for (inputs, labels, image_paths) in tqdm(dataloader):
    inputs = inputs.to(device)
    labels  = labels.to(device) 
    with torch.set_grad_enabled(False):
        outputs = model(inputs)
        preds = outputs.argmax(axis=1)
        for img_path, label, pred, softmax in zip(image_paths, labels.cpu().numpy(), preds.cpu().numpy(), outputs.cpu().numpy()):
            result_list.append([img_path, label, pred, *softmax])

forward_result = pd.DataFrame(result_list, columns=['image_path', 'label', 'prediction', 'softmax0', 'softmax1', 'softmax2', 'softmax3'])
forward_result.to_csv(os.path.join(args.out_dir, "forward_result.csv"), index=None, encoding="utf-8")

cm = pd.crosstab(forward_result['label'], forward_result['prediction'])
print(cm)
cm.to_csv(os.path.join(args.out_dir, "confusion_matrix.csv"), index=True, header=True, encoding="utf-8")