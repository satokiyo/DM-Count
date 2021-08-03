import argparse
import os
import torch
from segmentation_trainer import SegTrainer
import torch.backends.cudnn as cudnn

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--data-dir', help='data path')
    parser.add_argument('--dataset', default='segmentation', help='dataset name: segmentation')
    parser.add_argument('--lr', type=float, default=1e-5, help='the initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='the weight decay')
    parser.add_argument('--resume', default='', type=str, help='the path of resume training model')
    parser.add_argument('--max-epoch', type=int, default=1000, help='max training epoch')
    parser.add_argument('--val-epoch', type=int, default=5, help='the num of steps to log training information')
    parser.add_argument('--val-start', type=int, default=50, help='the epoch start to val')
    parser.add_argument('--batch-size', type=int, default=10, help='train batch size')
    parser.add_argument('--device', default='0', help='assign device')
    parser.add_argument('--num-workers', type=int, default=3, help='the num of training process')
    parser.add_argument('--crop-size', type=int, default=512, help='the crop size of the train image')
    parser.add_argument('--downsample-ratio', type=int, default=1)
    parser.add_argument('--visdom-env',
                           default='default_environment',
                           type=str,
                           metavar='NAME',
                           help="Name of the environment in Visdom.")
    parser.add_argument('--visdom-server',
                           default=None, #"http://localhost",
                           metavar='SRV',
                           help="Hostname of the Visdom server. "
                                "If not provided, nothing will "
                                "be sent to Visdom.")
    parser.add_argument('--visdom-port',
                           default=8991,
                           metavar='PRT',
                           help="Port of the Visdom server.")

    parser.add_argument('--encoder_name', type=str, default='resnet50')  # dpn98 resnet152 vgg19_bn timm-resnest50d efficientnet-b5 timm-resnest50d_4s2x40d vgg19_bn mobilenet_v2 timm-efficientnet-lite4 timm-skresnext50_32x4d se_resnext50_32x4d timm-efficientnet-b6 se_resnext101_32x4d xception 
       # efficientnet-b5 ロス計算のエラー出る
    parser.add_argument('--classes', type=int, default=1)
    parser.add_argument('--scale_pyramid_module', type=int, default=0,) 
    parser.add_argument('--use_attention_branch', type=int, default=0,) 
    parser.add_argument('--data_check', type=int, default=0,) 
    parser.add_argument('--use_albumentation', type=int, default=0,)
    parser.add_argument('--use_copy_paste', type=int, default=0,)
    parser.add_argument('--project', type=str, default="test")
    parser.add_argument('--input-size', type=int, default=512) 
    parser.add_argument('--t_0', type=int, default=500) #Number of iterations for the first restart of annealingwarmrestart
    parser.add_argument('--t_mult', type=int, default=1) # A factor increases after a restart. Default: 1.
    parser.add_argument('--neptune-tag', type=str, nargs='*')
    parser.add_argument('--activation', type=str,default=None)
    parser.add_argument('--deep_supervision', type=int,default=0)
    parser.add_argument('--use_ocr', type=int,default=0)
    parser.add_argument('--cfg', type=str,default="")
    parser.add_argument('--loss', type=str,default="ce") # dice softce focal jaccard lovasz wing combo mae nrdice
    parser.add_argument('--use_ssl', type=int,default=0) 
    parser.add_argument('--batch-size-ul', type=int, default=10) 
    parser.add_argument('--unsupervised_w', type=int, default=30) 
    parser.add_argument('--rampup_ends', type=float, default=0.4)
    parser.add_argument('--data_dir_ul', help='data path', type=str, default='')



    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    torch.manual_seed(0)
    # cudnn related setting
    cudnn.benchmark = True
    cudnn.deterministic = True
    cudnn.enabled = True

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu
    torch.cuda.empty_cache()
    trainer = SegTrainer(args)
    trainer.setup()
    trainer.train()
