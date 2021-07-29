import argparse
import os
import torch
from classification_trainer import ClassificationTrainer
import torch.backends.cudnn as cudnn

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--data-dir', help='data path')
    parser.add_argument('--dataset', default='classification', help='dataset name')
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

    parser.add_argument('--encoder_name', type=str, default='resnet50')
    parser.add_argument('--classes', type=int, default=1)
    parser.add_argument('--data_check', type=int, default=0,) 
    parser.add_argument('--use_albumentation', type=int, default=0,)
    parser.add_argument('--use_copy_paste', type=int, default=0,)
    parser.add_argument('--project', type=str, default="test")
    parser.add_argument('--input-size', type=int, default=512) 
    parser.add_argument('--t_0', type=int, default=500) #Number of iterations for the first restart of annealingwarmrestart
    parser.add_argument('--t_mult', type=int, default=1) # A factor increases after a restart. Default: 1.
    parser.add_argument('--neptune_workspace_name', type=str)
    parser.add_argument('--neptune_project_name', type=str)
    parser.add_argument('--neptune_api_token', type=str)
    parser.add_argument('--neptune-tag', type=str, nargs='*')
    parser.add_argument('--activation', type=str,default=None)
    parser.add_argument('--loss', type=str,default="ce") # dice softce focal jaccard lovasz wing combo mae nrdice

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    # cudnn related setting
    cudnn.benchmark = True
    cudnn.deterministic = True
    cudnn.enabled = True

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu
    torch.cuda.empty_cache()
    trainer = ClassificationTrainer(args)
    trainer.setup()
    trainer.train()
