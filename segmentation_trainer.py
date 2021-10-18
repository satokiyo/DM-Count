import os
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.cuda import amp
import numpy as np
import cv2
from datetime import datetime, timedelta, timezone
from tqdm import tqdm
from logger import Logger
from torch.utils.data.dataloader import default_collate

from utils.pytorch_utils import Save_Handle, AverageMeter
import utils.log_utils as log_utils
import utils.utils as utils

#from datasets.crowd import Crowd_qnrf, Crowd_nwpu, Crowd_sh, CellDataset
from datasets.segmentation_data import SegDataset
from models.segmentation_model import SegModel

from losses.cross_entropy import CrossEntropy, OhemCrossEntropy
from losses.dice import DiceLoss
from losses.focal import FocalLoss
from losses.jaccard import JaccardLoss
from losses.lovasz import LovaszLoss
from losses.soft_ce import SoftCrossEntropyLoss
from losses.wing import WingLoss
from losses.combo import ComboLoss
from losses.nrdice import NoiseRobustDiceLoss
from losses.losses import abCE_loss
from losses.losses import softmax_kl_loss, softmax_mse_loss, softmax_js_loss, consistency_weight
 

import neptune.new as neptune
from neptune.new.types import File

from PIL import Image
import copy
import gc

PALETTE = [
    255,0,0, #ff0000 0 cancer
    0,255,0, #00ff00 1 not_cancer
    0,0,255, #0000ff 2 bronchial_epitherial
    0,255,255, #00ffff 3 macrophage
    255,0,255, #ff00ff 4 lymphocyte
    255,255,0, #ffff00 5 stroma
    0,0,0,   #000000 6 background
]

def cycle(iterable):
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


class ModelEMA(nn.Module):
    def __init__(self, model, decay=0.999, device=None):
        super().__init__()
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device
        if self.device is not None:
            self.module.to(device=device)

    def forward(self, input, unsupervised=False):
        return self.module(input, unsupervised)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.parameters(), model.parameters()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))
            for ema_v, model_v in zip(self.module.buffers(), model.buffers()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(model_v)

    def update_parameters(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state_dict):
        self.module.load_state_dict(state_dict)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            #self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            #self.save_checkpoint(val_loss, model)
            self.counter = 0

    #def save_checkpoint(self, val_loss, model):
    #    '''Saves model when validation loss decrease.'''
    #    if self.verbose:
    #        self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
    #    torch.save(model.state_dict(), self.path)
    #    self.val_loss_min = val_loss


class SegTrainer(object):
    def __init__(self, args):
        self.args = args

    def setup(self):
        args = self.args
        use_ocr=False

        # device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            assert self.device_count == 1
        else:
            raise Exception("gpu is not available")

        # Model
        if "hrnet" in args.encoder_name:
            from models.hrnet_seg_ocr import seg_hrnet, seg_hrnet_ocr
            from models.hrnet_seg_ocr.config import config, update_config
            update_config(config, args)
            self.config = config
            if "ocr" in args.encoder_name:
                use_ocr=True
     
            #self.model = seg_hrnet.get_seg_model(config)
            self.model = seg_hrnet_ocr.get_seg_model(config)
        else:
            self.model = SegModel(args)
            if args.use_ocr:
                use_ocr=True
    
            #from kiunet import kiunet, densekiunet, reskiunet
            #self.model = kiunet()
    
            #from models.hrnet.models import seg_hrnet, seg_hrnet_ocr
            #from models.hrnet.config import config
            #from models.hrnet.config import update_config
            #update_config(config, args)
            #self.model = seg_hrnet.get_seg_model(config)
            #self.model = seg_hrnet_ocr.get_seg_model(config)

        self.model = ModelEMA(self.model, 0.9999, self.device)
        self.model.to(self.device)

        # criterion
        # ce dice softce focal jaccard lovasz wing combo mae nrdice
        if args.loss == 'ce':
            self.criterion = CrossEntropy(ignore_label=-1,  # DiceLoss
                                          weight=None).to(self.device)
            if use_ocr:
                self.criterion_ocr = CrossEntropy(ignore_label=-1,
                                              weight=None,
                                              use_ocr=use_ocr,
                                              w_loss=1.,
                                              w_loss_ocr=0.1).to(self.device)

            ## Focal lossと類似。Top70%の勾配のみ使うことでonline hard example miningをする。若干Focalの方が良いらしい
            #self.criterion = OhemCrossEntropy(ignore_label=-1,
            #                                  thres=config.LOSS.OHEMTHRES,
            #                                  min_kept=config.LOSS.OHEMKEEP,
            #                                  weight=None).to(self.device)

        elif args.loss == 'dice':
            self.criterion = DiceLoss(
                                               mode='multiclass',  # mode: Loss mode 'binary', 'multiclass' or 'multilabel'
                                               classes=None, # all channel contribute for loss
                                               log_loss=False,
                                               from_logits=True,
                                               smooth=0.0,
                                               ignore_index=None,).to(self.device)
 
            if use_ocr:
                self.criterion_ocr = DiceLoss(
                                               mode='multiclass',  # mode: Loss mode 'binary', 'multiclass' or 'multilabel'
                                               classes=None, # all channel contribute for loss
                                               log_loss=False,
                                               from_logits=True,
                                               smooth=0.0,
                                               ignore_index=None,
                                               use_ocr=use_ocr,
                                               w_loss=1.,
                                               w_loss_ocr=0.1).to(self.device)
        elif args.loss == 'focal':
            self.criterion = FocalLoss(
                                               mode='multiclass',  # mode: Loss mode 'binary', 'multiclass' or 'multilabel'
                                               alpha = None, # alpha: Prior probability of having positive value in target.
                                               gamma = 2., # gamma: Power factor for dampening weight (focal strength).
                                               ignore_index=None,).to(self.device)
 
 
            if use_ocr:
                self.criterion_ocr = FocalLoss(
                                               mode='multiclass',  # mode: Loss mode 'binary', 'multiclass' or 'multilabel'
                                               alpha = None, # alpha: Prior probability of having positive value in target.
                                               gamma = 2., # gamma: Power factor for dampening weight (focal strength).
                                               ignore_index=None,
                                               use_ocr=use_ocr,
                                               w_loss=1.,
                                               w_loss_ocr=0.1).to(self.device)

        elif args.loss == 'jaccard':
            self.criterion = JaccardLoss(
                                               mode='multiclass',  # mode: Loss mode 'binary', 'multiclass' or 'multilabel'
                                               classes=None, # all channel contribute for loss
                                               log_loss=False,
                                               from_logits=True,
                                               smooth=0.0,).to(self.device)
 
            if use_ocr:
                self.criterion_ocr = JaccardLoss(
                                               mode='multiclass',  # mode: Loss mode 'binary', 'multiclass' or 'multilabel'
                                               classes=None, # all channel contribute for loss
                                               log_loss=False,
                                               from_logits=True,
                                               smooth=0.0,
                                               use_ocr=use_ocr,
                                               w_loss=1.,
                                               w_loss_ocr=0.1).to(self.device)
        elif args.loss == 'lovasz':
            self.criterion = LovaszLoss(
                                               mode='multiclass',  # mode: Loss mode 'binary', 'multiclass' or 'multilabel'
                                               per_image=False, #  If True loss computed per each image and then averaged, else computed per whole batch
                                               from_logits=True,
                                               ignore_index=None,).to(self.device)

            if use_ocr:
                self.criterion_ocr = LovaszLoss(
                                               mode='multiclass',  # mode: Loss mode 'binary', 'multiclass' or 'multilabel'
                                               per_image=False, #  If True loss computed per each image and then averaged, else computed per whole batch
                                               from_logits=True,
                                               ignore_index=None,
                                               use_ocr=use_ocr,
                                               w_loss=1.,
                                               w_loss_ocr=0.1).to(self.device)

        elif args.loss == 'softce':
            self.criterion = SoftCrossEntropyLoss(
                                               reduction = "mean",
                                               smooth_factor = 0.1,
                                               ignore_index = -100,
                                               dim = 1,).to(self.device)
 
 
            if use_ocr:
                self.criterion_ocr = SoftCrossEntropyLoss(
                                               reduction = "mean",
                                               smooth_factor = None,
                                               ignore_index = -100,
                                               dim = 1,
                                               use_ocr=use_ocr,
                                               w_loss=1.,
                                               w_loss_ocr=0.1).to(self.device)

        elif args.loss == 'wing':
            self.criterion = WingLoss(
                                               mode='multiclass',
                                               reduction = "mean",
                                               ignore_index = -100,
                                               width = 5,
                                               curvature = 0.5,).to(self.device)
 
            if use_ocr:
                self.criterion_ocr = WingLoss(
                                               mode='multiclass',
                                               reduction = "mean",
                                               ignore_index = -100,
                                               width = 5,
                                               curvature = 0.5,
                                               use_ocr=use_ocr,
                                               w_loss=1.,
                                               w_loss_ocr=0.1).to(self.device)

        elif args.loss == 'combo':
            self.criterion = ComboLoss(
                                               loss_weight=1.0,
                                               smooth=1,
                                               alpha=0.5,
                                               ce_ratio=0.5,
                                               reduction='mean',
                                               class_weight=None, 
                                               use_sigmoid=True,).to(self.device)

            if use_ocr:
                self.criterion_ocr = ComboLoss(
                                               loss_weight=1.0,
                                               smooth=1,
                                               alpha=0.5,
                                               ce_ratio=0.5,
                                               reduction='mean',
                                               class_weight=None, 
                                               use_sigmoid=True,
                                               use_ocr=use_ocr,
                                               w_loss=1.,
                                               w_loss_ocr=0.1).to(self.device)
        elif args.loss == 'abCE':
            self.criterion = None # abCEロスはsemi-supervisedモードのsupervise lossのみで使用する。下で設定。


        elif args.loss == 'mae':
            pass


        elif args.loss == 'nrdice':
            self.criterion = NoiseRobustDiceLoss(
                                               mode='multiclass',  # mode: Loss mode 'binary', 'multiclass' or 'multilabel'
                                               classes=None, # all channel contribute for loss
                                               log_loss=False,
                                               from_logits=True,
                                               smooth=0.0,
                                               gamma=1.5,
                                               ignore_index=None,).to(self.device)
 
            if use_ocr:
                self.criterion_ocr = NoiseRobustDiceLoss(
                                               mode='multiclass',  # mode: Loss mode 'binary', 'multiclass' or 'multilabel'
                                               classes=None, # all channel contribute for loss
                                               log_loss=False,
                                               from_logits=True,
                                               smooth=0.0,
                                               gamma=1.5,
                                               ignore_index=None,
                                               use_ocr=use_ocr,
                                               w_loss=1.,
                                               w_loss_ocr=0.1).to(self.device)


        # optimizer
#        trainable_params = [{'params': filter(lambda p:p.requires_grad, self.model.get_other_params())},
#                            {'params': filter(lambda p:p.requires_grad, self.model.get_backbone_params()), 'lr': args.lr / 10}]
#
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr_max, weight_decay=args.weight_decay)

        # scheduler
        #T_0=35 #Number of iterations for the first restart.
        #T_mult=1 # A factor increases after a restart. Default: 1.
        lr_min=args.lr_min
        args.t_0 = int(args.max_epoch)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, args.t_0, args.t_mult, lr_min)
        #self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda = lambda epoch: epoch*epoch/(args.max_epoch*args.max_epoch)) # check LR @max50epoch.
        #self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5, threshold=0.001, min_lr=lr_min)

        self.amp = bool(args.amp)
        self.scaler = amp.GradScaler(enabled=self.amp)


        # dataset
        if args.dataset.lower() == 'segmentation':
            self.datasets = {'train': SegDataset(os.path.join(args.datadir, 'images', 'training'),
                                                 os.path.join(args.datadir, 'annotations', 'training'),
                                                 args.crop_size, args.downsample_ratio, 'train', use_albumentation=args.use_albumentation, use_copy_paste=args.use_copy_paste),
                             'val': SegDataset(os.path.join(args.datadir, 'images', 'validation'),
                                               os.path.join(args.datadir, 'annotations', 'validation'),
                                               args.crop_size, args.downsample_ratio, 'val'),
                             }
            self.dataset_ul = SegDataset(args.datadir_ul,
                                         root_path_ano=None,
                                         crop_size=args.crop_size,
                                         downsample_ratio=args.downsample_ratio,
                                         method='test_no_gt')
        else:
            raise NotImplementedError

        self.dataloaders = {x: DataLoader(self.datasets[x],
                                          collate_fn=default_collate,
                                          batch_size=(args.batch_size
                                                      if x == 'train' else 1),
                                          shuffle=(True if x == 'train' else False),
                                          drop_last=(True if x == 'train' else False),
                                          num_workers=args.num_workers * self.device_count,
                                          pin_memory=(True if x == 'train' else False))
                            for x in ['train', 'val']}

        use_size = len(self.datasets['train'])*2 # use x2 data for unsupervised training
        if use_size > len(self.dataset_ul):
            use_indices=None
            print(f'use images unlabeled : {len(self.dataset_ul)}')
            self.dataloader_ul = DataLoader(self.dataset_ul,
                                              collate_fn=default_collate,
                                              batch_size=args.batch_size_ul,
                                              shuffle=True,
                                              num_workers=args.num_workers * self.device_count,
                                              pin_memory=True)
        else:
            np.random.seed(43)
            use_indices = np.random.choice(len(self.dataset_ul), use_size, replace=False).tolist()
            print(f'use images unlabeled : {use_size}')
            self.dataloader_ul = DataLoader(self.dataset_ul,
                                              collate_fn=default_collate,
                                              batch_size=args.batch_size_ul,
                                              num_workers=args.num_workers * self.device_count,
                                              pin_memory=True,
                                              sampler=SubsetRandomSampler(use_indices))

        if args.use_ssl:
            # Supervised and unsupervised losses
            #self.unsuper_loss = softmax_kl_loss
            self.unsuper_loss = softmax_mse_loss
            #self.unsuper_loss = softmax_js_loss
            #rampup_ends = int(config['ramp_up'] * config['trainer']['epochs'])
            rampup_ends = int(args.max_epoch * args.rampup_ends)
            iters_per_epoch = int(len(self.dataloader_ul) // args.batch_size_ul)
            cons_w_unsup = consistency_weight(final_w=args.unsupervised_w, iters_per_epoch=iters_per_epoch ,
                                                rampup_ends=rampup_ends)
            self.unsup_loss_w = cons_w_unsup
            if args.loss == 'abCE' and self.criterion is None:
                self.criterion = abCE_loss(iters_per_epoch=iters_per_epoch, epochs=args.max_epoch, num_classes=args.classes).to(self.device)
                if use_ocr:
                    self.criterion_ocr = CrossEntropy(ignore_label=-1,
                                              weight=None,
                                              use_ocr=use_ocr,
                                              w_loss=1.,
                                              w_loss_ocr=0.1).to(self.device)


        # data check
        if args.data_check:
            self.data_check() # only check dataloader
            exit()

        # neptune
        self.run = neptune.init(project='satokiyo/{}'.format(args.project),
                           api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwMTAxZjFkZS1jODNmLTQ2MWQtYWJhYi1kZTM5OGQ3NWYyZDAifQ==',
                           source_files=['*.py', '*.sh', 'models'])
        for arg in vars(args):
            self.run[f'param_{arg}'] = getattr(args, arg)
        self.run["sys/tags"].add(args.neptune_tag)  # tag

        # save dir setting
        sub_dir = 'encoder-{}_input-{}_downrate-{}_scale_pyramid_module-{}_attention_branch-{}_albumentation-{}_copy_paste-{}_deep_supervision-{}_ocr-{}_ssl-{}'.format(
            args.encoder_name, args.crop_size, args.downsample_ratio, args.scale_pyramid_module, args.use_attention_branch, args.use_albumentation, args.use_copy_paste, args.deep_supervision, args.use_ocr, args.use_ssl)


        save_dir_root = os.path.join('ckpts', sub_dir)
        if not os.path.exists(save_dir_root):
            os.makedirs(save_dir_root)

        JST = timezone(timedelta(hours=+9), 'JST') # タイムゾーンの生成
        time_str = datetime.strftime(datetime.now(JST), '%m%d-%H%M%S')
        self.save_dir = os.path.join(save_dir_root, time_str)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # logger
        self.logger = log_utils.get_logger(os.path.join(self.save_dir, 'train-{:s}.log'.format(time_str)))
        self.logger.info('using {} gpus'.format(self.device_count))

        # Visdom setup
        self.vlog = Logger(server=args.visdom_server,
                            port=args.visdom_port,
                            env_name=args.visdom_env)


        log_utils.print_config(vars(args), self.logger)


        self.start_epoch = 0
        if args.resume:
            self.logger.info('loading pretrained model from ' + args.resume)
            suf = args.resume.rsplit('.', 1)[-1]
            if suf == 'tar':
                checkpoint = torch.load(args.resume, self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
            elif suf == 'pth':
                self.model.load_state_dict(torch.load(args.resume, self.device))
        else:
            self.logger.info('random initialization')

        self.save_list = Save_Handle(max_num=1)
        self.best_loss = np.inf
        self.best_count = 0

        # experimental
        if args.finetune:
            for param in self.model.encoder.parameters:
                param.requires_grad = False # freeze


    def train(self):
        """training process"""
        args = self.args
        early_stopping = EarlyStopping(patience=10, verbose=True)
        for epoch in range(self.start_epoch, args.max_epoch + 1):
            #self.logger.info('-' * 5 + 'Epoch {}/{}'.format(epoch, args.max_epoch) + '-' * 5)
            self.epoch = epoch
            self.train_eopch()
            if epoch % args.val_epoch == 0 and epoch >= args.val_start:
                self.val_epoch(early_stopping)
            if early_stopping.early_stop:
                print("Early stopping")
                break

    def train_eopch(self):
        epoch_loss = AverageMeter()
        epoch_start = time.time()
        self.model.train()  # Set model to training mode

        if self.args.use_ssl:
            iters = len(self.dataloader_ul)
            stream = tqdm(zip(cycle(self.dataloaders['train']), self.dataloader_ul), total=iters)
        else:
            iters = len(self.dataloaders['train'])
            stream = tqdm(zip(self.dataloaders['train'], cycle(self.dataloader_ul)), total=iters)
        for step, (x_l, x_ul) in enumerate(stream):
            inputs, masks = x_l
            if self.args.use_ssl:
                x_ul = x_ul.to(self.device)
            inputs = inputs.to(self.device)
            masks  = masks.squeeze(1).to(self.device) # squeeze ch(1ch)
            N = inputs.size(0)
            with torch.set_grad_enabled(True):
                with amp.autocast(enabled=self.amp):
                    if "hrnet" in self.args.encoder_name and "ocr" in self.args.encoder_name: # with OCR output
                        outputs = self.model(inputs)
                        _loss = self.criterion_ocr(outputs, masks)
                        loss = _loss.mean()

                    elif self.args.deep_supervision:
                        if self.args.use_ocr:
                            outputs, intermediates, out_ocr = self.model(inputs)
                            _loss = self.criterion_ocr([out_ocr, outputs], masks) # out_ocr, out 順番注意
                            w_deep_sup = 0.4
                            w_each = w_deep_sup / len(intermediates)
                            deep_sup_loss = [w_each*self.criterion(out, masks) for out in intermediates]
                            loss = _loss + torch.stack(deep_sup_loss).sum()
                            del _loss, deep_sup_loss

                        else:
                            outputs, intermediates = self.model(inputs)
                            if self.args.loss == "abCE":
                                _loss = self.criterion(outputs, masks, ignore_index=-1, curr_iter=step, epoch=self.epoch)
                            else:
                                _loss = self.criterion(outputs, masks)
                            w_deep_sup = 0.4
                            w_each = w_deep_sup / len(intermediates)
                            if self.args.loss == "abCE":
                                deep_sup_loss = [w_each*self.criterion(out, masks, ignore_index=-1, curr_iter=step, epoch=self.epoch) for out in intermediates]
                            else:
                                deep_sup_loss = [w_each*self.criterion(out, masks) for out in intermediates]
                            del intermediates
                            loss = _loss + torch.stack(deep_sup_loss).sum()
                            del _loss, deep_sup_loss

                    else:
                        if self.args.use_ocr:
                            outputs, out_ocr = self.model(inputs)
                            _loss = self.criterion_ocr([out_ocr, outputs], masks) # out_ocr, out 順番注意
                            loss = _loss
                            del _loss
                        else:
                            outputs = self.model(inputs)
                            # Compute loss.
                            if self.args.loss == "abCE":
                                _loss = self.criterion(outputs, masks, ignore_index=-1, curr_iter=step, epoch=self.epoch)
                            else:
                                _loss = self.criterion(outputs, masks)
                            loss = _loss
                            del _loss

                    if self.args.use_ssl: # semi-supervised
                        output_ul_main, outputs_ul_aux = self.model(x_ul, unsupervised=True)
                        #targets = F.softmax(output_ul_main.detach(), dim=1) # main decoder output
                        targets = output_ul_main.detach() # main decoder output
                        loss_unsup = []
                        for aux_out in outputs_ul_aux: # aux decoder outputs
                            loss_unsup.append(self.unsuper_loss(inputs=aux_out, targets=targets, conf_mask=False, threshold=None, use_softmax=True))
                            #loss_unsup.append(self.criterion(aux_out, targets))
                            del aux_out
                        del output_ul_main, outputs_ul_aux, targets
                        # Compute the unsupervised loss
                        loss_unsup = sum(loss_unsup) / len(loss_unsup)
                        weight_u = self.unsup_loss_w(epoch=self.epoch, curr_iter=step)
                        loss_unsup = loss_unsup * weight_u

                        loss = loss + loss_unsup
                        del loss_unsup

                loss = self.scaler.scale(loss)
                epoch_loss.update(loss.item(), N)

                #if torch.isnan(loss):
                #    self.model = self.prev_model
                #    self.optimizer.load_state_dict(self.prev_optimizer.state_dict())
                #else:
                #    self.prev_model = copy.deepcopy(self.model)
                #    self.prev_optimizer = copy.deepcopy(self.optimizer)

                #    self.optimizer.zero_grad()
                #    loss.backward()
                #    self.optimizer.step()

                loss.backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step(self.epoch + step / iters)
                self.optimizer.zero_grad(set_to_none=True)

            stream.set_description(
                'Epoch {} Train, Loss: {:.2f}'
                    .format(self.epoch, epoch_loss.get_avg()))
            del loss
 
            # show image
            if step % 3 == 0:
               with torch.set_grad_enabled(False):
                    # show images to original size (1枚目の画像だけを表示する)
                    if "hrnet" in self.args.encoder_name and "ocr" in self.args.encoder_name: # with OCR output
                        outputs = outputs[1] # 0:ocr output/1:normal output
                    # show images to original size (1枚目の画像だけを表示する)
                    vis_img = outputs[0].detach().cpu().numpy()
                    # normalize density map values from 0 to 1, then map it to 0-255.
                    vis_img = (vis_img - np.min(vis_img)) / np.ptp(vis_img)
                    vis_img = (vis_img*255).astype(np.uint8)
                    vis_map = np.argmax(vis_img, axis=0)
                    vis_map = Image.fromarray(vis_map.astype(np.uint8), mode="P")
                    vis_map.putpalette(PALETTE)
    #                vis_map = vis_map.transpose(1,2,0)
    #                vis_img = cv2.resize(vis_img, dsize=(int(self.args.input_size), int(self.args.input_size)), interpolation=cv2.INTER_NEAREST)
                    org_img = inputs[0].detach().cpu().numpy().transpose(1,2,0)
                    org_img = (org_img - np.min(org_img)) / np.ptp(org_img)
                    org_img = (org_img*255).astype(np.uint8)
                    if (vis_map.size) != (org_img.shape[:1]):
                        vis_map = vis_map.resize(org_img.shape[:2])
                    vis_map = np.array(vis_map.convert("RGB"))
                    # overlay
                    overlay = np.uint8((org_img/2) + (vis_map/2)).transpose(2,0,1)
                    # visdom
                    self.vlog.image(imgs=[overlay],
                              titles=['(Training) Image overlay'],
                              window_ids=[1])
                    ## neptune
                    #self.run['image_train'].upload(File.as_image(overlay.astype(np.uint8)))


        model_state_dic = self.model.state_dict()
        save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(self.epoch))
        torch.save({
            'epoch': self.epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_state_dict': model_state_dic
        }, save_path)
        self.save_list.append(save_path)

        # Log training losses to visdom
        self.vlog.train_losses(terms=[epoch_loss.get_avg()],#,
                               iteration_number=self.epoch,
                               terms_legends=['Loss'])#,
        # Log training losses to neptune
        self.run['train/Loss'].log(epoch_loss.get_avg())
        self.run['train/lr'].log(self.scheduler.get_last_lr())

        del epoch_loss
        gc.collect()


    def val_epoch(self, early_stopping):
        args = self.args
        epoch_start = time.time()
        self.model.eval()  # Set model to evaluate mode
        epoch_loss = []
        nums=1
        epoch_iou_class = {f'{i}' : [] for i in range(self.args.classes)}
        epoch_dice_class = {f'{i}' : [] for i in range(self.args.classes)}
        epoch_recall_class = {f'{i}' : [] for i in range(self.args.classes)} 
        epoch_precision_class = {f'{i}' : [] for i in range(self.args.classes)}
        epoch_acc = []
        epoch_miou = []
        epoch_dice_macro = []
        epoch_recall_macro = []
        epoch_precision_macro = []
 
        stream = tqdm(self.dataloaders['val'])
        for step, (inputs, masks) in enumerate(stream):
            inputs = inputs.to(self.device)
            assert inputs.size(0) == 1, 'the batch size should equal to 1 in validation mode'
            masks  = masks.squeeze(1).to(self.device) # squeeze ch(1ch)
            with torch.set_grad_enabled(False):
                if "hrnet" in self.args.encoder_name and "ocr" in self.args.encoder_name: # with OCR output
                    nums = self.config.MODEL.NUM_OUTPUTS
                    outputs = self.model(inputs)
#                    _loss = self.criterion(outputs, masks)
                    _loss = self.criterion_ocr(outputs, masks)
                    loss = _loss.mean()
                elif self.args.deep_supervision:
                    if self.args.use_ocr:
                        outputs, intermediates, out_ocr = self.model(inputs)
                        del intermediates, out_ocr
                        _loss = self.criterion(outputs, masks) # not calc deep supervision loss / ocr loss in validation
                        loss = _loss
                    else:
                        outputs, intermediates = self.model(inputs)
                        del intermediates
                        if self.args.loss == "abCE":
                            _loss = self.criterion(outputs, masks, ignore_index=-1, curr_iter=step, epoch=self.epoch)
                        else:
                            _loss = self.criterion(outputs, masks)
                        loss = _loss
                else:
                    if self.args.use_ocr:
                        outputs, out_ocr = self.model(inputs)
                        _loss = self.criterion(outputs, masks) # not calc deep supervision loss / ocr loss in validation
                        loss = _loss
                    else:
                        outputs = self.model(inputs)
                        # Compute loss.
                        if self.args.loss == "abCE":
                            _loss = self.criterion(outputs, masks, ignore_index=-1, curr_iter=step, epoch=self.epoch)
                        else:
                            _loss = self.criterion(outputs, masks)
                        loss = _loss

                epoch_loss.append(loss.item())
                del loss

                pred = outputs
                confusion_matrix = np.zeros((self.args.classes, self.args.classes, nums))
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
                        self.args.classes,
                        ignore=-1,
                    )
                    del x
                del pred
                for i in range(nums):
                    pos = confusion_matrix[..., i].sum(0) # pred
                    res = confusion_matrix[..., i].sum(1) # label
                    tp = np.diag(confusion_matrix[..., i])
                    fn = res - tp
                    fp = pos - tp
                    #iou_array = (tp / np.maximum(1.0, pos + res - tp)) # iou=tp/tp+fp+fn -> pos=pred=tp+fp, res-tp=fn
                    iou_array = (tp / np.maximum(1.0, tp+fp+fn))
                    recall_array = (tp / np.maximum(1.0, res)) 
                    precision_array = (tp / np.maximum(1.0, pos)) 
                    dice_array = tp / np.maximum(1.0, (tp + 0.5*(fp+fn)))
                    mean_iou = iou_array.mean()
                    recall_macro = recall_array.mean()
                    precision_macro = precision_array.mean()
                    dice_macro = dice_array.mean()
                    acc = (tp / np.maximum(1.0, confusion_matrix.sum())).sum() 
                    #print('{} {} {}'.format(i, iou_array, mean_iou))

                for i in range(self.args.classes):
                    epoch_iou_class[f'{i}'].append(iou_array[i])
                    epoch_dice_class[f'{i}'].append(dice_array[i])
                    epoch_recall_class[f'{i}'].append(recall_array[i])
                    epoch_precision_class[f'{i}'].append(precision_array[i])

                epoch_acc.append(acc)
                epoch_miou.append(mean_iou)
                epoch_dice_macro.append(dice_macro)
                epoch_recall_macro.append(recall_macro)
                epoch_precision_macro.append(precision_macro)

                stream.set_description('Epoch {} Val, Loss: {:.2f} iou0: {:.2f} iou1: {:.2f} iou2: {:.2f} iou3: {:.2f} iou4: {:.2f} iou5: {:.2f} iou6: {:.2f} miou: {:.2f}'
                                              .format(self.epoch,
                                               np.mean(np.array(epoch_loss)),
                                               np.mean(np.array(epoch_iou_class['0'])),
                                               np.mean(np.array(epoch_iou_class['1'])),
                                               np.mean(np.array(epoch_iou_class['2'])),
                                               np.mean(np.array(epoch_iou_class['3'])),
                                               np.mean(np.array(epoch_iou_class['4'])),
                                               np.mean(np.array(epoch_iou_class['5'])),
                                               np.mean(np.array(epoch_iou_class['6'])),
                                               np.mean(np.array(epoch_miou))))
    
                # show image
                if step % 15 == 0:
                    if "hrnet" in self.args.encoder_name and "ocr" in self.args.encoder_name: # with OCR output
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
                    del outputs
                    del inputs
                    org_img = (org_img - np.min(org_img)) / np.ptp(org_img)
                    org_img = (org_img*255).astype(np.uint8)
                    if (vis_map.size) != (org_img.shape[:1]):
                        vis_map = vis_map.resize(org_img.shape[:2])
                    vis_map = np.array(vis_map.convert("RGB"))
     
                    # overlay
                    overlay = np.uint8((org_img/2) + (vis_map/2)).transpose(2,0,1)
    
                    # visdom
                    self.vlog.image(imgs=[overlay],
                              titles=['(Validation) Image w/ output heatmap and labeled points'],
                              window_ids=[2])
                    ## neptune
                    #self.run['image_val'].upload(File.as_image(overlay.transpose(1,2,0)))
    

        model_state_dic = self.model.state_dict()

        loss = np.mean(np.array(epoch_loss))
        if loss < self.best_loss:
            self.best_loss = loss
            self.logger.info("save best loss {:.2f} model epoch {}".format(self.best_loss, self.epoch))
            torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model_{}.pth'.format(self.best_count)))
            self.best_count += 1

        # Log validation losses
        self.vlog.val_losses(terms=[loss],
                               iteration_number=self.epoch,
                               terms_legends=['LOSS'])
        # Log validation losses to neptune
        self.run['val/Loss'].log(loss)

        for i in range(self.args.classes):
            self.run[f'val/iou{i}'].log(np.mean(np.array(epoch_iou_class[f'{i}'])))
            self.run[f'val/dice{i}'].log(np.mean(np.array(epoch_dice_class[f'{i}'])))
            self.run[f'val/recall{i}'].log(np.mean(np.array(epoch_recall_class[f'{i}'])))
            self.run[f'val/precision{i}'].log(np.mean(np.array(epoch_precision_class[f'{i}'])))

        self.run['val/acc'].log(np.mean(np.array(epoch_acc)))
        self.run['val/iou_macro'].log(np.mean(np.array(epoch_miou)))
        self.run['val/dice_macro'].log(np.mean(np.array(epoch_dice_macro)))
        self.run['val/recall_macro'].log(np.mean(np.array(epoch_recall_macro)))
        self.run['val/precision_macro'].log(np.mean(np.array(epoch_precision_macro)))

        early_stopping(loss)

        del loss
        del epoch_loss
        del epoch_iou_class
        del epoch_dice_class
        del epoch_recall_class
        del epoch_precision_class
        del epoch_acc
        del epoch_miou
        del epoch_dice_macro
        del epoch_recall_macro
        del epoch_precision_macro
        gc.collect()
 
 

    def data_check(self):
        import matplotlib.pyplot as plt
        from PIL import Image

        def display_image_grid(train_dataset, num=24):
            cols = 2
            rows = num // cols
            figure, ax = plt.subplots(nrows=rows*2, ncols=cols, figsize=(10, 50))
            for i in range(num):
                img, mask = train_dataset[i]
                img = img.detach().cpu().numpy()
                mask = mask.detach().cpu().numpy()
                img = (img - np.min(img)) / np.ptp(img)
                img = (img*255).astype(np.uint8)
                img = img.transpose(1,2,0)
                mask = Image.fromarray(mask.astype(np.uint8), mode="P")
                mask.putpalette(PALETTE)
                if (mask.size) != (img.shape[:1]):
                    mask = mask.resize(img.shape[:2])
                mask = np.array(mask.convert("RGB"))
                #mask = (mask*255).astype(np.uint8)
                #row = i // cols
                #col = i % cols
                #blend = img /2 + np.array(mask.convert("RGB")) / 2
                #ax[row,col].imshow(Image.fromarray(blend.astype(np.uint8)))
                #ax[row,col].set_title("train image")
                #ax[row,col].set_axis_off()

                ax[i, 0].imshow(img)
                ax[i, 1].imshow(mask, interpolation="nearest")
                ax[i, 0].set_title("train image")
                ax[i, 1].set_title("Ground truth mask")
                ax[i, 0].set_axis_off()
                ax[i, 1].set_axis_off()
         
                plt.tight_layout()
                #plt.show()
                figure.savefig("datacheck_segmentation.png")
    
        train_set = self.datasets['train']
        display_image_grid(train_set) 

