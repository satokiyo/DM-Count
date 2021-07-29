import os
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim
from torch.utils.data import DataLoader
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
from datasets.classification_data import ClassificationDataset
from models.classification_model import ClassificationModel

import neptune.new as neptune
from neptune.new.types import File

from PIL import Image
import copy

class ClassificationTrainer(object):
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
        self.model = ClassificationModel(args).to(self.device)

        # criterion
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # scheduler
        #T_0=35 #Number of iterations for the first restart.
        #T_mult=1 # A factor increases after a restart. Default: 1.
        eta_min=1e-5 #Minimum learning rate. Default: 0.
        args.t_0 = int(args.max_epoch / 2)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, args.t_0, args.t_mult, eta_min)

        # dataset
        if args.dataset.lower() == 'classification':
            self.datasets = {'train': ClassificationDataset(os.path.join(args.data_dir), args.crop_size, 'train',
                                        use_albumentation=args.use_albumentation , use_copy_paste=args.use_copy_paste),
                             'val'  : ClassificationDataset(os.path.join(args.data_dir), args.crop_size, 'val')}
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

        # data check
        if args.data_check:
            self.data_check() # only check dataloader
            exit()


        # neptune
        self.run = neptune.init(project=f'{args.neptune_workspace_name}/{args.neptune_project_name}',
                           api_token=args.neptune_api_token,
                           source_files=['*.py','*.sh', 'models'])#, 'requirements.txt'])
        for arg in vars(args):
            self.run[f'param_{arg}'] = getattr(args, arg)
        self.run["sys/tags"].add(args.neptune_tag)  # tag

        # save dir setting
        sub_dir = 'encoder-{}_albumentation-{}_copy_paste-{}'.format(args.encoder_name, args.use_albumentation, args.use_copy_paste)


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


    def train(self):
        """training process"""
        args = self.args
        self.prev_model = copy.deepcopy(self.model)
        self.prev_optimizer = copy.deepcopy(self.optimizer)
        for epoch in range(self.start_epoch, args.max_epoch + 1):
            #self.logger.info('-' * 5 + 'Epoch {}/{}'.format(epoch, args.max_epoch) + '-' * 5)
            self.epoch = epoch
            self.train_eopch()
            if epoch % args.val_epoch == 0 and epoch >= args.val_start:
                self.val_epoch()

    def train_eopch(self):
        epoch_loss = AverageMeter()
        epoch_mae = AverageMeter()
        epoch_mse = AverageMeter()
        epoch_start = time.time()
        self.model.train()  # Set model to training mode

        stream = tqdm(self.dataloaders['train'], total=len(self.dataloders['train']))
        for step, (x_l, x_ul) in enumerate(stream):
            inputs, masks = x_l
            if self.args.use_ssl:
                x_ul = x_ul.to(self.device)
            inputs = inputs.to(self.device)
            masks  = masks.squeeze(1).to(self.device) # squeeze ch(1ch)
            N = inputs.size(0)
            with torch.set_grad_enabled(True):
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

                epoch_loss.update(loss.item(), N)

                if torch.isnan(loss):
                    self.model = self.prev_model
                    self.optimizer.load_state_dict(self.prev_optimizer.state_dict())
                else:
                    self.prev_model = copy.deepcopy(self.model)
                    self.prev_optimizer = copy.deepcopy(self.optimizer)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                self.scheduler.step(self.epoch + step / iters)

            stream.set_description(
                'Epoch {} Train, Loss: {:.2f}'
                    .format(self.epoch, epoch_loss.get_avg()))
            del loss
 
            # show image
            if step % 20 == 0:
               with torch.set_grad_enabled(False):
                    # show images to original size (1枚目の画像だけを表示する)
                    PALETTE = [
                        0,0,0,
                        0,255,0,
                        255,0,0,
                        0,0,255
                    ]
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


    def val_epoch(self):
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
                outputs, intermediates = self.model(inputs)
                del intermediates
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
                    pos = confusion_matrix[..., i].sum(1) # pred
                    res = confusion_matrix[..., i].sum(0) # label
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

                stream.set_description('Epoch {} Val, Loss: {:.2f} iou0: {:.2f} iou1: {:.2f} iou2: {:.2f} iou3: {:.2f} miou: {:.2f}'
                                              .format(self.epoch,
                                               np.mean(np.array(epoch_loss)),
                                               np.mean(np.array(epoch_iou_class['0'])),
                                               np.mean(np.array(epoch_iou_class['1'])),
                                               np.mean(np.array(epoch_iou_class['2'])),
                                               np.mean(np.array(epoch_iou_class['3'])),
                                               np.mean(np.array(epoch_miou))))
    
                # show image
                if step % 20 == 0:
                    PALETTE = [
                        0,0,0,
                        0,255,0,
                        255,0,0,
                        0,0,255
                    ]
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
 
 

    def data_check(self):
        import matplotlib.pyplot as plt
        from PIL import Image

        PALETTE = [
            0,0,0,
            0,255,0,
            255,0,0,
            0,0,255
        ]

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

