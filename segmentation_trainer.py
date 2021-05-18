import os
import time
import torch
import torch.nn as nn
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
from datasets.segmentation_data import SegDataset
from models.segmentation_model import SegModel

import neptune.new as neptune
from neptune.new.types import File

from PIL import Image

class SegTrainer(object):
    def __init__(self, args):
        self.args = args

    def setup(self):
        args = self.args

        # device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            assert self.device_count == 1
        else:
            raise Exception("gpu is not available")

        # Model
        self.model = SegModel(args)

        #from kiunet import kiunet, densekiunet, reskiunet
        #self.model = kiunet()

        #from models.hrnet.models import seg_hrnet, seg_hrnet_ocr
        #from models.hrnet.config import config
        #from models.hrnet.config import update_config
        #update_config(config, args)
        #self.model = seg_hrnet.get_seg_model(config)
        #self.model = seg_hrnet_ocr.get_seg_model(config)

        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        #self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        #T_0=35 #Number of iterations for the first restart.
        #T_mult=1 # A factor increases after a restart. Default: 1.
        eta_min=1e-5 #Minimum learning rate. Default: 0.
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, args.t_0, args.t_mult, eta_min)

        # dataset
        downsample_ratio = args.downsample_ratio # for U-Net like architecture
        if args.dataset.lower() == 'segmentation':
            self.datasets = {'train': SegDataset(os.path.join(args.data_dir, 'images', 'training'),
                                                 os.path.join(args.data_dir, 'annotations', 'training'),
                                                 args.crop_size, downsample_ratio, 'train', use_albumentation=args.use_albumentation),
                             'val': SegDataset(os.path.join(args.data_dir, 'images', 'validation'),
                                               os.path.join(args.data_dir, 'annotations', 'validation'),
                                               args.crop_size, downsample_ratio, 'val'),
                             }
        else:
            raise NotImplementedError

        self.dataloaders = {x: DataLoader(self.datasets[x],
                                          collate_fn=default_collate,
                                          batch_size=(args.batch_size
                                                      if x == 'train' else 1),
                                          shuffle=(True if x == 'train' else False),
                                          num_workers=args.num_workers * self.device_count,
                                          pin_memory=(True if x == 'train' else False))
                            for x in ['train', 'val']}

        
        # data check
        if args.data_check:
            self.data_check() # only check dataloader
            exit()

        # neptune
        self.run = neptune.init(project='satokiyo/{}'.format(args.project),
                           api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwMTAxZjFkZS1jODNmLTQ2MWQtYWJhYi1kZTM5OGQ3NWYyZDAifQ==',
                           source_files=['*.py','*.sh'])#, 'requirements.txt'])
        for arg in vars(args):
            self.run[f'param_{arg}'] = getattr(args, arg)
        self.run["sys/tags"].add(args.neptune_tag)  # tag

        # save dir setting
        #sub_dir = 'encoder-{}_input-{}_wot-{}_wtv-{}_reg-{}_nIter-{}_normCood-{}'.format(
        sub_dir = 'encoder-{}_input-{}_'.format(
            args.encoder_name, args.crop_size)

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

        self.cce = nn.CrossEntropyLoss().to(self.device)
#        self.mse = nn.MSELoss().to(self.device)
#        self.mae = nn.L1Loss().to(self.device)
        self.save_list = Save_Handle(max_num=1)
        #self.best_mae = np.inf
        #self.best_mse = np.inf
        self.best_loss = np.inf
        self.best_count = 0


    def train(self):
        """training process"""
        args = self.args
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
        iters = len(self.dataloaders['train'])
        stream = tqdm(self.dataloaders['train'])
        for step, (inputs, masks) in enumerate(stream):
            inputs = inputs.to(self.device)
            masks  = masks.squeeze(1).to(self.device) # squeeze ch(1ch)
            N = inputs.size(0)
            with torch.set_grad_enabled(True):
                outputs = self.model(inputs)
                # Compute loss.
                cce = self.cce(outputs, masks)
#                mse = self.mse(outputs, masks)
#                mae = self.mae(outputs, masks)
                #pred_err = outputs - masks
                epoch_loss.update(cce.item(), N)
                #epoch_mse.update(torch.mean(pred_err * pred_err), N)
                #epoch_mae.update(torch.mean(torch.abs(pred_err)), N)

                self.optimizer.zero_grad()
                cce.backward()
                self.optimizer.step()
                self.scheduler.step(self.epoch + step / iters)

            stream.set_description(
                'Epoch {} Train, Loss: {:.2f}'
                    .format(self.epoch, epoch_loss.get_avg()))
                #'Epoch {} Train, Loss: {:.2f}, MSE: {:.2f} MAE: {:.2f}'
                #    .format(self.epoch, epoch_loss.get_avg(), torch.sqrt(epoch_mse.get_avg()), epoch_mae.get_avg()))
 
            # show image
            if step % 50 == 0:
                # show images to original size (1枚目の画像だけを表示する)
                PALETTE = [
                    0,0,0,
                    0,255,0,
                    255,0,0,
                    0,0,255
                ]
                vis_img = outputs[0].detach().cpu().numpy()
                # normalize density map values from 0 to 1, then map it to 0-255.
                vis_img = (vis_img - np.min(vis_img)) / np.ptp(vis_img)
                vis_img = (vis_img*255).astype(np.uint8)
                vis_map = np.argmax(vis_img, axis=0)
                vis_map = Image.fromarray(vis_map.astype(np.uint8), mode="P")
                vis_map.putpalette(PALETTE)
                vis_map = np.array(vis_map.convert("RGB"))
#                vis_map = vis_map.transpose(1,2,0)
#                vis_img = cv2.resize(vis_img, dsize=(int(self.args.input_size), int(self.args.input_size)), interpolation=cv2.INTER_NEAREST)
                org_img = inputs[0].detach().cpu().numpy().transpose(1,2,0)
                org_img = (org_img - np.min(org_img)) / np.ptp(org_img)
                org_img = (org_img*255).astype(np.uint8)
                # overlay
                #overlay = ((org_img/4) + (vis_map/1.5))
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
                                        #epoch_mse.get_avg(),
                                        #epoch_mae.get_avg(),],
                               iteration_number=self.epoch,
                               terms_legends=['Loss'])#,
                                                #'MSE',
                                                #'MAE',])
        # Log training losses to neptune
        self.run['train/Loss'].log(epoch_loss.get_avg())
        #self.run['train/MSE'].log(epoch_mse.get_avg())
        #self.run['train/MAE'].log(epoch_mae.get_avg())
        self.run['train/lr'].log(self.scheduler.get_last_lr())


    def val_epoch(self):
        args = self.args
        epoch_start = time.time()
        self.model.eval()  # Set model to evaluate mode
        epoch_loss = []
        epoch_mse = []
        epoch_mae = []
        stream = tqdm(self.dataloaders['val'])
        for step, (inputs, masks) in enumerate(stream):
            inputs = inputs.to(self.device)
            assert inputs.size(0) == 1, 'the batch size should equal to 1 in validation mode'
            masks  = masks.squeeze(1).to(self.device) # squeeze ch(1ch)
            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                cce = self.cce(outputs, masks)
                epoch_loss.append(cce.item())
                # Compute MSE,MAE
                #pred_err = outputs - masks
                #mse = torch.mean(pred_err * pred_err).detach().cpu().numpy()
                #mae = torch.mean(torch.abs(pred_err)).detach().cpu().numpy()
                #epoch_mse.append(mse)
                #epoch_mae.append(mae)

            # show image
            if step % 50 == 0:
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
                vis_map = np.array(vis_map.convert("RGB"))
                org_img = inputs[0].detach().cpu().numpy().transpose(1,2,0)
                org_img = (org_img - np.min(org_img)) / np.ptp(org_img)
                org_img = (org_img*255).astype(np.uint8)
                # overlay
                overlay = np.uint8((org_img/2) + (vis_map/2)).transpose(2,0,1)

                # visdom
                self.vlog.image(imgs=[overlay],
                          titles=['(Validation) Image w/ output heatmap and labeled points'],
                          window_ids=[2])
                ## neptune
                #self.run['image_val'].upload(File.as_image(overlay.transpose(1,2,0)))

            loss = np.mean(np.array(epoch_loss))
            #mse = np.mean(np.array(epoch_mse))
            #mae = np.mean(np.array(epoch_mae))
            #stream.set_description('Epoch {} Val, Loss: {:.2f} MSE: {:.2f} MAE: {:.2f}'.format(self.epoch, loss, mse, mae))
            stream.set_description('Epoch {} Val, Loss: {:.2f}'.format(self.epoch, loss))


        model_state_dic = self.model.state_dict()
        #if (2.0 * mse + mae) < (2.0 * self.best_mse + self.best_mae):
        #    self.best_mse = mse
        #    self.best_mae = mae
        #    self.logger.info("save best mse {:.2f} mae {:.2f} model epoch {}".format(self.best_mse,
        #                                                                             self.best_mae,
        #                                                                             self.epoch))
        #    torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model_{}.pth'.format(self.best_count)))
        #    self.best_count += 1

        if loss < self.best_loss:
            self.best_loss = loss
            self.logger.info("save best loss {:.2f} model epoch {}".format(self.best_loss, self.epoch))
            torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model_{}.pth'.format(self.best_count)))
            self.best_count += 1

        # Log validation losses
        #self.vlog.val_losses(terms=[loss,mse,mae],
        self.vlog.val_losses(terms=[loss],
                               iteration_number=self.epoch,
                               terms_legends=['LOSS'])
                               #terms_legends=['LOSS', 'MSE', 'MAE',])
        # Log validation losses to neptune
        self.run['val/Loss'].log(loss)
        #self.run['val/MSE'].log(mse)
        #self.run['val/MAE'].log(mae)



    def data_check(self):
        import matplotlib.pyplot as plt
        from PIL import Image

        PALETTE = [
            0,0,0,
            0,255,0,
            255,0,0,
            0,0,255
        ]

        def display_image_grid(train_dataset, num=30):
            cols = 2
            rows = num // cols
            figure, ax = plt.subplots(nrows=rows*2, ncols=cols, figsize=(10, 50))
            for i in range(num):
                img, mask = train_dataset[i]
                img = img.detach().cpu().numpy()
                mask = mask.detach().cpu().numpy()
                img = (img - np.min(img)) / np.ptp(img)
                img = (img*255).astype(np.uint8)
                mask = (mask*255).astype(np.uint8)
                img = img.transpose(1,2,0)
                mask = mask.transpose(1,2,0)
                mask = Image.fromarray(mask.squeeze(1), mode="P")
                mask.putpalette(PALETTE)

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

