import os
import time
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import numpy as np
import cv2
from datetime import datetime, timedelta, timezone
from tqdm import tqdm
from logger import Logger

from utils.pytorch_utils import Save_Handle, AverageMeter
import utils.log_utils as log_utils
import utils.utils as utils

from datasets.crowd import Crowd_qnrf, Crowd_nwpu, Crowd_sh, CellDataset
#from models.scale_pyramid_module import Vgg16Spm, Resnet50Spm
from losses.ot_loss import OT_Loss

from models.dmcount_model import DMCountModel
import neptune.new as neptune
from neptune.new.types import File
#from itertools import cycle
import copy
import torch.nn.functional as F


def cycle(iterable):
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


def train_collate(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[1]  # the number of points is not fixed, keep it as a list of tensor
    gt_discretes = torch.stack(transposed_batch[2], 0)
    return images, points, gt_discretes


class Trainer(object):
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
        self.model = DMCountModel(args)

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
        #args.t_0 = int(args.max_epoch // 2)
        args.t_0 = 10
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, args.t_0, args.t_mult, eta_min)

        # dataset
        downsample_ratio = args.downsample_ratio # for U-Net like architecture
        if args.dataset.lower() == 'qnrf':
            self.datasets = {x: Crowd_qnrf(os.path.join(args.data_dir, x),
                                           args.crop_size, downsample_ratio, x) for x in ['train', 'val']}
        elif args.dataset.lower() == 'nwpu':
            self.datasets = {x: Crowd_nwpu(os.path.join(args.data_dir, x),
                                           args.crop_size, downsample_ratio, x) for x in ['train', 'val']}
        elif args.dataset.lower() == 'sha' or args.dataset.lower() == 'shb':
            self.datasets = {'train': Crowd_sh(os.path.join(args.data_dir, 'train_data'),
                                               args.crop_size, downsample_ratio, 'train', use_albumentation=args.use_albumentation),
                             'val': Crowd_sh(os.path.join(args.data_dir, 'test_data'),
                                             args.crop_size, downsample_ratio, 'val'),
                             }
        elif args.dataset.lower() == 'cell':
            self.datasets = {'train': CellDataset(os.path.join(args.data_dir, 'train'),
                                               args.crop_size, args.resize, downsample_ratio, 'train', use_albumentation=args.use_albumentation),
                             'val': CellDataset(os.path.join(args.data_dir, 'val'),
                                             args.crop_size, args.resize, downsample_ratio, 'val'),
                             }
            self.dataset_ul = CellDataset(os.path.join(args.data_dir_ul, 'ssl'),
                                            args.crop_size,
                                            args.resize,
                                            downsample_ratio,
                                            'val_no_gt')
 
        else:
            raise NotImplementedError

        self.dataloaders = {x: DataLoader(self.datasets[x],
                                          collate_fn=(train_collate
                                                      if x == 'train' else default_collate),
                                          batch_size=(args.batch_size
                                                      if x == 'train' else 1),
                                          shuffle=(True if x == 'train' else False),
                                          drop_last=(True if x == 'train' else False),
                                          num_workers=args.num_workers * self.device_count,
                                          pin_memory=(True if x == 'train' else False))
                            for x in ['train', 'val']}

        self.dataloader_ul = DataLoader(self.dataset_ul,
                                          collate_fn=default_collate,
                                          batch_size=args.batch_size_ul,
                                          shuffle=True,
                                          drop_last=True,
                                          num_workers=args.num_workers * self.device_count,
                                          pin_memory=True)

        if args.use_ssl:
            from losses.losses import softmax_kl_loss, softmax_mse_loss, softmax_js_loss, consistency_weight
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
        sub_dir = 'encoder-{}_input-{}_wot-{}_wtv-{}_reg-{}_nIter-{}_normCood-{}'.format(
            args.encoder_name, args.crop_size, args.wot, args.wtv, args.reg, args.num_of_iter_in_ot, args.norm_cood)

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
        # 20210428 tmp
        if args.crop_size != args.resize:
            self.ot_loss = OT_Loss(args.resize, downsample_ratio, args.norm_cood, self.device, args.num_of_iter_in_ot, args.reg)
        else:
            self.ot_loss = OT_Loss(args.crop_size, downsample_ratio, args.norm_cood, self.device, args.num_of_iter_in_ot, args.reg)
        self.tv_loss = nn.L1Loss(reduction='none').to(self.device)
        self.mse = nn.MSELoss().to(self.device)
        self.mae = nn.L1Loss().to(self.device)
        self.save_list = Save_Handle(max_num=1)
        self.best_mae = np.inf
        self.best_mse = np.inf
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
        epoch_ot_loss = AverageMeter()
        epoch_ot_obj_value = AverageMeter()
        epoch_wd = AverageMeter()
        epoch_count_loss = AverageMeter()
        epoch_tv_loss = AverageMeter()
        epoch_loss = AverageMeter()
        epoch_mae = AverageMeter()
        epoch_mse = AverageMeter()
        epoch_start = time.time()
        self.model.train()  # Set model to training mode

        if self.args.use_ssl:
            iters = len(self.dataloader_ul)
            stream = tqdm(zip(cycle(self.dataloaders['train']), self.dataloader_ul), total=iters)
            #stream = tqdm(range(len(self.dataloader_ul)), ncols=135)
            #dataloader = iter(zip(cycle(self.dataloaders['train']), self.dataloader_ul))
        else:
            iters = len(self.dataloaders['train'])
            stream = tqdm(zip(self.dataloaders['train'], cycle(self.dataloader_ul)), total=iters)
            #stream = tqdm(range(len(self.dataloaders['train'])), ncols=135)
            #dataloader = iter(zip(self.dataloaders['train'], cycle(self.dataloader_ul)))
        for step, (x_l, x_ul) in enumerate(stream):
            inputs, points, gt_discrete = x_l
            if self.args.use_ssl:
                x_ul, name = x_ul
                x_ul = x_ul.to(self.device)
            else:
                del x_ul
            inputs = inputs.to(self.device)
            gd_count = np.array([len(p) for p in points], dtype=np.float32)
            points = [p.to(self.device) for p in points]
            gt_discrete = gt_discrete.to(self.device)
            N = inputs.size(0)

            with torch.set_grad_enabled(True):
                if self.args.deep_supervision:
                    outputs, intermediates = self.model(inputs)
                else:
                    outputs = self.model(inputs)

                B, C, H, W = outputs.size()
                mu_sum = outputs.view([B, -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
                outputs_normed = outputs / (mu_sum + 1e-6)

                # Compute OT loss.
                ot_loss, wd, ot_obj_value = self.ot_loss(outputs_normed, outputs, points, self.epoch)
                ot_loss = ot_loss * self.args.wot
                ot_obj_value = ot_obj_value * self.args.wot
                epoch_ot_loss.update(ot_loss.item(), N)
                epoch_ot_obj_value.update(ot_obj_value.item(), N)
                epoch_wd.update(wd, N)
                del wd, ot_obj_value

                # Compute counting loss.
                count_loss = self.mae(outputs.sum(1).sum(1).sum(1),
                                      torch.from_numpy(gd_count).float().to(self.device))
                epoch_count_loss.update(count_loss.item(), N)

                # Compute TV loss.
                gd_count_tensor = torch.from_numpy(gd_count).float().to(self.device).unsqueeze(1).unsqueeze(
                    2).unsqueeze(3)
                gt_discrete_normed = gt_discrete / (gd_count_tensor + 1e-6)
                tv_loss = (self.tv_loss(outputs_normed, gt_discrete_normed).sum(1).sum(1).sum(
                    1) * torch.from_numpy(gd_count).float().to(self.device)).mean(0) * self.args.wtv
                epoch_tv_loss.update(tv_loss.item(), N)
                del outputs_normed, gd_count_tensor, gt_discrete_normed

                loss = ot_loss + count_loss + tv_loss
                del ot_loss, count_loss, tv_loss

                if self.args.deep_supervision:
                    w_deep_sup = 0.4
                    w_each = w_deep_sup / len(intermediates)
                    #deep_sup_loss = [w_each*self.criterion(out, masks) for out in intermediates]
                    deep_sup_loss = []
                    for i in intermediates:
                        B, C, H, W = i.size()
                        mu_sum = i.view([B, -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
                        outputs_normed = i / (mu_sum + 1e-6)
                        ot_loss, wd, ot_obj_value = self.ot_loss(outputs_normed, i, points, self.epoch)
                        del wd, ot_obj_value, i, B, C, H, W
                        ot_loss = ot_loss * w_each
                        deep_sup_loss.append(ot_loss)
                        del ot_loss
                    loss = loss + torch.stack(deep_sup_loss).sum()
                    del intermediates, deep_sup_loss

                if self.args.use_ssl: # semi-supervised
                    output_ul_main, outputs_ul_aux = self.model(x_ul, unsupervised=True)
                    del x_ul
                    #targets = F.softmax(output_ul_main.detach(), dim=1) # main decoder output
                    targets = output_ul_main.detach() # main decoder output
                    loss_unsup = []
                    for aux_out in outputs_ul_aux: # aux decoder outputs
                        loss_unsup.append(self.unsuper_loss(inputs=aux_out, targets=targets, conf_mask=False, threshold=None, use_softmax=True))
                        del aux_out
                    # Compute the unsupervised loss
                    loss_unsup = sum(loss_unsup) / len(loss_unsup)
                    weight_u = self.unsup_loss_w(epoch=self.epoch, curr_iter=step)
                    loss_unsup = loss_unsup * weight_u
                    loss = loss + loss_unsup

                    del output_ul_main, outputs_ul_aux, loss_unsup

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

                pred_count = torch.sum(outputs.view(N, -1), dim=1).detach().cpu().numpy()
                pred_err = pred_count - gd_count
                epoch_loss.update(loss.item(), N)
                epoch_mse.update(np.mean(pred_err * pred_err), N)
                epoch_mae.update(np.mean(abs(pred_err)), N)
                del loss

            stream.set_description(
                'Epoch {} Train, Loss: {:.2f}, OT Loss: {:.2e}, Wass Distance: {:.2f}, OT obj value: {:.2f}, '
                'Count Loss: {:.2f}, TV Loss: {:.2f}, MSE: {:.2f} MAE: {:.2f}'
                    .format(self.epoch, epoch_loss.get_avg(), epoch_ot_loss.get_avg(), epoch_wd.get_avg(),
                            epoch_ot_obj_value.get_avg(), epoch_count_loss.get_avg(), epoch_tv_loss.get_avg(),
                            np.sqrt(epoch_mse.get_avg()), epoch_mae.get_avg()))
 
            # show image
            if step % 10 == 0:
                # show images to original size (1枚目の画像だけを表示する)
                vis_img = outputs[0, 0].detach().cpu().numpy()
                del outputs
                # normalize density map values from 0 to 1, then map it to 0-255.
                vis_img = (vis_img - np.min(vis_img)) / np.ptp(vis_img)
                vis_img = (vis_img*255).astype(np.uint8)
                vis_img = cv2.applyColorMap(vis_img, cv2.COLORMAP_VIRIDIS)
#                vis_img = cv2.resize(vis_img, dsize=(int(self.args.input_size), int(self.args.input_size)), interpolation=cv2.INTER_NEAREST)
                vis_img = vis_img.transpose(2,0,1) # cv2.resize returns cv image
                org_img = inputs[0].detach().cpu().numpy()
                org_img = (org_img - np.min(org_img)) / np.ptp(org_img)
                org_img = (org_img*255).astype(np.uint8)
                # overlay
                overlay = ((org_img/2) + (vis_img/2))
                # paint annotation point
                points = points[0].detach().cpu().numpy()
                if(len(points)>0):
                    points[:,[0,1]] = points[:,[1,0]] # x,y -> y,x
                    show_img = utils.paint_circles(img=overlay,
                                                       points=points,
                                                       color='white')
                    # visdom
                    self.vlog.image(imgs=[show_img],
                              titles=['(Training) Image w/ output heatmap and labeled points'],
                              window_ids=[1])
                    ## neptune
                    #self.run['image_train'].upload(File.as_image(show_img.transpose(1,2,0)))


        #self.logger.info(
        #    'Epoch {} Train, Loss: {:.2f}, OT Loss: {:.2e}, Wass Distance: {:.2f}, OT obj value: {:.2f}, '
        #    'Count Loss: {:.2f}, TV Loss: {:.2f}, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
        #        .format(self.epoch, epoch_loss.get_avg(), epoch_ot_loss.get_avg(), epoch_wd.get_avg(),
        #                epoch_ot_obj_value.get_avg(), epoch_count_loss.get_avg(), epoch_tv_loss.get_avg(),
        #                np.sqrt(epoch_mse.get_avg()), epoch_mae.get_avg(),
        #                time.time() - epoch_start))
        model_state_dic = self.model.state_dict()
        save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(self.epoch))
        torch.save({
            'epoch': self.epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_state_dict': model_state_dic
        }, save_path)
        self.save_list.append(save_path)

        # Log training losses to visdom
        self.vlog.train_losses(terms=[epoch_loss.get_avg(),
                                        epoch_ot_loss.get_avg(),
                                        epoch_wd.get_avg(),
                                        epoch_ot_obj_value.get_avg(),
                                        epoch_count_loss.get_avg(),
                                        epoch_tv_loss.get_avg(),
                                        np.sqrt(epoch_mse.get_avg()),
                                        epoch_mae.get_avg(),],
                               iteration_number=self.epoch,
                               terms_legends=['Loss',
                                                'OT Loss',
                                                'Wass Distance',
                                                'OT obj value',
                                                'Count Loss',
                                                'TV Loss',
                                                'MSE',
                                                'MAE',])
        # Log training losses to neptune
        self.run['train/Loss'].log(epoch_loss.get_avg())
        self.run['train/OT Loss'].log(epoch_ot_loss.get_avg())
        self.run['train/Wass Distance'].log(epoch_wd.get_avg())
        self.run['train/OT obj value'].log(epoch_ot_obj_value.get_avg())
        self.run['train/Count Loss'].log(epoch_count_loss.get_avg())
        self.run['train/TV Loss'].log(epoch_tv_loss.get_avg())
        self.run['train/MSE'].log(np.sqrt(epoch_mse.get_avg()))
        self.run['train/MAE'].log(epoch_mae.get_avg())
        self.run['train/lr'].log(self.scheduler.get_last_lr())


    def val_epoch(self):
        args = self.args
        epoch_start = time.time()
        self.model.eval()  # Set model to evaluate mode
        epoch_ot_loss = []
        epoch_tv_loss = []
        epoch_ot_obj_value = []
        epoch_wd = []
        epoch_res = []
        stream = tqdm(self.dataloaders['val'])
        for step, (inputs, points, gt_discrete, name) in enumerate(stream):
            inputs = inputs.to(self.device)
            gd_count = np.array([len(p) for p in points], dtype=np.float32)
            points = [p.to(self.device) for p in points]
            gt_discrete = gt_discrete.to(self.device)
            assert inputs.size(0) == 1, 'the batch size should equal to 1 in validation mode'
            with torch.set_grad_enabled(False):
                #x = self.model(inputs)
                #mu = x
                #B, C, H, W = mu.size()
                #mu_sum = mu.view([B, -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
                #mu_normed = mu / (mu_sum + 1e-6)
                #outputs, outputs_normed = mu, mu_normed
                #del x, mu, mu_normed

                if self.args.deep_supervision:
                    outputs, intermediates = self.model(inputs)
                else:
                    outputs = self.model(inputs)

                B, C, H, W = outputs.size()
                mu_sum = outputs.view([B, -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
                outputs_normed = outputs / (mu_sum + 1e-6)

                # Compute OT loss.
                ot_loss, wd, ot_obj_value = self.ot_loss(outputs_normed, outputs, points, self.epoch)
                ot_loss = ot_loss * self.args.wot
                ot_obj_value = ot_obj_value * self.args.wot
                epoch_ot_loss.append(ot_loss.item())
                epoch_ot_obj_value.append(ot_obj_value.item())
                epoch_wd.append(wd)
                # Compute MSE,MAE
                count = len(points[0])
                res = count - torch.sum(outputs).item()
                #res = count[0].item() - torch.sum(outputs).item()
                epoch_res.append(res)
                # Compute TV loss.
                gd_count_tensor = torch.from_numpy(gd_count).float().to(self.device).unsqueeze(1).unsqueeze(
                    2).unsqueeze(3)
                gt_discrete_normed = gt_discrete / (gd_count_tensor + 1e-6)
                tv_loss = (self.tv_loss(outputs_normed, gt_discrete_normed).sum(1).sum(1).sum(
                    1) * torch.from_numpy(gd_count).float().to(self.device)).mean(0) * self.args.wtv
                epoch_tv_loss.append(tv_loss.item())
 

            # show image
            if step % 10 == 0:
                # show images to original size (1枚目の画像だけを表示する)
                vis_img = outputs[0, 0].detach().cpu().numpy()
                del outputs
                # normalize density map values from 0 to 1, then map it to 0-255.
                vis_img = (vis_img - np.min(vis_img)) / np.ptp(vis_img)
                vis_img = (vis_img*255).astype(np.uint8)
                vis_img = cv2.applyColorMap(vis_img, cv2.COLORMAP_VIRIDIS)
#                vis_img = cv2.resize(vis_img, dsize=(int(self.args.input_size), int(self.args.input_size)), interpolation=cv2.INTER_NEAREST)
                vis_img = vis_img.transpose(2,0,1) # cv2.resize returns cv image
                org_img = inputs[0].detach().cpu().numpy()
                org_img = (org_img - np.min(org_img)) / np.ptp(org_img)
                org_img = (org_img*255).astype(np.uint8)
                # overlay
                overlay = ((org_img/2) + (vis_img/2))

                # visdom
                self.vlog.image(imgs=[overlay],
                          titles=['(Validation) Image w/ output heatmap and labeled points'],
                          window_ids=[2])
                ## neptune
                #self.run['image_val'].upload(File.as_image(overlay.transpose(1,2,0)))

            ot_loss = np.mean(np.array(epoch_ot_loss))
            ot_obj = np.mean(np.array(epoch_ot_obj_value))
            wd = np.mean(np.array(epoch_wd))
            mse = np.sqrt(np.mean(np.square(np.array(epoch_res))))
            mae = np.mean(np.abs(epoch_res))
            tv_loss = np.mean(np.array(epoch_tv_loss))
            stream.set_description('Epoch {} Val, MSE: {:.2f} MAE: {:.2f} OT Loss: {:.2f} OT obj value: {:.2f} Wass Distance: {:.2f} TV Loss: {:.2f}'.format(self.epoch, mse, mae, ot_loss, ot_obj, wd, tv_loss))

        #epoch_res = np.array(epoch_res)
        #mse = np.sqrt(np.mean(np.square(epoch_res)))
        #mae = np.mean(np.abs(epoch_res))
        #self.logger.info('Epoch {} Val, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
        #                 .format(self.epoch, mse, mae, time.time() - epoch_start))

        model_state_dic = self.model.state_dict()
        if (2.0 * mse + mae) < (2.0 * self.best_mse + self.best_mae):
            self.best_mse = mse
            self.best_mae = mae
            self.logger.info("save best mse {:.2f} mae {:.2f} model epoch {}".format(self.best_mse,
                                                                                     self.best_mae,
                                                                                     self.epoch))
            torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model_{}.pth'.format(self.best_count)))
            self.best_count += 1

        # Log validation losses
        self.vlog.val_losses(terms=[mse,mae],
                               iteration_number=self.epoch,
                               terms_legends=['MSE', 'MAE',])
        # Log validation losses to neptune
        self.run['val/MSE'].log(mse)
        self.run['val/MAE'].log(mae)
        self.run['val/OT Loss'].log(ot_loss)
        self.run['val/Wass Distance'].log(ot_obj)
        self.run['val/OT obj value'].log(wd)
        self.run['val/TV Loss'].log(tv_loss)


    def data_check(self):
        import matplotlib.pyplot as plt

        def display_image_grid(train_dataset, num=30):
            cols = 2
            rows = num // cols
            figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(10, 50))
            for i in range(num):
                img, keypoints, gt_discrete = train_dataset[i]
                img = img.detach().cpu().numpy()
                img = (img - np.min(img)) / np.ptp(img)
                img = (img*255).astype(np.uint8)
                # paint annotation point
                points = keypoints.detach().cpu().numpy()
                if(len(points)>0):
                    row = i // cols
                    col = i % cols
                    
                    points[:,[0,1]] = points[:,[1,0]] # x,y -> y,x
                    img = utils.paint_circles(img=img,
                                              points=points,
                                              color='white')
                    img = img.transpose(1,2,0)
    
                    ax[row,col].imshow(img)
                    #ax[i, 0].imshow(img)
                    #ax[i, 1].imshow(mask, interpolation="nearest")
                    ax[row,col].set_title("train image")
                    #ax[i, 0].set_title("train image")
                    #ax[i, 1].set_title("Ground truth mask")
                    ax[row,col].set_axis_off()
                    #ax[i, 0].set_axis_off()
                    #ax[i, 1].set_axis_off()
            
                plt.tight_layout()
                #plt.show()
                figure.savefig("train.png")
    
        train_set = self.datasets['train']
        display_image_grid(train_set) 

