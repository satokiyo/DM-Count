import os
import time
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import numpy as np
import cv2
from datetime import datetime
from tqdm import tqdm
from logger import Logger

from utils.pytorch_utils import Save_Handle, AverageMeter
import utils.log_utils as log_utils
import utils.utils as utils

from datasets.crowd import Crowd_shBL, CellDatasetBL

from models.dmcount_model import DMCountModel
import neptune.new as neptune
from neptune.new.types import File

from losses.bay_loss import Bay_Loss
from losses.post_prob import Post_Prob


def train_collate(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[1]  # the number of points is not fixed, keep it as a list of tensor
    gt_discretes = torch.stack(transposed_batch[2], 0)
    return images, points, gt_discretes


class RegTrainer(object):
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
#        args.encoder_name = "resnet50" # dpn98 resnet152 vgg19_bn timm-resnest50d efficientnet-b5 timm-resnest50d_4s2x40d vgg19_bn mobilenet_v2 timm-efficientnet-lite4 timm-skresnext50_32x4d se_resnext50_32x4d timm-efficientnet-b6 se_resnext101_32x4d xception 
#           # efficientnet-b5 ロス計算のエラー出る
#        args.classes = 1
#        args.scale_pyramid_module=True, # add
#        args.use_attention_branch=True, # add
        self.model = DMCountModel(args)

        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # dataset
        downsample_ratio = args.downsample_ratio # for U-Net like architecture
        if args.dataset.lower() == 'qnrf':
            self.datasets = {x: Crowd_qnrf(os.path.join(args.data_dir, x),
                                           args.crop_size, downsample_ratio, x) for x in ['train', 'val']}
        elif args.dataset.lower() == 'nwpu':
            self.datasets = {x: Crowd_nwpu(os.path.join(args.data_dir, x),
                                           args.crop_size, downsample_ratio, x) for x in ['train', 'val']}
        elif args.dataset.lower() == 'sha' or args.dataset.lower() == 'shb':
            self.datasets = {'train': Crowd_shBL(os.path.join(args.data_dir, 'train_data'),
                                               args.crop_size, downsample_ratio, 'train', use_albumentation=args.use_albumentation),
                             'val': Crowd_shBL(os.path.join(args.data_dir, 'test_data'),
                                             args.crop_size, downsample_ratio, 'val'),
                             }
        elif args.dataset.lower() == 'cell':
            self.datasets = {'train': CellDatasetBL(os.path.join(args.data_dir, 'train'),
                                               args.crop_size, downsample_ratio, 'train', use_albumentation=args.use_albumentation),
                             'val': CellDatasetBL(os.path.join(args.data_dir, 'val'),
                                             args.crop_size, downsample_ratio, 'val'),
                             }
 
        else:
            raise NotImplementedError

        self.dataloaders = {x: DataLoader(self.datasets[x],
                                          collate_fn=(train_collate
                                                      if x == 'train' else default_collate),
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
        self.run["sys/tags"].add(['run-organization', 'me'])  # organize things

        # save dir setting
        sub_dir = 'encoder-{}_input-{}_wot-{}_wtv-{}_reg-{}_nIter-{}_normCood-{}'.format(
            args.encoder_name, args.crop_size, args.wot, args.wtv, args.reg, args.num_of_iter_in_ot, args.norm_cood)

        save_dir_root = os.path.join('ckpts', sub_dir)
        if not os.path.exists(save_dir_root):
            os.makedirs(save_dir_root)
        time_str = datetime.strftime(datetime.now(), '%m%d-%H%M%S')
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

        self.post_prob = Post_Prob(args.sigma,
                                   args.crop_size,
                                   args.downsample_ratio,
                                   args.background_ratio,
                                   args.use_background,
                                   self.device)
        self.criterion = Bay_Loss(args.use_background, self.device)
        self.save_list = Save_Handle(max_num=1)
        self.best_mae = np.inf
        self.best_mse = np.inf
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

        stream = tqdm(self.dataloaders['train'])
        #for step, (inputs, points, gt_discrete) in enumerate(self.dataloaders['train']):
        for step, (inputs, points, targets, st_sizes) in enumerate(stream):
            inputs = inputs.to(self.device)
            st_sizes = st_sizes.to(self.device)
            gd_count = np.array([len(p) for p in points], dtype=np.float32)
            points = [p.to(self.device) for p in points]
            targets = [t.to(self.device) for t in targets]

            with torch.set_grad_enabled(True):
                outputs = self.model(inputs)
                prob_list = self.post_prob(points, st_sizes)
                loss = self.criterion(prob_list, targets, outputs)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                N = inputs.size(0)
                pre_count = torch.sum(outputs.view(N, -1), dim=1).detach().cpu().numpy()
                res = pre_count - gd_count
                epoch_loss.update(loss.item(), N)
                epoch_mse.update(np.mean(res * res), N)
                epoch_mae.update(np.mean(abs(res)), N)


            stream.set_description('Epoch {} Train, Loss: {:.2f}, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                     .format(self.epoch, epoch_loss.get_avg(), np.sqrt(epoch_mse.get_avg()), epoch_mae.get_avg(),
                             time.time()-epoch_start))


            # show image
            if step % 50 == 0:
                # show images to original size (1枚目の画像だけを表示する)
                vis_img = outputs[0, 0].detach().cpu().numpy()
                # normalize density map values from 0 to 1, then map it to 0-255.
                vis_img = (vis_img - vis_img.min()) / (vis_img.max() - vis_img.min() + 1e-5)
                vis_img = (vis_img*255).astype(np.uint8)
                vis_img = cv2.applyColorMap(vis_img, cv2.COLORMAP_VIRIDIS)
                vis_img = vis_img.transpose(2,0,1)

                org_img = inputs[0].detach().cpu().numpy()
                org_img = (org_img - org_img.min()) / (org_img.max() - org_img.min() + 1e-5)
                org_img = (org_img*255).astype(np.uint8)
                # overlay
                overlay = ((org_img/4) + (vis_img/1.5))
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
                                        np.sqrt(epoch_mse.get_avg()),
                                        epoch_mae.get_avg(),],
                               iteration_number=self.epoch,
                               terms_legends=['Loss',
                                                'MSE',
                                                'MAE',])


        # Log training losses to neptune
        self.run['train/Loss'].log(epoch_loss.get_avg())
        self.run['train/MSE'].log(np.sqrt(epoch_mse.get_avg()))
        self.run['train/MAE'].log(epoch_mae.get_avg())


    def val_epoch(self):
        args = self.args
        epoch_start = time.time()
        self.model.eval()  # Set model to evaluate mode
        epoch_res = []
        stream = tqdm(self.dataloaders['val'])
        #for step, (inputs, count, name) in enumerate(self.dataloaders['val']):
        for step, (inputs, count, name) in enumerate(stream):
            inputs = inputs.to(self.device)
            assert inputs.size(0) == 1, 'the batch size should equal to 1 in validation mode'
            with torch.set_grad_enabled(False):
                outputs, _ = self.model(inputs)
                res = count[0].item() - torch.sum(outputs).item()
                epoch_res.append(res)

            # show image
            if step % 50 == 0:
                # show images to original size (1枚目の画像だけを表示する)
                vis_img = outputs[0, 0].detach().cpu().numpy()
                # normalize density map values from 0 to 1, then map it to 0-255.
                vis_img = (vis_img - vis_img.min()) / (vis_img.max() - vis_img.min() + 1e-5)
                vis_img = (vis_img*255).astype(np.uint8)
                vis_img = cv2.applyColorMap(vis_img, cv2.COLORMAP_VIRIDIS)
                vis_img = vis_img.transpose(2,0,1)

                org_img = inputs[0].detach().cpu().numpy()
                org_img = (org_img - org_img.min()) / (org_img.max() - org_img.min() + 1e-5)
                org_img = (org_img*255).astype(np.uint8)
                # overlay
                overlay = ((org_img/5) + (vis_img))

                # visdom
                self.vlog.image(imgs=[overlay],
                          titles=['(Validation) Image w/ output heatmap and labeled points'],
                          window_ids=[2])
                ## neptune
                #self.run['image_val'].upload(File.as_image(overlay.transpose(1,2,0)))

            mse = np.sqrt(np.mean(np.square(np.array(epoch_res))))
            mae = np.mean(np.abs(epoch_res))
            stream.set_description('Epoch {} Val, MSE: {:.2f} MAE: {:.2f}'.format(self.epoch, mse, mae))

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


    def data_check(self):
        import matplotlib.pyplot as plt

        def display_image_grid(train_dataset, num=30):
            cols = 2
            rows = num // cols
            figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(10, 50))
            for i in range(num):
                img, keypoints, gt_discrete = train_dataset[i]
                img = img.detach().cpu().numpy()
                img = (img - img.min()) / (img.max() - img.min() + 1e-5)
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

