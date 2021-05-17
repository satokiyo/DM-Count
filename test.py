import argparse
import torch
import os
import numpy as np
import datasets.crowd as crowd
#from models import vgg19 , vgg16
from models.dmcount_model import DMCountModel
import time
import utils.utils as utils

start = time.time()

parser = argparse.ArgumentParser(description='Test ')
parser.add_argument('--device', default='0', help='assign device')
parser.add_argument('--crop-size', type=int, default=512,
                    help='the crop size of the train image')
parser.add_argument('--model-path', type=str, default='pretrained_models/model_qnrf.pth',
                    help='saved model path')
parser.add_argument('--data-path', type=str,
                    default='data/QNRF-Train-Val-Test',
                    help='saved model path')
parser.add_argument('--dataset', type=str, default='qnrf',
                    help='dataset name: qnrf, nwpu, sha, shb')
parser.add_argument('--pred-density-map-path', type=str, default='',
                    help='save predicted density maps when pred-density-map-path is not empty.')
parser.add_argument('--test-type', type=str, default='val_with_gt',) # 'val':count only, 'val_with_gt' : show gt overlay, 'test_no_gt': without gt
parser.add_argument('--encoder_name', type=str, default='vgg19_bn')  # dpn98 resnet152 vgg19_bn timm-resnest50d efficientnet-b5 timm-resnest50d_4s2x40d vgg19_bn mobilenet_v2 timm-efficientnet-lite4 timm-skresnext50_32x4d se_resnext50_32x4d timm-efficientnet-b6 se_resnext101_32x4d xception 
parser.add_argument('--classes', type=int, default=1)
parser.add_argument('--scale_pyramid_module', type=int, default=0,) 
parser.add_argument('--use_attention_branch', type=int, default=0,) 
parser.add_argument('--downsample-ratio', type=int, default=1)
parser.add_argument('--input-size', type=int, default=512) 
parser.add_argument('--cfg', type=str, default='')
parser.add_argument('opts',
                    help="Modify config options using the command-line",
                    default=None,
                    nargs=argparse.REMAINDER)



args = parser.parse_args()

def paint_center(args, img, liklihoodmap, taus=[-1], org_img=None):
    """ 'tau=-1 means dynamic Otsu thresholding. '
        'tau=-2 means Beta Mixture Model-based thresholding.') 
        RGB img org_img
        """
    # The estimated map must be thresholded to obtain estimated points
    #taus = [-1] # 5ROI/sec 
    #taus = [-2] # 10ROI/sec 
    for t, tau in enumerate(taus):
        if tau != -2:
            mask, _ = utils.threshold(liklihoodmap, tau)
        else:
            mask, _, mix = utils.threshold(liklihoodmap, tau)
        # Save thresholded map to disk
        cv2.imwrite(os.path.join(args.pred_density_map_path, str(name[0]) + f'mask_tau_{round(tau, 4)}.jpg'), mask)
        est_count_int = int(torch.sum(outputs).item())

        # method 1. detect center by GMM fitting
        #centroids_wrt_orig = utils.cluster(mask, est_count_int, max_mask_pts=500)

        # meghod 2. detect center by labeling
        # 膨張・収縮処理
        #kernel = np.ones((2, 2), np.uint8)
        ##mask = cv2.dilate(mask, kernel)
        ##mask = cv2.erode(mask, kernel)
        ##mask = cv2.erode(mask, kernel)
        #mask = cv2.erode(mask, kernel)
        ##mask = cv2.erode(mask, kernel)
        ##mask = cv2.dilate(mask, kernel)
        #nLabels, labelImages, data, center = cv2.connectedComponentsWithStats(mask)
        #centroids_wrt_orig = center[:,[1,0]].astype(np.uint32)

        # method 3. find local maxima
        kernel = np.ones((2, 2), np.uint8)
        mask_copy = mask.copy()
        dilate = cv2.dilate(mask_copy, kernel)
        erode = cv2.erode(mask_copy, kernel)
        #cv2.imwrite(os.path.join(args.pred_density_map_path, str(name[0]) + f'dilate.jpg'), dilate)
        #cv2.imwrite(os.path.join(args.pred_density_map_path, str(name[0]) + f'erode.jpg'), erode)
        peak = dilate - mask_copy
        flat = mask_copy - erode
        #cv2.imwrite(os.path.join(args.pred_density_map_path, str(name[0]) + f'mask_tau_{round(tau, 4)}_peak.jpg'), cv2.bitwise_not(peak))
        #cv2.imwrite(os.path.join(args.pred_density_map_path, str(name[0]) + f'mask_tau_{round(tau, 4)}_flat.jpg'), cv2.bitwise_not(flat))
        peak[flat > 0] = 255
        #cv2.imwrite(os.path.join(args.pred_density_map_path, str(name[0]) + f'mask_tau_{round(tau, 4)}_peak2.jpg'), cv2.bitwise_not(peak))
        con, hierarchy = cv2.findContours(peak,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        # compute the center of the contour
        center = []
        for c in con:
            M = cv2.moments(c)
            cX = int(M["m10"] / (M["m00"] + 1e-5))
            cY = int(M["m01"] / (M["m00"] + 1e-5))
            center.append(np.array([cX, cY]))
        center = np.array(center).astype(np.uint32)
        centroids_wrt_orig = center[:,[1,0]]
        print(f'estimated count: {est_count_int}, ncenters: {len(center)}')



        # Paint a cross at the estimated centroids
        img_with_x_n_map = utils.paint_circles(img=img,
                                               points=centroids_wrt_orig,
                                               color='red',
                                               crosshair=True)
        # Save to disk
        cv2.imwrite(os.path.join(args.pred_density_map_path, str(name[0]) + f'painted_on_estmap_tau_{round(tau, 4)}.jpg'), img_with_x_n_map.transpose(1,2,0))

        # voronoi
        rect = (0, 0, img.shape[1], img.shape[2])
        subdiv = cv2.Subdiv2D(rect)
        for p in center: # center[:,[1,0]]?
            subdiv.insert((int(p[0]), int(p[1])))
        facets, centers = subdiv.getVoronoiFacetList([])
        img_draw = img.transpose(1,2,0).copy()
        img_draw_DAB = org_img.transpose(1,2,0).copy()
        org_img_copy = org_img.transpose(1,2,0).copy()

        org_img_copy = org_img_copy.astype(np.float32)
        B = org_img_copy[:,:,0] # B-ch
        G = org_img_copy[:,:,1] # G-ch
        R = org_img_copy[:,:,2] # R-ch
        #e=1e-6
        BN = 255*np.divide(B, (B+G+R), out=np.zeros_like(B), where=(B+G+R)!=0) # ref.paper : Automated Selection of DAB-labeled Tissue for Immunohistochemical Quantification
        DAB = 255 - BN
        #cv2.imwrite(os.path.join(args.pred_density_map_path, str(name[0]) + f'BN{round(tau, 4)}.jpg'), BN)
        #cv2.imwrite(os.path.join(args.pred_density_map_path, str(name[0]) + f'DAB{round(tau, 4)}.jpg'), DAB)

        #cv2.polylines(img_draw, [f.astype(int) for f in facets], True, (255, 255, 255), thickness=2) # draw voronoi
        #cv2.imwrite(os.path.join(args.pred_density_map_path, str(name[0]) + f'painted_on_estmap_tau_{round(tau, 4)}_volonoi0.jpg'), img_draw)

        # voronoi with restricted radius
        mat = np.zeros((img_draw.shape[0], img_draw.shape[1]), np.uint8)
        facets = [f.astype(int) for f in facets]
        radius = 25
        cells = []
        for i,(center, points) in enumerate(zip(centers, facets)):
            mask1 = cv2.fillPoly(mat.copy(), [points], (255)) # make binary mask
            mask2 = cv2.circle(mat.copy(),(int(center[0]), int(center[1])), radius, (255), -1)
            intersection = mask1 & mask2
            con, hierarchy = cv2.findContours(intersection,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            # Create a mask image that contains the contour filled in
            mask3 = np.zeros_like(mat, np.uint8)
            mask3 = cv2.drawContours(mask3, con, -1, 255, 1)
            contour_area = np.count_nonzero(mask3) # 後で使うpixel単位面積
            mask4 = (mask3==255).astype(np.uint8) # contour region mask
            contour_DAB = DAB * mask4
            intensity_thres=175
            area_thres=0.1 # over 10% area
            over_thres_area = np.count_nonzero(contour_DAB > intensity_thres)

            if (over_thres_area / contour_area) > area_thres: # 1-NonNucleusArea > 0.1
                #描画
                img_draw_DAB = cv2.drawContours(img_draw_DAB , con, -1, (0,0,255), 1) # red
            else:
                img_draw_DAB = cv2.drawContours(img_draw_DAB , con, -1, (180,180,0), 1) # cyan

            img_draw = cv2.drawContours(img_draw, con, -1, (0,255,0), 1) # draw voronoi with restricted redius

        cv2.imwrite(os.path.join(args.pred_density_map_path, str(name[0]) + f'painted_on_estmap_tau_{round(tau, 4)}_volonoi.jpg'), img_draw) 
        cv2.imwrite(os.path.join(args.pred_density_map_path, str(name[0]) + f'painted_on_estmap_tau_{round(tau, 4)}_dab.jpg'), img_draw_DAB)
        

os.environ['CUDA_VISIBLE_DEVICES'] = args.device  # set vis gpu
device = torch.device('cuda')

model_path = args.model_path
crop_size = args.crop_size
data_path = args.data_path
if args.dataset.lower() == 'qnrf':
    dataset = crowd.Crowd_qnrf(os.path.join(data_path, 'test'), crop_size, 8, method='val')
elif args.dataset.lower() == 'nwpu':
    dataset = crowd.Crowd_nwpu(os.path.join(data_path, 'val'), crop_size, 8, method='val')
elif args.dataset.lower() == 'sha' or args.dataset.lower() == 'shb':
    dataset = crowd.Crowd_sh(os.path.join(data_path, 'test_data'), crop_size, 8, method='val')
elif args.dataset.lower() == 'cell':
    if args.test_type in ['val_with_gt', 'val']:
        dataset = crowd.CellDataset(os.path.join(data_path, 'val'), crop_size, 1,  method=args.test_type)
    elif args.test_type == 'test_no_gt':
        dataset = crowd.CellDataset(os.path.join(data_path, 'test'), crop_size, 1,  method=args.test_type)
 
else:
    raise NotImplementedError
dataloader = torch.utils.data.DataLoader(dataset, 1, shuffle=False,
                                         num_workers=1, pin_memory=True)

if args.pred_density_map_path:
    import cv2
    if not os.path.exists(args.pred_density_map_path):
        os.makedirs(args.pred_density_map_path)

model = DMCountModel(args)

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
    image_errs = []
    #for inputs, count, name in dataloader:
    for inputs, points, name in dataloader:
        inputs = inputs.to(device)
        assert inputs.size(0) == 1, 'the batch size should equal to 1'
        with torch.set_grad_enabled(False):
            outputs, _ = model(inputs)
        img_err = len(points[0]) - torch.sum(outputs).item()
        #img_err = count[0].item() - torch.sum(outputs).item()
    
        print(name, img_err, len(points[0]), torch.sum(outputs).item())
        image_errs.append(img_err)
    
        if args.pred_density_map_path:
            vis_img = outputs[0, 0].cpu().numpy()
            # normalize density map values from 0 to 1, then map it to 0-255.
            vis_img = (vis_img - vis_img.min()) / (vis_img.max() - vis_img.min() + 1e-5)
            vis_img2 = vis_img
            vis_img = (vis_img*255).astype(np.uint8)
            vis_img = cv2.applyColorMap(vis_img, cv2.COLORMAP_VIRIDIS)
            if args.downsample_ratio > 1:
                vis_img = cv2.resize(vis_img, dsize=(args.input_size, args.input_size), interpolation=cv2.INTER_NEAREST) #tmp
                vis_img2 = cv2.resize(vis_img2, dsize=(args.input_size, args.input_size), interpolation=cv2.INTER_NEAREST) #tmp
#                vis_img = cv2.resize(vis_img, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
#                vis_img2 = cv2.resize(vis_img2, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
            vis_img = vis_img.transpose(2,0,1)
    
            # ADD blend with original image
            org_img = inputs[0].detach().cpu().numpy()
            org_img = (org_img - org_img.min()) / (org_img.max() - org_img.min() + 1e-5)
            org_img = (org_img*255).astype(np.uint8)
            org_img = org_img[::-1,:,:] # RGB2BGR
 
            # Fusion!
            img_w_heatmap = ((org_img/2) + (vis_img/2))
            #cv2.imwrite(os.path.join(args.pred_density_map_path, str(name[0]) + '.jpg'), cv2.cvtColor(img_w_heatmap.transpose(1,2,0).astype(np.uint8), cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(args.pred_density_map_path, str(name[0]) + '.jpg'), img_w_heatmap.transpose(1,2,0))

            # detect/draw center point
            paint_center(args, img_w_heatmap, vis_img2, taus=[-1], org_img=org_img)
            #paint_center(args, img_w_heatmap, vis_img2, taus=[0.47])
    
    image_errs = np.array(image_errs)
    mse = np.sqrt(np.mean(np.square(image_errs)))
    mae = np.mean(np.abs(image_errs))
    print('{}: mae {}, mse {}\n'.format(model_path, mae, mse))
    print("elapsed : {}".format(time.time()-start))


elif args.test_type == 'val_with_gt': # show gt overlay 
    image_errs = []
    #for inputs, count, name in dataloader:
    for inputs, keypoints, name in dataloader:
        inputs = inputs.to(device)
        assert inputs.size(0) == 1, 'the batch size should equal to 1'
        with torch.set_grad_enabled(False):
            outputs, _ = model(inputs)
        img_err = len(keypoints[0]) - torch.sum(outputs).item()
    
        print(name, img_err, len(keypoints[0]), torch.sum(outputs).item())
        image_errs.append(img_err)
        keypoints = [p.to('cpu').numpy() for p in keypoints[0]]
    
        if args.pred_density_map_path:
            vis_img = outputs[0, 0].cpu().numpy()
            # normalize density map values from 0 to 1, then map it to 0-255.
            vis_img = (vis_img - vis_img.min()) / (vis_img.max() - vis_img.min() + 1e-5)
            vis_img2 = vis_img
            vis_img = (vis_img*255).astype(np.uint8)
            vis_img = cv2.applyColorMap(vis_img, cv2.COLORMAP_VIRIDIS)
            if args.downsample_ratio > 1:
                vis_img = cv2.resize(vis_img, dsize=(args.input_size, args.input_size), interpolation=cv2.INTER_NEAREST) # tmp
                vis_img2 = cv2.resize(vis_img2, dsize=(args.input_size, args.input_size), interpolation=cv2.INTER_NEAREST) # tmp
#                vis_img = cv2.resize(vis_img, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
#                vis_img2 = cv2.resize(vis_img2, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
            vis_img = vis_img.transpose(2,0,1)
    
            # ADD blend with original image
            org_img = inputs[0].detach().cpu().numpy()
            org_img = (org_img - org_img.min()) / (org_img.max() - org_img.min() + 1e-5)
            org_img = (org_img*255).astype(np.uint8)
            org_img = org_img[::-1,:,:] # RGB2BGR

            # Fusion!
            img_w_heatmap = ((org_img/2) + (vis_img/2))
            # paint annotation point
            keypoints = np.array(keypoints)
            if(len(keypoints)>0):
                keypoints[:,[0,1]] = keypoints[:,[1,0]] # x,y -> y,x
                img_w_gt = utils.paint_circles(img=img_w_heatmap,
                                                   points=keypoints,
                                                   color='white')
 
            cv2.imwrite(os.path.join(args.pred_density_map_path, str(name[0]) + '.jpg'), cv2.cvtColor(img_w_gt.transpose(1,2,0).astype(np.uinit8), cv2.COLOR_RGB2BGR))

            # detect/draw center point
            paint_center(args, img_w_gt, vis_img2, taus=[-1], org_img=org_img)
            #paint_center(args, img_w_gt, vis_img2, taus=[0.47])
    
    image_errs = np.array(image_errs)
    mse = np.sqrt(np.mean(np.square(image_errs)))
    mae = np.mean(np.abs(image_errs))
    print('{}: mae {}, mse {}\n'.format(model_path, mae, mse))
    print("elapsed : {}".format(time.time()-start))



elif args.test_type == 'test_no_gt': # without gt
    #image_errs = []
    #for inputs, count, name in dataloader:
    for inputs, name in dataloader:

        inputs = inputs.to(device)
        assert inputs.size(0) == 1, 'the batch size should equal to 1'
        with torch.set_grad_enabled(False):
            outputs, _ = model(inputs)

        print(name,  torch.sum(outputs).item())
    
        if args.pred_density_map_path:
            vis_img = outputs[0, 0].cpu().numpy()
            # normalize density map values from 0 to 1, then map it to 0-255.
            vis_img = (vis_img - vis_img.min()) / (vis_img.max() - vis_img.min() + 1e-5)
            vis_img2 = vis_img
            vis_img = (vis_img*255).astype(np.uint8)
            vis_img = cv2.applyColorMap(vis_img, cv2.COLORMAP_VIRIDIS)
            if args.downsample_ratio > 1:
                vis_img = cv2.resize(vis_img, dsize=(args.input_size, args.input_size), interpolation=cv2.INTER_NEAREST)
                vis_img2 = cv2.resize(vis_img2, dsize=(args.input_size, args.input_size), interpolation=cv2.INTER_NEAREST)
            vis_img = vis_img.transpose(2,0,1)
    
            # ADD blend with original image
            org_img = inputs[0].detach().cpu().numpy()
            org_img = (org_img - org_img.min()) / (org_img.max() - org_img.min() + 1e-5)
            org_img = (org_img*255).astype(np.uint8)
            org_img = org_img[::-1,:,:] # RGB2BGR

            # Fusion!
            img_w_heatmap = ((org_img/2) + (vis_img/2))
            cv2.imwrite(os.path.join(args.pred_density_map_path, str(name[0]) + '.jpg'), cv2.cvtColor(img_w_heatmap.transpose(1,2,0).astype(np.uint8), cv2.COLOR_RGB2BGR))
 
            # detect/draw center point
            paint_center(args, img_w_heatmap, vis_img2, taus=[-1], org_img=org_img)
            #paint_center(args, img_w_heatmap, vis_img2, taus=[0.47])

    print("elapsed : {}".format(time.time()-start))


