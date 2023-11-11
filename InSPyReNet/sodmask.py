import os
import cv2
import sys
import torch
import torchvision
import argparse

import numpy as np
import torchvision.transforms as transform
from torchvision.transforms import InterpolationMode
from PIL import Image
from torch.utils.data.dataloader import DataLoader

filepath = os.path.split(os.path.abspath(__file__))[0]
repopath = os.path.split(filepath)[0]
sys.path.append(repopath)
import copy
from .lib import *
from .utils.misc import *
from .data.dataloader import *
# from .data.custom_transforms import *
from .lib.InSPyReNet import InSPyReNet_SwinB
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c',     type=str,            default='InSPyReNet/configs/InSPyReNet_SwinB.yaml')
    parser.add_argument('--source', '-s',     type=str, default='data/')
    parser.add_argument('--dest', '-d',       type=str,            default=None)
    parser.add_argument('--type', '-t',       type=str,            default='map')
    parser.add_argument('--jit', '-j',        action='store_true', default=False)
    parser.add_argument('--verbose', '-v',    action='store_true', default=False)
    return parser.parse_args()

def get_format(source):
    img_count = len([i for i in source if i.lower().endswith(('.jpg', '.png', '.jpeg'))])
    vid_count = len([i for i in source if i.lower().endswith(('.mp4', '.avi', '.mov' ))])
    
    if img_count * vid_count != 0:
        return ''
    elif img_count != 0:
        return 'Image'
    elif vid_count != 0:
        return 'Video'
    else:
        return ''

def mask(datapath):
    mask_out_list=[]
    mask_out_bound=[]
    config="InSPyReNet/configs/InSPyReNet_SwinB.yaml"
    opt = load_config(config)
    model = InSPyReNet_SwinB(depth=64,pretrained=True,base_size=[384, 384])
    model.load_state_dict(torch.load(os.path.join(
        'InSPyReNet/snapshots/InSPyReNet_SwinB/', 'latest.pth'), map_location=torch.device('cpu')), strict=True)
    
    # if args.gpu is True:
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    # save_dir='InSPyReNet/result/cub/'
    # if save_dir is not None:
    #     os.makedirs(save_dir, exist_ok=True)
    dataset=eval('MaskLoader')(datapath,opt.Test.Dataset.transforms)
    samples  = DataLoader(dataset=dataset, batch_size=1, num_workers=2, pin_memory=True,shuffle=False)
    for i,sample in enumerate(samples):
        # print(i)
        if torch.cuda.is_available():
            # sample=samples.cuda()
            sample = to_cuda(sample)
        with torch.no_grad():
            # if args.jit is True:
            #     out = model(sample['image'])
            # else:
            out = model(sample)
            pred = out['pred']
            pred[pred<0.2]=0.
            pred[pred>=0.2]=1.
            pred=pred.squeeze()
            transform_mask = transform.Compose([
                transform.ToPILImage(),  # (mode='1'),
                transform.Resize((sample['shape'][0], sample['shape'][1]), Image.NEAREST),
                transform.ToTensor()
            ])
            pred=transform_mask(pred).squeeze()
            row_indices, col_indices = np.where(pred == 1.)
            min_row, min_col = min(row_indices), min(col_indices)
            max_row, max_col = max(row_indices), max(col_indices)
            bounding_box = (min_row, min_col, max_row, max_col)
            mask_out_list.append(pred)
            mask_out_bound.append(bounding_box)
    return mask_out_list,mask_out_bound

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi)/(ma-mi)
    return dn
def new_mask(img_path,shape_hw):
    # mask_out_list = []
    # mask_out_bound = []
    config = "InSPyReNet/configs/InSPyReNet_SwinB.yaml"
    opt = load_config(config)
    model = InSPyReNet_SwinB(depth=64, pretrained=True, base_size=[384, 384])
    model.load_state_dict(torch.load(os.path.join(
        'InSPyReNet/snapshots/InSPyReNet_SwinB/', 'latest.pth'), map_location=torch.device('cpu')), strict=True)

    # if args.gpu is True:
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    img_name_list = img_path
    shape_hw_list = shape_hw
    mask_out_np_list = []
    start_x_list = []
    start_y_list = []
    h_list = []
    w_list = []
    bad_mask_count = 0
    refined_mask_count = 0
    dataset = SalObjDataset(img_name_list = img_name_list,
                                        lbl_name_list = [],
                                        transform=transforms.Compose([RescaleT(320),
                                                                     ToTensorLab(flag=0)])
                                        )
    samples = DataLoader(dataset=dataset, batch_size=1, num_workers=2, shuffle=False)
    for i, sample in enumerate(samples):
        shape_hw_i = shape_hw_list[i]
        # print(i)
        if torch.cuda.is_available():
            # sample=samples.cuda()
            sample = to_cuda(sample)
        with torch.no_grad():
            out = model(sample)
            # torch.Size([1, 1, 384, 384])
        pred = out['pred']
        pred=pred[:,0,:,:]
        pred = normPRED(pred)
        # 设置遮罩的阈值
        THRESHOLD = 0.8  # 0.5 # 0.8 # 0.5 #0.8 # for the original mask (better not smaller than 0.7 cuz artifacts)
        THRESHOLD_resize = 0.2  # 0.1 # 0.2 # for the resized mask
        THRESHOLD_deep = 0.1  # for the non-detected mask
        pred = pred[0, :, :]
        pred_cpu = pred.cpu()
        # 得到输出的图片
        out_img = pred_cpu.detach().numpy()
        out_img_refine = copy.deepcopy(out_img)
        # 若该像素的值大于阈值则值为1，否则为0
        out_img[out_img > THRESHOLD] = 1
        out_img[out_img <= THRESHOLD] = 0
        # 将大于阈值的设为像素设为白色，其余的设为黑色
        out_img = (out_img * 255).astype(np.uint8)
        out_img = Image.fromarray(out_img, mode='L')
        # BOUNDING BOX CREATION
        transform_mask = transforms.Compose([
            # 将图像转为张量
            transforms.ToTensor(),
            # 张量转为PIL图片，并转为单通道灰度图像
            transforms.ToPILImage(mode='L'),  # (mode='1'),
            # transforms.Resize((image.shape[1],image.shape[0]), Image.BILINEAR),
            # 将图像调整为对应大小
            transforms.Resize((shape_hw_i[0], shape_hw_i[1]), InterpolationMode.BILINEAR),
            # shape_hw (0 - height, 1 - width)
            # 最后再转成张量
            transforms.ToTensor(),
        ])
        # 将模型得到的图片进行处理
        out_img = transform_mask(out_img)
        # 得到图片高宽
        # [500, 415]
        out_img = out_img[0, :, :]
        mask_out = out_img
        mask_out = mask_out.cpu()
        # 对像素进行判断，大于阈值的设为0小于的设为1
        mask_out = torch.where(mask_out > THRESHOLD_resize, torch.tensor(0.),
                               torch.tensor(1.))
        # 转为np
        mask_out_np = mask_out.detach().numpy()

        out_layer = out_img
        out_layer = out_layer.cpu()

        out_layer = torch.where(out_layer > THRESHOLD_resize, torch.tensor(1.),
                                torch.tensor(0.))
        out_layer = out_layer.detach().numpy()
        # 分别找到每行和每列中值为1的区域的起始和结束坐标。这些起始和结束坐标用于描述或定位二维数组中包含1的区域的位置和大小

        # 查找x值的起始点，遍历out_layer每一行找到第一个值为1的位置，若未找到则将其设置为宽度加1
        x_starts = [np.where(out_layer[i] == 1)[0][0] if len(np.where(out_layer[i] == 1)[0]) != 0 \
                        else out_layer.shape[0] + 1 for i in range(out_layer.shape[0])]
        # 查找x值的终点，遍历out_layer每一行找到最后一个值为1的位置，若未找到则将其设置为0
        x_ends = [np.where(out_layer[i] == 1)[0][-1] if len(np.where(out_layer[i] == 1)[0]) != 0 \
                      else 0 for i in range(out_layer.shape[0])]
        #
        y_starts = [np.where(out_layer.T[i] == 1)[0][0] if len(np.where(out_layer.T[i] == 1)[0]) != 0 \
                        else out_layer.T.shape[0] + 1 for i in range(out_layer.T.shape[0])]
        y_ends = [np.where(out_layer.T[i] == 1)[0][-1] if len(np.where(out_layer.T[i] == 1)[0]) != 0 \
                      else 0 for i in range(out_layer.T.shape[0])]
        # 分别找到最小的x_starts，最大的x_ends，最小的y_starts，最大的y_ends
        # 将startx,starty存入start，endx,endy存入end
        startx = min(x_starts)
        endx = max(x_ends)
        starty = min(y_starts)
        endy = max(y_ends)
        start = (startx, starty)
        end = (endx, endy)

        ## For cases when U2N couldn't detect mask:
        # [DONE] 1.1 if (end - start) < 30-50px -> decrease the THRESHOLD
        # [DONE] 1.2 if (start>end) or end==0 ? -> decrease the THRESHOLD
        # [DONE] 2.1 if no mask found anyway -> create center crop mask (x, y) +-10 %
        # [DONE] 2.2 + restore h,w from (0,0) to (x, y) +-10 %
        # 宽度
        w_temp = end[0] - start[0]
        # 高度
        h_temp = end[1] - start[1]
        # 计算数值大于0.9的元素的像素数量
        mask_px = np.count_nonzero(out_layer > 0.9)  # (expected to be == 1.0)
        # 如果符合一下条件的则该图片掩码有问题
        if (end[0] <= start[0]) or (end[1] <= start[1]) or (mask_px < 5000) or (w_temp < 50) or (h_temp < 50):

            # 再次对该图片进行掩码操作,符合条件的设为白色,其余为黑色
            out_img_refine[out_img_refine > THRESHOLD_deep] = 1
            out_img_refine[out_img_refine <= THRESHOLD_deep] = 0

            out_img_refine = (out_img_refine * 255).astype(np.uint8)
            out_img_refine = Image.fromarray(out_img_refine, mode='L')

            transform_mask = transforms.Compose([
                transforms.ToTensor(),
                transforms.ToPILImage(mode='L'),  # (mode='1'),
                # transforms.Resize((image.shape[1],image.shape[0]), Image.BILINEAR),
                transforms.Resize((shape_hw_i[0], shape_hw_i[1]), InterpolationMode.BILINEAR),
                # shape_hw (0 - height, 1 - width)
                transforms.ToTensor(),
            ])

            out_img_refine = transform_mask(out_img_refine)

            out_img_refine = out_img_refine[0, :, :]

            out_layer_refine = out_img_refine
            out_layer_refine = out_layer_refine.cpu()
            out_layer_refine = torch.where(out_img_refine > THRESHOLD_resize, torch.tensor(1.),
                                           torch.tensor(0.))
            out_layer_refine = out_layer_refine.detach().numpy()

            x_starts = [np.where(out_layer_refine[i] == 1)[0][0] if len(np.where(out_layer_refine[i] == 1)[0]) != 0 \
                            else out_layer_refine.shape[0] + 1 for i in range(out_layer_refine.shape[0])]
            x_ends = [np.where(out_layer_refine[i] == 1)[0][-1] if len(np.where(out_layer_refine[i] == 1)[0]) != 0 \
                          else 0 for i in range(out_layer_refine.shape[0])]
            y_starts = [np.where(out_layer_refine.T[i] == 1)[0][0] if len(np.where(out_layer_refine.T[i] == 1)[0]) != 0 \
                            else out_layer_refine.T.shape[0] + 1 for i in range(out_layer_refine.T.shape[0])]
            y_ends = [np.where(out_layer_refine.T[i] == 1)[0][-1] if len(np.where(out_layer_refine.T[i] == 1)[0]) != 0 \
                          else 0 for i in range(out_layer_refine.T.shape[0])]

            startx = min(x_starts)
            endx = max(x_ends)
            starty = min(y_starts)
            endy = max(y_ends)
            start = (startx, starty)
            end = (endx, endy)



            w_temp = end[0] - start[0]
            h_temp = end[1] - start[1]

            mask_px = np.count_nonzero(out_layer_refine > 0.9)  # (expected to be == 1.0)

            # 若改进之后仍未被掩码成功,则强制进行掩码操作,记录掩码失败数量+1
            if (end[0] <= start[0]) or (end[1] <= start[1]) or (mask_px < 5000) or (w_temp < 50) or (h_temp < 50):



                startx = shape_hw_i[1] * 0.1
                endx = shape_hw_i[1] * 0.9  # w -> x

                starty = shape_hw_i[0] * 0.1
                endy = shape_hw_i[0] * 0.9  # h -> y

                start = (startx, starty)
                end = (endx, endy)

                mask_out_np = np.ones((int(shape_hw_i[0]), int(shape_hw_i[1])))
                mask_out_np[int(starty):int(endy), int(startx):int(endx)] = 0
                bad_mask_count += 1

            else:
                mask_out_refine = out_img_refine
                mask_out_refine = mask_out_refine.cpu()
                mask_out_refine = torch.where(mask_out_refine > THRESHOLD_resize, torch.tensor(0.),
                                              torch.tensor(1.))
                mask_out_np = mask_out_refine.detach().numpy()

                refined_mask_count += 1

        w = end[0] - start[0]
        h = end[1] - start[1]

        # save results to test_results folder
        # if not os.path.exists(prediction_dir):
        #     os.makedirs(prediction_dir, exist_ok=True)
        # save_output(img_name_list[i_test], mask_out, prediction_dir)

        del out
        mask_out_np_list.append(mask_out_np)
        start_x_list.append(start[0])
        start_y_list.append(start[1])
        h_list.append(h)
        w_list.append(w)

        # if i_test % 1000 == 0:
        #     print(i_test)

        # print("Refined masks total:", refined_mask_count)
        # print("Bad masks total:", bad_mask_count)
        # 返回结果
    return mask_out_np_list, start_x_list, start_y_list, h_list, w_list

