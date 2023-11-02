import os
import cv2
import sys
import tqdm
import torch
import argparse

import numpy as np
import torchvision.transforms as transform
from PIL import Image
from torch.utils.data.dataloader import DataLoader

# filepath = os.path.split(os.path.abspath(__file__))[0]
# repopath = os.path.split(filepath)[0]
# sys.path.append(repopath)

from .lib import *
from .utils.misc import *
from .data.dataloader import *
from .data.custom_transforms import *
from .lib.InSPyReNet import InSPyReNet_SwinB
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c',     type=str,            default='InSPyReNet/configs/InSPyReNet_SwinB.yaml')
    parser.add_argument('--source', '-s',     type=str, default='data/')
    parser.add_argument('--dest', '-d',       type=str,            default=None)
    parser.add_argument('--type', '-t',       type=str,            default='map')
    parser.add_argument('--gpu', '-g',        action='store_true', default=True)
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

def cub():
    args = _args()
    opt = load_config(args.config)
    model = InSPyReNet_SwinB(depth=64,pretrained=True,base_size=[384, 384])
    model.load_state_dict(torch.load(os.path.join(
        'InSPyReNet/snapshots/InSPyReNet_SwinB/', 'latest.pth'), map_location=torch.device('cpu')), strict=True)
    
    # if args.gpu is True:
    model = model.cuda()
    model.eval()
    save_dir='InSPyReNet/result/cub/'
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    dataset='cub'
    if dataset=='cub':
        path=os.path.join(args.source,'CUB200-2011')

        cub_dataset=eval('CUB')(path,opt.Test.Dataset.transforms,True)

        cub_loader  = DataLoader(dataset=cub_dataset, batch_size=1, num_workers=opt.Test.Dataloader.num_workers, pin_memory=opt.Test.Dataloader.pin_memory)
        for i,sample in enumerate(cub_loader):
            # print(i)
            print(sample['image'].shape)
            sample = to_cuda(sample)
            with torch.no_grad():
                if args.jit is True:
                    out = model(sample['image'])
                else:
                    out = model(sample)
                if i==0:
                    # pred = to_numpy(out['pred'], sample['shape'])
                    pred = out['pred']
                    pred[pred<0.2]=0.
                    pred[pred>=0.2]=1.
                    pred=pred[0,0,:,:]
                    transform_mask = transform.Compose([
                        transform.ToPILImage(),  # (mode='1'),
                        # non-overlapped:
                        transform.Resize((448//16, 448//16), interpolation=Image.NEAREST),
                        # Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST
                        transform.ToTensor()
                    ])
                    pred=transform_mask(pred)
                    pred=torch.flatten(pred)
                    print(pred.shape)
                    # pred =pred.numpy()
                    # img = (np.stack([pred] * 3, axis=-1) * 255).astype(np.uint8)
                    # Image.fromarray((pred * 255).astype(np.uint8)).save(os.path.join(args.savepath, sample['name'][0] + '.jpg'))
                    break
                # pred = to_numpy(out['pred'], sample['shape'])
                # pred[pred < 0.2] = 0.
                # # img = (np.stack([pred] * 3, axis=-1) * 255).astype(np.uint8)
                # Image.fromarray((pred * 255).astype(np.uint8)).save(
                #     os.path.join(args.savepath, sample['name'][0] + '.jpg'))

if __name__ == "__main__":
    args = _args()
    opt = load_config(args.config)
    # cub(opt, args)
