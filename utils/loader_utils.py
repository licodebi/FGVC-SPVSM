from torchvision import transforms, datasets
from PIL import Image
from torch.utils.data.dataset import Dataset
import os
import random
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler

import sys
from InSPyReNet import sodmask
def getloader(args):
    if args.dataset=='CUB':
        if args.isSOD:
            if args.split=='non-overlap':
                patch_size=args.img_size//16
                train_transform_mask = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((args.img_size, args.img_size), interpolation=Image.NEAREST),
                    transforms.Resize((patch_size, patch_size), interpolation=Image.NEAREST),
                    transforms.ToTensor(),
                ])
                test_transform_mask = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((args.img_size, args.img_size), interpolation=Image.NEAREST),
                    transforms.Resize((patch_size, patch_size), interpolation=Image.NEAREST),
                    transforms.ToTensor(),
                ])
            else:
                patch_size=(args.img_size - 16) // 12 + 1
                train_transform_mask = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((args.img_size, args.img_size), interpolation=Image.NEAREST),
                    transforms.Resize((patch_size, patch_size), interpolation=Image.NEAREST),
                    transforms.ToTensor(),
                ])
                test_transform_mask = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((args.img_size, args.img_size), interpolation=Image.NEAREST),
                    transforms.Resize((patch_size, patch_size), interpolation=Image.NEAREST),
                    transforms.ToTensor(),
                ])
            train_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((args.img_size, args.img_size), interpolation=Image.BILINEAR),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            test_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((args.img_size, args.img_size), interpolation=Image.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            trainset = CUB(args, transform=train_transform, tranform_mask=train_transform_mask, is_train=True)
            testset = CUB(args, transform=test_transform, tranform_mask=test_transform_mask, is_train=True)
        else:
            train_transform = transforms.Compose([
                transforms.Resize((args.img_size, args.img_size), interpolation=Image.BILINEAR),
                transforms.RandomCrop((args.img_size, args.img_size)),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            test_transform = transforms.Compose([
                transforms.Resize((args.img_size, args.img_size), interpolation=Image.BILINEAR),
                transforms.CenterCrop((args.img_size, args.img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            trainset=CUB(args,transform=train_transform,is_train=True)
            testset=CUB(args,transform=test_transform,is_train=True)

        train_sampler = RandomSampler(trainset)
        test_sampler = SequentialSampler(testset)
        train_loader = DataLoader(trainset,
                                  sampler=train_sampler,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers,
                                  pin_memory=True)
        test_loader = DataLoader(testset,
                                 sampler=test_sampler,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers,
                                 pin_memory=True) if testset is not None else None
        return train_loader,test_loader

class CUB(Dataset):
    def __init__(self,args,transform,tranform_mask=None,is_train=True):
        self.base_folder='images'
        # 数据跟目录
        self.root=args.data_root
        # 输入图片大小
        self.img_size=args.img_size
        # 是否对mask进行填充
        self.padding=args.padding
        # patch形式，overlap或non-overlap
        self.split=args.split
        # 是否使用sod增强
        self.is_SOD=args.isSOD
        # 是否是数据集
        self.is_train=is_train
        # 图片预处理
        self.transform=transform
        self.transform_mask=tranform_mask
        # 得到数据集中的文本
        img_txt_file = open(os.path.join(self.root, 'images.txt'))
        label_txt_file = open(os.path.join(self.root, 'image_class_labels.txt'))
        train_val_file = open(os.path.join(self.root, 'train_test_split.txt'))
        img_name_list = []
        train_test_list = []
        label_list = []
        # 得到所有的图片名
        for line in img_txt_file:
            img_name_list.append(line[:-1].split(' ')[-1])
        # 得到所有标签
        for line in label_txt_file:
            label_list.append(int(line[:-1].split(' ')[-1]) - 1)
        # 得到所有图片是否是训练集的信息
        for line in train_val_file:
            train_test_list.append(int(line[:-1].split(' ')[-1]))

        # 如果是训练阶段
        if self.is_train:
            # zip之后会产生元组(1, 'image1.jpg')
            # 如果是训练集才加入到file_list之中
            self.file_list = [x for i, x in zip(train_test_list, img_name_list) if i]
            # 得到每张图片的路径
            file_path_list = [os.path.join(self.root, self.base_folder, x) for i, x in
                            zip(train_test_list, img_name_list) if i]
            self.label = [x for i, x in zip(train_test_list, label_list) if i]
        else:
            self.file_list = [x for i, x in zip(train_test_list, img_name_list) if not i]
            file_path_list = [os.path.join(self.root, self.base_folder, x) for i, x in
                            zip(train_test_list, img_name_list) if not i]
            self.label = [x for i, x in zip(train_test_list, label_list) if i]

        self.size=len(self.file_list)
        if self.is_SOD:
            mask_out_list, mask_out_bound=sodmask.mask(file_path_list)
            self.mask_list=mask_out_list
            self.mask_bound_list=mask_out_bound
        self.file_path_list=file_path_list
    def __getitem__(self, index):
        lable=self.label[index]
        bound = self.mask_bound_list[index]
        if self.padding:
            bound[0]=bound[0]-8
            bound[2]=bound[2]+8
            bound[1] = bound[1] - 8
            bound[3] = bound[3] + 8
        image = Image.open(self.file_path_list[index]).convert('RGB')
        if self.is_SOD:
            image_mask=self.mask_list[index]
            # 转为张量
            image_tensor=transforms.ToTensor(image)
            image_tensor=image_tensor[:,:,bound[0]:bound[2], bound[1]:bound[3]].squeeze()
            image_mask=image_mask[bound[0]:bound[2], bound[1]:bound[3]]
            if self.is_train:
                if random.random() < 0.5:
                    image=self.transform(image_tensor)
                    image_mask=self.transform_mask(image_mask)
                else:
                    train_transform = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize((self.img_size, self.img_size), interpolation=Image.BILINEAR),
                        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                        transforms.RandomHorizontalFlip(p=1.0),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])
                    if self.split == 'non-overlap':
                        patch_size = self.img_size // 16
                        train_transform_mask = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Resize((self.img_size, self.img_size), interpolation=Image.NEAREST),
                            transforms.Resize((patch_size, patch_size), interpolation=Image.NEAREST),
                            transforms.RandomHorizontalFlip(p=1.0),
                            transforms.ToTensor(),
                        ])
                    else:
                        patch_size = (self.img_size - 16) // 12 + 1
                        train_transform_mask = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Resize((self.img_size, self.img_size), interpolation=Image.NEAREST),
                            transforms.Resize((patch_size, patch_size), interpolation=Image.NEAREST),
                            transforms.RandomHorizontalFlip(p=1.0),
                            transforms.ToTensor(),
                        ])
                    image=train_transform(image_tensor)
                    image_mask=train_transform_mask(image_mask)
            else:
                image = self.transform(image_tensor)
                image_mask = self.transform_mask(image_mask)
            return image,lable,image_mask
        else:
            image=self.transform(image)
            return image,lable
    def __len__(self):
        return self.size