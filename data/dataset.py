import os
import numpy as np
import cv2
import torchvision.transforms as transforms
from PIL import Image
import copy
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from .randaug import RandAugment


def build_loader(args):
    train_set, train_loader = None, None
    # 如果训练集存储的位置不为空
    if args.train_root is not None:
        # 传入练集存储的位置，训练集的图片大小，
        train_set = ImageDataset(istrain=True, root=args.train_root, data_size=args.data_size, return_index=True)
        # 读取数据集
        train_loader = DataLoader(train_set, num_workers=args.num_workers, shuffle=True, batch_size=args.batch_size)

    val_set, val_loader = None, None
    if args.val_root is not None:
        val_set = ImageDataset(istrain=False, root=args.val_root, data_size=args.data_size, return_index=True)
        # 读取数据集
        val_loader = DataLoader(val_set, num_workers=1, shuffle=True, batch_size=args.batch_size)

    return train_loader, val_loader

def get_dataset(args):
    if args.train_root is not None:
        train_set = ImageDataset(istrain=True, root=args.train_root, data_size=args.data_size, return_index=True)
        return train_set
    return None

# 加载图像数据集的类，用于获取图像样本和标签
class ImageDataset(Dataset):

    def __init__(self, 
                 istrain: bool,
                 root: str,
                 data_size: int,
                 return_index: bool = False):
        # notice that:
        # sub_data_size mean sub-image's width and height.
        """ basic information """
        self.root = root
        self.data_size = data_size
        self.return_index = return_index

        """ declare data augmentation """
        # 图像标准化
        normalize = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )

        # 448:600
        # 384:510
        # 768:
        # 如果是训练集
        if istrain:
            # transforms.RandomApply([RandAugment(n=2, m=3, img_size=data_size)], p=0.1)
            # RandAugment(n=2, m=3, img_size=sub_data_size)
            self.transforms = transforms.Compose([
                        # 将图像调整为固定(510, 510)的尺寸，使用双线性插值法进行图像的缩放
                        transforms.Resize((510, 510), Image.BILINEAR),
                        # 随机裁剪图像为指定的尺寸(data_size, data_size)
                        transforms.RandomCrop((data_size, data_size)),
                        # 随机对图像进行水平翻转，增加训练数据的多样性
                        transforms.RandomHorizontalFlip(),
                        # 按一定概率（p=0.1）应用给定的数据增强操作
                        #随机应用高斯模糊，使用给定的核大小和标准差的范围进行模糊操作
                        # sigma即标准差，标准差越大模糊效果越强
                        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5))], p=0.1),
                        #随机调整图像的锐化度，使用给定的锐化系数范围进行调整
                        # sharpness_factor：需要调整多少锐度。可以是任何非负数 0-模糊，1-原图，2-锐度提高两倍
                        transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.1),
                        # 图像转为张量并将像素进行归一化
                        transforms.ToTensor(),
                        normalize
                ])
        else:
            self.transforms = transforms.Compose([
                        #将图像调整为固定(510, 510)的尺寸
                        transforms.Resize((510, 510), Image.BILINEAR),
                        # 将图像中心裁剪为指定的尺寸(data_size, data_size)，确保图像中的主要内容被保留
                        transforms.CenterCrop((data_size, data_size)),
                        # 将图像转换为张量形式，并将像素值归一化到[0, 1]的范围
                        transforms.ToTensor(),
                        # 之前定义的图像标准化操作，将图像的每个通道数值减去均值并除以标准差
                        normalize
                ])

        """ read all data information """
        # 获取所有图片的路径以及对应的类别索引
        self.data_infos = self.getDataInfo(root)

    # 传入图片所在文件夹路径
    def getDataInfo(self, root):
        # 创建一个空数组
        data_infos = []
        #获取所有类别的图片的文件夹名，并形成一个数组
        folders = os.listdir(root)
        folders.sort() # sort by alphabet
        # 根据文件数判断训练集的类别大小
        print("[dataset] class number:", len(folders))
        # 遍历folders数组
        # class_id为索引，folder为文件夹名
        for class_id, folder in enumerate(folders):
            # 根据训练集路径和目录名读取目录下的所有图片名
            files = os.listdir(root+folder)
            for file in files:
                # 得到文件的具体路径
                data_path = root+folder+"/"+file
                # 将图片路径以及类别索引作为元组存入data_infos数组
                data_infos.append({"path":data_path, "label":class_id})
        return data_infos
    # 返回图片的数量
    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, index):
        # get data information.
        # 根据索引获得图片的路径以及对应的类别索引
        image_path = self.data_infos[index]["path"]
        label = self.data_infos[index]["label"]
        # read image by opencv.
        #根据文件路径读取图片
        img = cv2.imread(image_path)
        #将颜色通道从BGR转为RGB,::-1代表倒序
        img = img[:, :, ::-1] # BGR to RGB.
        
        # to PIL.Image
        # 将 NumPy 数组表示的图像转换为 PIL 图像对象
        img = Image.fromarray(img)
        img = self.transforms(img)
        # 如果return_index为true返回对应索引，图片，图片对应的标签索引
        if self.return_index:
            # return index, img, sub_imgs, label, sub_boundarys
            return index, img, label
        
        # return img, sub_imgs, label, sub_boundarys
        return img, label
