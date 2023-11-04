import os
import cv2
import sys

import numpy as np
import torchvision.transforms as transforms

from torch.utils.data.dataset import Dataset
from PIL import Image
from threading import Thread

filepath = os.path.split(__file__)[0]
repopath = os.path.split(filepath)[0]
sys.path.append(repopath)

from .custom_transforms import *
from InSPyReNet.utils.misc import *
from skimage import io, transform, color

Image.MAX_IMAGE_PIXELS = None

class RescaleT(object):

	def __init__(self,output_size):
		assert isinstance(output_size,(int,tuple))
		self.output_size = output_size

	def __call__(self,sample):
		imidx, image, label = sample['imidx'], sample['image'],sample['label']

		h, w = image.shape[:2]

		if isinstance(self.output_size,int):
			if h > w:
				new_h, new_w = self.output_size*h/w,self.output_size
			else:
				new_h, new_w = self.output_size,self.output_size*w/h
		else:
			new_h, new_w = self.output_size

		new_h, new_w = int(new_h), int(new_w)

		# # #resize the image to new_h x new_w and convert image from range [0,255] to [0,1]
		# # img = transform.resize(image,(new_h,new_w),mode='constant')
		# # lbl = transform.resize(label,(new_h,new_w),mode='constant', order=0, preserve_range=True)

		# img = transform.resize(image,(self.output_size,self.output_size),mode='constant')
		# lbl = transform.resize(label,(self.output_size,self.output_size),mode='constant', order=0, preserve_range=True)
		img = transform.resize(image,(self.output_size,self.output_size),mode='constant', anti_aliasing=True, anti_aliasing_sigma=None)
		lbl = transform.resize(label,(self.output_size,self.output_size),mode='constant', order=0, preserve_range=True, anti_aliasing=True, anti_aliasing_sigma=None)

		return {'imidx':imidx, 'image':img,'label':lbl}
class ToTensorLab(object):
	"""Convert ndarrays in sample to Tensors."""
	def __init__(self,flag=0):
		self.flag = flag

	def __call__(self, sample):

		imidx, image, label =sample['imidx'], sample['image'], sample['label']

		tmpLbl = np.zeros(label.shape)

		if(np.max(label)<1e-6):
			label = label
		else:
			label = label/np.max(label)

		# change the color space
		if self.flag == 2: # with rgb and Lab colors
			tmpImg = np.zeros((image.shape[0],image.shape[1],6))
			tmpImgt = np.zeros((image.shape[0],image.shape[1],3))
			if image.shape[2]==1:
				tmpImgt[:,:,0] = image[:,:,0]
				tmpImgt[:,:,1] = image[:,:,0]
				tmpImgt[:,:,2] = image[:,:,0]
			else:
				tmpImgt = image
			tmpImgtl = color.rgb2lab(tmpImgt)

			# nomalize image to range [0,1]
			tmpImg[:,:,0] = (tmpImgt[:,:,0]-np.min(tmpImgt[:,:,0]))/(np.max(tmpImgt[:,:,0])-np.min(tmpImgt[:,:,0]))
			tmpImg[:,:,1] = (tmpImgt[:,:,1]-np.min(tmpImgt[:,:,1]))/(np.max(tmpImgt[:,:,1])-np.min(tmpImgt[:,:,1]))
			tmpImg[:,:,2] = (tmpImgt[:,:,2]-np.min(tmpImgt[:,:,2]))/(np.max(tmpImgt[:,:,2])-np.min(tmpImgt[:,:,2]))
			tmpImg[:,:,3] = (tmpImgtl[:,:,0]-np.min(tmpImgtl[:,:,0]))/(np.max(tmpImgtl[:,:,0])-np.min(tmpImgtl[:,:,0]))
			tmpImg[:,:,4] = (tmpImgtl[:,:,1]-np.min(tmpImgtl[:,:,1]))/(np.max(tmpImgtl[:,:,1])-np.min(tmpImgtl[:,:,1]))
			tmpImg[:,:,5] = (tmpImgtl[:,:,2]-np.min(tmpImgtl[:,:,2]))/(np.max(tmpImgtl[:,:,2])-np.min(tmpImgtl[:,:,2]))

			# tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

			tmpImg[:,:,0] = (tmpImg[:,:,0]-np.mean(tmpImg[:,:,0]))/np.std(tmpImg[:,:,0])
			tmpImg[:,:,1] = (tmpImg[:,:,1]-np.mean(tmpImg[:,:,1]))/np.std(tmpImg[:,:,1])
			tmpImg[:,:,2] = (tmpImg[:,:,2]-np.mean(tmpImg[:,:,2]))/np.std(tmpImg[:,:,2])
			tmpImg[:,:,3] = (tmpImg[:,:,3]-np.mean(tmpImg[:,:,3]))/np.std(tmpImg[:,:,3])
			tmpImg[:,:,4] = (tmpImg[:,:,4]-np.mean(tmpImg[:,:,4]))/np.std(tmpImg[:,:,4])
			tmpImg[:,:,5] = (tmpImg[:,:,5]-np.mean(tmpImg[:,:,5]))/np.std(tmpImg[:,:,5])

		elif self.flag == 1: #with Lab color
			tmpImg = np.zeros((image.shape[0],image.shape[1],3))

			if image.shape[2]==1:
				tmpImg[:,:,0] = image[:,:,0]
				tmpImg[:,:,1] = image[:,:,0]
				tmpImg[:,:,2] = image[:,:,0]
			else:
				tmpImg = image

			tmpImg = color.rgb2lab(tmpImg)

			# tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

			tmpImg[:,:,0] = (tmpImg[:,:,0]-np.min(tmpImg[:,:,0]))/(np.max(tmpImg[:,:,0])-np.min(tmpImg[:,:,0]))
			tmpImg[:,:,1] = (tmpImg[:,:,1]-np.min(tmpImg[:,:,1]))/(np.max(tmpImg[:,:,1])-np.min(tmpImg[:,:,1]))
			tmpImg[:,:,2] = (tmpImg[:,:,2]-np.min(tmpImg[:,:,2]))/(np.max(tmpImg[:,:,2])-np.min(tmpImg[:,:,2]))

			tmpImg[:,:,0] = (tmpImg[:,:,0]-np.mean(tmpImg[:,:,0]))/np.std(tmpImg[:,:,0])
			tmpImg[:,:,1] = (tmpImg[:,:,1]-np.mean(tmpImg[:,:,1]))/np.std(tmpImg[:,:,1])
			tmpImg[:,:,2] = (tmpImg[:,:,2]-np.mean(tmpImg[:,:,2]))/np.std(tmpImg[:,:,2])

		else: # with rgb color
			tmpImg = np.zeros((image.shape[0],image.shape[1],3))
			image = image/np.max(image)
			if image.shape[2]==1:
				tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
				tmpImg[:,:,1] = (image[:,:,0]-0.485)/0.229
				tmpImg[:,:,2] = (image[:,:,0]-0.485)/0.229
			else:
				tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
				tmpImg[:,:,1] = (image[:,:,1]-0.456)/0.224
				tmpImg[:,:,2] = (image[:,:,2]-0.406)/0.225

		tmpLbl[:,:,0] = label[:,:,0]

		tmpImg = tmpImg.transpose((2, 0, 1))
		tmpLbl = label.transpose((2, 0, 1))

		return {'imidx':torch.from_numpy(imidx), 'image': torch.from_numpy(tmpImg).float(), 'label': torch.from_numpy(tmpLbl)}
# 自定义数据集类
class SalObjDataset(Dataset):
	# 图片路径列表，标签列表，数据预处理函数
	def __init__(self,img_name_list,lbl_name_list,transform=None):
		# self.root_dir = root_dir
		# self.image_name_list = glob.glob(image_dir+'*.png')
		# self.label_name_list = glob.glob(label_dir+'*.png')
		self.image_name_list = img_name_list
		self.label_name_list = lbl_name_list
		self.transform = transform
	# 图片数
	def __len__(self):
		return len(self.image_name_list)
	# 用于获取数据集中指定索引位置的样本
	def __getitem__(self,idx):

		# image = Image.open(self.image_name_list[idx])#io.imread(self.image_name_list[idx])
		# label = Image.open(self.label_name_list[idx])#io.imread(self.label_name_list[idx])
		# 根据idx读取对应的图片数据
		image = io.imread(self.image_name_list[idx])
		# 得到图片路径
		imname = self.image_name_list[idx]
		imidx = np.array([idx])
		# 如果未提供标签列表
		if(0==len(self.label_name_list)):
			#创造一个和图像相同大小的全零数组作为虚拟标签 label_3
			label_3 = np.zeros(image.shape)
		else:
			label_3 = io.imread(self.label_name_list[idx])

		label = np.zeros(label_3.shape[0:2])
		# 如果是三维数组
		if(3==len(label_3.shape)):
			#取其第一个通道作为标签
			label = label_3[:,:,0]
		elif(2==len(label_3.shape)):
			label = label_3
		# 如果图像是三维的而标签是二维的，就在标签上添加一个额外的通道。
		# 如果图像和标签都是二维的，就在它们各自上添加一个额外的通道
		if(3==len(image.shape) and 2==len(label.shape)):
			label = label[:,:,np.newaxis]
		elif(2==len(image.shape) and 2==len(label.shape)):
			image = image[:,:,np.newaxis]
			label = label[:,:,np.newaxis]
		# 创建一个字典 sample，包含三个条目：'imidx'、'image' 和 'label'，分别对应索引、图像和标签
		sample = {'imidx':imidx, 'image':image, 'label':label}

		if self.transform:
			sample = self.transform(sample)

		return sample
def get_transform(tfs):
    comp = []
    for key, value in zip(tfs.keys(), tfs.values()):
        if value is not None:
            tf = eval(key)(**value)
        else:
            tf = eval(key)()
        comp.append(tf)
    return transforms.Compose(comp)
class MaskLoader(Dataset):
    def __init__(self,file_path_list,tfs):
        self.transform = get_transform(tfs)
        self.file_path_list=file_path_list
        self.size = len(self.file_path_list)
    def __getitem__(self, index):
        image = Image.open(self.file_path_list[index]).convert('RGB')
        # gt = Image.open(self.gts[index]).convert('L')
        shape = image.size[::-1]
        name = self.file_path_list[index].split('/')[-1]
        name = os.path.splitext(name)[0]
        # print(name)
        sample = {'image': image, 'name': name,'shape':shape}

        sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.size
class CUB(Dataset):
    def __init__(self,root,tfs,is_train=True):
        self.base_folder="images"
        self.transform = get_transform(tfs)
        self.is_train = is_train
        self.root=root
        img_txt_file=open(os.path.join(self.root,'images.txt'))
        label_txt_file = open(os.path.join(self.root, 'image_class_labels.txt'))
        train_val_file = open(os.path.join(self.root, 'train_test_split.txt'))
        img_name_list=[]
        train_test_list=[]
        shape_hw_list = []
        # 得到所有的图片名
        for line in img_txt_file:
            img_name_list.append(line[:-1].split(' ')[-1])
        # 得到所有图片是否是训练集的信息
        for line in train_val_file:
            train_test_list.append(int(line[:-1].split(' ')[-1]))
        if self.is_train:
            # zip之后会产生元组(1, 'image1.jpg')
            # 如果是训练集才加入到file_list之中
            self.file_list = [x for i, x in zip(train_test_list, img_name_list) if i]
            # 得到每张图片的路径
            self.file_list_full = [os.path.join(self.root, self.base_folder, x) for i, x in
                              zip(train_test_list, img_name_list) if i]
        else:
            self.file_list = [x for i, x in zip(train_test_list, img_name_list) if not i]
            self.file_list_full = [os.path.join(self.root, self.base_folder, x) for i, x in
                              zip(train_test_list, img_name_list) if not i]
        for img_path in self.file_list_full:
            h,w=Image.open(img_path).size[::-1]
            shape_hw_temp = [h, w]  # h_max (y), w_max (x)
            shape_hw_list.append(shape_hw_temp)
        self.size = len(self.file_list)

    def __getitem__(self, index):
        image = Image.open(self.file_list_full[index]).convert('RGB')
        # gt = Image.open(self.gts[index]).convert('L')
        shape = image.size[::-1]
        name = self.file_list_full[index].split('/')[-1]
        name = os.path.splitext(name)[0]
        # print(name)
        sample = {'image': image, 'name': name,'shape':shape}

        sample = self.transform(sample)
        return sample
    def __len__(self):
        return self.size
class RGB_Dataset(Dataset):
    def __init__(self, root, sets, tfs):
        self.images, self.gts = [], []
        
        for set in sets:
            image_root, gt_root = os.path.join(root, set, 'images'), os.path.join(root, set, 'masks')

            images = [os.path.join(image_root, f) for f in os.listdir(image_root) if f.lower().endswith(('.jpg', '.png'))]
            images = sort(images)
            
            gts = [os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.lower().endswith(('.jpg', '.png'))]
            gts = sort(gts)
            
            self.images.extend(images)
            self.gts.extend(gts)
        
        self.filter_files()
        
        self.size = len(self.images)
        self.transform = get_transform(tfs)
        
    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        gt = Image.open(self.gts[index]).convert('L')
        shape = gt.size[::-1]
        name = self.images[index].split(os.sep)[-1]
        name = os.path.splitext(name)[0]
            
        sample = {'image': image, 'gt': gt, 'name': name, 'shape': shape}

        sample = self.transform(sample)
        return sample

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images, gts = [], []
        for img_path, gt_path in zip(self.images, self.gts):
            img, gt = Image.open(img_path), Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images, self.gts = images, gts

    def __len__(self):
        return self.size
