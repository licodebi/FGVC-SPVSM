from utils.data_utils import get_loader
import argparse
import numpy as np
from utils.loader_utils import getloader
from PIL import Image
from torchvision import transforms, datasets

def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    # parser.add_argument("--name", required=True,
    #                     default="output",
    #                     help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", choices=["CUB", "dogs", "nabirds"], default="CUB",
                        help="Which downstream task.")
    # parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
    #                                              "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
    #                     default="ViT-B_16",
    #                     help="Which ViT variant to use.")

    parser.add_argument("--img_size", default=448, type=int,
                        help="After-crop image resolution")
    parser.add_argument("--padding", default=True)
    parser.add_argument("--split", type=str,default="non-overlap")
    parser.add_argument("--batch_size", default=8, type=int,
                        help="Total batch size for training.")
    # parser.add_argument("--batch_size", default=8, type=int,
    #                     help="Total batch size for eval.")

    parser.add_argument("--num_workers", default=2, type=int,
                        help="Number of workers for dataset preparation.")

    parser.add_argument('--isU2net', default=True,help="Whether to use SM-ViT")
    parser.add_argument('--data_root', type=str, default='data/CUB200-2011/', # Originall
                        help="Path to the dataset\n")
                        # '/l/users/20020067/Datasets/CUB_200_2011/CUB_200_2011/CUB_200_2011') # CUB
                        # '/l/users/20020067/Datasets/Stanford Dogs/Stanford_Dogs') # dogs
                        # '/l/users/20020067/Datasets/NABirds/NABirds') # NABirds
    # image=Image.open("data/Black_Footed_Albatross_0046_18.jpg")
    # bbox = (60.0,27.0,325.0,304.0)
    # cropped_image = image.crop(bbox)

    # transform=transforms.ToTensor()
    # image=transform(image)
    # image=image[:,60:,27:305]
    # # image = image.numpy().squeeze()
    # transform=transforms.ToPILImage()
    # image=transform(image)
    # cropped_image.save("./data/test.jpg")
    args = parser.parse_args()
    # train_loader,val_loader=getloader(args)
    # print(len(train_loader))

    # out_layer = np.array([[0, 1, 0, 0, 1],
    #                       [1, 1, 0, 0, 0],
    #                       [0, 0, 0, 1, 0],
    #                       [1, 0, 0, 0, 1]])
    # print(out_layer.T)
    # y_starts = [np.where(out_layer.T[i] == 1)[0][0] if len(np.where(out_layer.T[i] == 1)[0]) != 0 \
    #                 else out_layer.T.shape[0] + 1 for i in range(out_layer.T.shape[0])]
    # print(y_starts)
    # y_ends = [np.where(out_layer.T[i] == 1)[0][-1] if len(np.where(out_layer.T[i] == 1)[0]) != 0 \
    #               else 0 for i in range(out_layer.T.shape[0])]
    # print(y_ends)
    # starty = min(y_starts)
    # endy = max(y_ends)
    # h_temp = endy - starty
    # print(h_temp)
    train_loader,test_loader=get_loader(args)
    # for i,batch in enumerate(train_loader):
    #     if i==0:
    #         img,label,mask=batch
    #         print("图片形状",img.shape)
    #         print("标签",label.shape)
    #         # print(mask)
    #         print("掩盖图片形状",mask.shape)

    # len(train_loader)
    # len(test_loader)
    # CUB.cub()
if __name__ == "__main__":
    main()