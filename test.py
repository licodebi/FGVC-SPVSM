from utils.data_utils import get_loader
import argparse
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

    parser.add_argument("--img_size", default=400, type=int,
                        help="After-crop image resolution")
    parser.add_argument("--resize_size", default=448, type=int,
                        help="Pre-crop image resolution")
    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Total batch size for eval.")

    parser.add_argument("--num_workers", default=2, type=int,
                        help="Number of workers for dataset preparation.")

    parser.add_argument('--sm_vit', action='store_true',
                        help="Whether to use SM-ViT")
    parser.add_argument('--low_memory', action='store_true',
                        help="Allows to use less memory (RAM) during input image feeding. False: Slower - Do image pre-processing for the whole dataset at the beginning and store the results in memory. True: Faster - Do pre-processing on-the-go.")

    parser.add_argument('--data_root', type=str, default='./data/CUB200-2011/', # Originall
                        help="Path to the dataset\n")
                        # '/l/users/20020067/Datasets/CUB_200_2011/CUB_200_2011/CUB_200_2011') # CUB
                        # '/l/users/20020067/Datasets/Stanford Dogs/Stanford_Dogs') # dogs
                        # '/l/users/20020067/Datasets/NABirds/NABirds') # NABirds

    args = parser.parse_args()
    train_loader,test_loader=get_loader(args)
    for i,batch in enumerate(train_loader):
        img,label,mask=batch
        print("图片形状",img.shape)
        print("标签",label.shape)
        print("掩盖图片形状",mask.shape)

    # len(train_loader)
    # len(test_loader)
if __name__ == "__main__":
    main()