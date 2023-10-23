import yaml
import os
import shutil
import argparse
# 读取yaml文件，并将其中的配置设置为args.参数名=参数
def load_yaml(args, yml):
    with open(yml, 'r', encoding='utf-8') as fyml:
        dic = yaml.load(fyml.read(), Loader=yaml.Loader)
        for k in dic:
            # setattr(x, 'y', v) is equivalent to ``x.y = v''
            setattr(args, k, dic[k])
# 建立记录文件夹
def build_record_folder(args):
    # 如果不存在records文件夹，则创建
    if not os.path.isdir("./records/"):
        os.mkdir("./records/")
    # 设置存储信息的目录地址为./records/项目名称/实验名/
    args.save_dir = "./records/" + args.project_name + "/" + args.exp_name + "/"
    # 根据args.save_dir创建目录，并如果存在则不报异常
    os.makedirs(args.save_dir, exist_ok=True)
    # 在args.save_dir目录下创建一个backup目录
    os.makedirs(args.save_dir + "backup/", exist_ok=True)
    # 将args.c复制到args.save_dir下的config.yaml文件中
    shutil.copy(args.c, args.save_dir+"config.yaml")

def get_args(with_deepspeed: bool=False):

    parser = argparse.ArgumentParser("Fine-Grained Visual Classification")
    # 添加一个--c参数默认为空
    parser.add_argument("--c", default="", type=str, help="config file path")
    # 获取配置参数
    args = parser.parse_args()

    load_yaml(args, args.c)
    build_record_folder(args)

    return args

