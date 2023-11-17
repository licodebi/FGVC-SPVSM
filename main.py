import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pim_module.con_loss_new import con_loss_new
import contextlib
import wandb
import warnings
import math
from models.builder import MODEL_GETTER
from data.dataset import build_loader
from utils.costom_logger import timeLogger
from utils.config_utils import load_yaml, build_record_folder, get_args
from utils.lr_schedule import cosine_decay, adjust_lr, get_lr
from eval import evaluate, cal_train_metrics, suppression
import sys
import random
from utils.data_utils import get_loader
from utils.loader_utils import getloader

sys.setrecursionlimit(10**5)
warnings.simplefilter("ignore")

# 设置随机种子
# def set_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
def eval_freq_schedule(args, epoch: int):
    if epoch >= args.max_epochs * 0.95:
        args.eval_freq = 1
    elif epoch >= args.max_epochs * 0.9:
        args.eval_freq = 1
    elif epoch >= args.max_epochs * 0.8:
        args.eval_freq = 2

def set_environment(args, tlogger):
    
    print("Setting Environment...")
    # 判断是否可以使用GPU计算，如果可以则设置gpu不行则使用cpu
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        tlogger.print("USE GPU....")
    ### = = = =  Dataset and Data Loader = = = =
    tlogger.print("Building Dataloader....")
    # 读取训练集以及验证集
    if not args.isU2net:
        train_loader, val_loader = build_loader(args)
    # elif args.isSOD:
    #     train_loader, val_loader=getloader(args)
    else:
        train_loader, val_loader = get_loader(args)
    # 如果训练集为空或验证集为空
    if train_loader is None and val_loader is None:
        raise ValueError("Find nothing to train or evaluate.")
    # 如果不为空则输出训练样本数以及batch数
    if train_loader is not None:
        print("    Train Samples: {} (batch: {})".format(len(train_loader.dataset), len(train_loader)))
    else:
        # raise ValueError("Build train loader fail, please provide legal path.")
        print("    Train Samples: 0 ~~~~~> [Only Evaluation]")
    if val_loader is not None:
        print("    Validation Samples: {} (batch: {})".format(len(val_loader.dataset), len(val_loader)))
    else:
        print("    Validation Samples: 0 ~~~~~> [Only Training]")
    tlogger.print()

    ### = = = =  Model = = = =  
    tlogger.print("Building Model....")
    # 创建模型
    # model = MODEL_GETTER[args.model_name](
    #     use_fpn = args.use_fpn,
    #     img_size=args.data_size,
    #     fpn_size = args.fpn_size,
    #     use_selection = args.use_selection,
    #     num_classes = args.num_classes,
    #     num_selects = args.num_selects,
    #     use_combiner = args.use_combiner,
    # ) # about return_nodes, we use our default setting
    # set_seed(args.seed)
    model = MODEL_GETTER[args.model_name](
        num_classes=args.num_classes,
        img_size=args.img_size,
        update_warm=args.warmup_batchs,
        num_selects=args.num_selects,
        patch_num=args.patch_num,
        split=args.split,
        coeff_max=args.coeff_max
    )
    # 如果预训练不为空
    if args.pretrained is not None:
        # 从预训练文件中加载模型
        checkpoint = torch.load(args.pretrained, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']
    else:
        model.load_from(np.load("./models/ViT-B_16.npz"))
        start_epoch = 0

    # model = torch.nn.DataParallel(model, device_ids=None) # device_ids : None --> use all gpus.
    # 模型使用GPU
    model.to(args.device)
    tlogger.print()
    
    """
    if you have multi-gpu device, you can use torch.nn.DataParallel in single-machine multi-GPU 
    situation and use torch.nn.parallel.DistributedDataParallel to use multi-process parallelism.
    more detail: https://pytorch.org/tutorials/beginner/dist_overview.html
    """
    #如果训练数据为空
    if train_loader is None:
        return train_loader, val_loader, model, None, None, None, None
    
    ### = = = =  Optimizer = = = =
    # 加载优化器
    tlogger.print("Building Optimizer....")
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.max_lr, nesterov=True, momentum=0.9, weight_decay=args.wdecay)
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.max_lr)
    # 如果是预训练模型可以加载预训练中的优化器
    if args.pretrained is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])

    tlogger.print()
    #根据余弦衰减学习率方法得到所有batch的学习率序列
    schedule = cosine_decay(args, len(train_loader))
    # 是否使用混合精度训练
    if args.use_amp:
        # 创建一个GradScaler对象用于自动调节梯度的缩放
        scaler = torch.cuda.amp.GradScaler()
        # 使用上下文管理器自动执行混合精度转换
        amp_context = torch.cuda.amp.autocast
    else:
        scaler = None
        amp_context = contextlib.nullcontext

    return train_loader, val_loader, model, optimizer, schedule, scaler, amp_context, start_epoch


# 训练函数
# args:配置参数
# epoch:迭代数
# model:模型
# scaler:用于自动调节梯度的缩放
# amp_context:上下文管理器自动执行混合精度转换
# optimizer:优化器
# schedule:学习率序列
# train_loader:训练集
def train(args, epoch, model, scaler, amp_context, optimizer, schedule, train_loader):
    
    optimizer.zero_grad()
    #获得批次数，等于训练集图片数/batch_size
    # 5994/8=750
    total_batchs = len(train_loader) # just for log
    # 用于显示百分比
    show_progress = [x/10 for x in range(11)] # just for log

    progress_i = 0

    # temperature = 2 ** (epoch // 10 - 1)
    #温度衰减函数，0.5 ** (epoch // 10)为衰减系数，每十个epoch系数减半
    # 论文原文的温度衰减
    temperature=0.5 ** (epoch // (-(math.log(0.0625/args.temperature,2))))
    # temperature = 0.5 ** (epoch // 10) * args.temperature
    # temperature = args.temperature
    # 总批次数%update_freq（更新频率）=更新的次数
    #n_left_batchs=750%4=2
    n_left_batchs = len(train_loader) % args.update_freq
    # 迭代训练集
    # batch_id批次号，ids图片索引，datas图片，lables图片对应的标签索引
    # for batch_id, (ids, datas, labels) in enumerate(train_loader):
    # 如果使用mask处理
    for batch_id, batch in enumerate(train_loader):
        if args.isSOD:
            datas,labels,masks=batch
        else:
            idx,datas,labels=batch
        model.train()
        """ = = = = adjust learning rate = = = = """
        # 获得总迭代数，等于epoch*批次数*当前批次索引
        # 如每个 epoch 中使有100批次进行训练
        # 当完成第2个 epoch 的第30个批次时等于2*100+29
        iterations = epoch * len(train_loader) + batch_id
        # 输入总迭代数，优化器以及lr衰减序列
        # 对当前迭代的学习率进行调整
        adjust_lr(iterations, optimizer, schedule)
        # schedule.step(iterations)
        # temperature = (args.temperature - 1) * (get_lr(optimizer) / args.max_lr) + 1

        batch_size = labels.size(0)

        """ = = = = forward and calculate loss = = = = """
        # 将图片以及对应的标签存入gpu
        datas, labels = datas.to(args.device), labels.to(args.device)
        # 将模型、优化器和数据加载器包装在 amp_context() 中
        # PyTorch 将自动为我们执行梯度缩放和类型转换的操作
        with amp_context():
            """
            [Model Return]
                FPN + Selector + Combiner --> return 'layer1', 'layer2', 'layer3', 'layer4', ...(depend on your setting)
                    'preds_0', 'preds_1', 'comb_outs'
                FPN + Selector --> return 'layer1', 'layer2', 'layer3', 'layer4', ...(depend on your setting)
                    'preds_0', 'preds_1'
                FPN --> return 'layer1', 'layer2', 'layer3', 'layer4' (depend on your setting)
                ~ --> return 'ori_out'
            
            [Retuen Tensor]
                'preds_0': logit has not been selected by Selector.
                'preds_1': logit has been selected by Selector.
                'comb_outs': The prediction of combiner.
            """
            # 数据传入模型进行向前传播
            # outs=[B,200]
            if args.isSOD:
                outs = model(datas,False,masks)
            else:
                outs = model(datas)

            loss_pi = 0.
            loss_si = 0.
            loss_cl=0.
            loss= 0.
            for name in outs:
                if "struct_outs" in name:
                    loss_so=nn.CrossEntropyLoss()(outs[name],labels)
                    loss_si+=loss_so
                # 如果时上采样的结果
                # if "layer" in name:
                #     if args.lambda_b0 != 0:
                #         loss_b0 = nn.CrossEntropyLoss()(outs[name].mean(1), labels)
                #         loss += args.lambda_b0 * loss_b0
                #     else:
                #         loss_b0 = 0.0
                elif "last_token" in name:
                    loss_co = con_loss_new(outs[name], labels)
                    loss_cl+=loss_co
                elif "assist_outs" in name:
                    loss_ao=nn.CrossEntropyLoss()(outs[name],labels)
                    loss_pi+=args.lambda_a*loss_ao
                elif "comb_outs" in name:
                    loss_co=nn.CrossEntropyLoss()(outs[name],labels)
                    # loss_pi+=(1-args.lambda_a)*loss_co
                    loss_pi+=4.0*loss_co

                # 如果使用了选择器
                # elif "select_" in name:
                #     if not args.use_selection:
                #         raise ValueError("Selector not use here.")
                #     if args.lambda_s != 0:
                #         # 得到每个选择器的输出通道
                #         # outs[name]为([B, 32, 200])
                #         # S=32
                #         S = outs[name].size(1)
                #         #logit大小变为[B*32,200]
                #         logit = outs[name].view(-1, args.num_classes).contiguous()
                #         # labels.unsqueeze(1).repeat(1, S).flatten(0).shape
                #         # lables变为（B*32）
                #         loss_s = nn.CrossEntropyLoss()(logit,
                #                                        labels.unsqueeze(1).repeat(1, S).flatten(0))
                #         loss += args.lambda_s * loss_s
                #     else:
                #         loss_s = 0.0
                # # BS模块中被丢弃的映射
                # # 如果为被丢弃的映射
                # elif "drop_" in name:
                #     if not args.use_selection:
                #         raise ValueError("Selector not use here.")
                #
                #     if args.lambda_n != 0:
                #         # 得到被丢弃数
                #         S = outs[name].size(1)
                #         # 112为被丢弃的样本数
                #         # logit大小变为torch.Size([B*112, 200])
                #         logit = outs[name].view(-1, args.num_classes).contiguous()
                #         n_preds = nn.Tanh()(logit)
                #         # 创建一个大小与logit一样的全为-1的张量
                #         labels_0 = torch.zeros([batch_size * S, args.num_classes]) - 1
                #         labels_0 = labels_0.to(args.device)
                #         # 计算两者的均方误差
                #         # 让被抑制的背景更加接近-1
                #         loss_n = nn.MSELoss()(n_preds, labels_0)
                #         # 论文中的loss_d
                #         loss += args.lambda_n * loss_n
                #     else:
                #         loss_n = 0.0
                # 合并分类预测损失
                # elif "comb_outs" in name:
                #     if not args.use_combiner:
                #         raise ValueError("Combiner not use here.")
                #
                #     if args.lambda_c != 0:
                #         # 将合并后的分类进行损失计算
                #         # comb_outs=[B,200]
                #         loss_c = nn.CrossEntropyLoss()(outs[name], labels)
                #         # 论文中的loss_m
                #         loss += args.lambda_c * loss_c
                elif "ori_out" in name:
                    loss_ori = F.cross_entropy(outs[name], labels)
                    loss += loss_ori
            loss = args.lambda_b * loss_si + (1 - args.lambda_b) * loss_pi+loss_cl
            # 当前批次不是最后 n_left_batchs 个未处理的批次
            if batch_id < len(train_loader) - n_left_batchs:
                loss /= args.update_freq
            else:
                loss /= n_left_batchs
        
        """ = = = = calculate gradient = = = = """
        # 计算梯度
        if args.use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        """ = = = = update model = = = = """
        # 更新模型,如果当前的批次号+1可以整除update_freq或者是最后一个批次则进行梯度更新
        # 批次号总数为=图片数/batch_size
        if (batch_id + 1) % args.update_freq == 0 or (batch_id + 1) == len(train_loader):
            if args.use_amp:
                scaler.step(optimizer)
                scaler.update() # next batch
            else:
                optimizer.step()
            optimizer.zero_grad()

        """ log (MISC) """
        if args.use_wandb and ((batch_id + 1) % args.log_freq == 0):
            model.eval()
            msg = {}
            # 得到当前迭代数以及对应的学习率
            msg['info/epoch'] = epoch + 1
            msg['info/lr'] = get_lr(optimizer)
            # 计算训练精确度,并存入wandb的日志当中
            cal_train_metrics(args, msg, outs, labels, batch_size)
            # cal_train_metrics(args, msg, outs, labels, batch_size, model.selector.thresholds)
            wandb.log(msg)
        # 现实进度，每次现实10%
        train_progress = (batch_id + 1) / total_batchs
        # print(train_progress, show_progress[progress_i])
        if train_progress > show_progress[progress_i]:
            print(".."+str(int(show_progress[progress_i] * 100)) + "%", end='', flush=True)
            progress_i += 1


def main(args, tlogger):
    """
    save model last.pt and best.pt
    """
    # 设置环境
    # 得到训练集,验证集,模型,优化器,学习率序列,上下文管理器,开始的epoch
    train_loader, val_loader, model, optimizer, schedule, scaler, amp_context, start_epoch = set_environment(args, tlogger)
    # 精确度
    best_acc = 0.0
    # 最好的测试名
    best_eval_name = "null"
    # 如果使用wandb
    if args.use_wandb:
        wandb.init(entity=args.wandb_entity,
                   project=args.project_name,
                   name=args.exp_name,
                   config=args)
        wandb.run.summary["best_acc"] = best_acc
        wandb.run.summary["best_eval_name"] = best_eval_name
        wandb.run.summary["best_epoch"] = 0
    #  开始迭代
    for epoch in range(start_epoch, args.max_epochs):

        """
        Train
        """
        # 如果训练集不为空
        if train_loader is not None:
            # 开始训练
            tlogger.print("Start Training {} Epoch".format(epoch+1))
            train(args, epoch, model, scaler, amp_context, optimizer, schedule, train_loader)
            tlogger.print()
        else:
            from eval import eval_and_save
            eval_and_save(args, model, val_loader)
            break

        eval_freq_schedule(args, epoch)

        model_to_save = model.module if hasattr(model, "module") else model
        checkpoint = {"model": model_to_save.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                      "best_acc": best_acc}
        torch.save(checkpoint, args.save_dir + "backup/last.pt")
        # 如果当前的epoch为十的倍数则
        if epoch == 0 or (epoch + 1) % args.eval_freq == 0:
            """
            Evaluation
            """
            acc = -1
            if val_loader is not None:
                tlogger.print("Start Evaluating {} Epoch".format(epoch + 1))
                acc, eval_name, accs = evaluate(args, model, val_loader)
                tlogger.print("....BEST_ACC: {}% ({}%)".format(max(acc, best_acc), acc))
                tlogger.print()

            if args.use_wandb:
                wandb.log(accs)

            if acc > best_acc:
                best_acc = acc
                best_eval_name = eval_name
                checkpoint = {"model": model_to_save.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                              "best_acc": best_acc}
                torch.save(checkpoint, args.save_dir + "backup/best.pt")
            if args.use_wandb:
                wandb.run.summary["best_acc"] = best_acc
                wandb.run.summary["best_eval_name"] = best_eval_name
                wandb.run.summary["best_epoch"] = epoch + 1


if __name__ == "__main__":
    # 实例化日志类
    tlogger = timeLogger()
    tlogger.print("Reading Config...")
    # 获取保存实验配置参数的yaml文件地址
    args = get_args()
    # 判断地址是否为空
    assert args.c != "", "Please provide config file (.yaml)"
    load_yaml(args, args.c)
    # 建立存储结果的文件
    build_record_folder(args)
    tlogger.print()
    # 传入参数以及日志对象
    main(args, tlogger)