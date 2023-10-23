import math
import numpy as np
from timm.scheduler.cosine_lr import CosineLRScheduler
# 实现余弦衰减学习率
def cosine_decay(args, batchs: int, decay_type: int = 1):
    # 总batchs数=迭代数*batch数
    total_batchs = args.max_epochs * batchs
    #计算剩余batch数
    # warmup_batchs为热身batch数
    iters = np.arange(total_batchs - args.warmup_batchs)
    # 如果衰减类型为1
    if decay_type == 1:
        schedule = np.array([1e-12 + 0.5 * (args.max_lr - 1e-12) * (1 + \
                             math.cos(math.pi * t / total_batchs)) for t in iters])
    elif decay_type == 2:
        schedule = args.max_lr * np.array([math.cos(7*math.pi*t / (16*total_batchs)) for t in iters])
    else:
        raise ValueError("Not support this deccay type")
    #增加一个长度为warmup_batchs的等间距序列,从1e-9到最大学习率之间的均匀分布
    # 这是学习率衰减过程的一个热身阶段
    if args.warmup_batchs > 0:
        warmup_lr_schedule = np.linspace(1e-9, args.max_lr, args.warmup_batchs)
        # 将warmup_lr_schedule与之间schedule的进行连接
        schedule = np.concatenate((warmup_lr_schedule, schedule))
    # lr_min=args.max_lr* 0
    # warmup_lr = args.max_lr * 1e-3
    # schedule=CosineLRScheduler(
    #     optimizer=optimizer,
    #     t_initial=total_batchs,
    #     lr_min=lr_min,
    #     warmup_lr_init=warmup_lr,
    #     warmup_t=args.warmup_batchs,
    #     warmup_prefix=True,
    #     cycle_limit=1,
    #     t_in_epochs=False,
    # )
    return schedule

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        if param_group["lr"] is not None:
            return param_group["lr"]

def adjust_lr(iteration, optimizer, schedule):
    # 获取当前的参数组中的参数
    for param_group in optimizer.param_groups:
        # 将参数中的学习率设置为之前设置的当前迭代的学习率
        param_group["lr"] = schedule[iteration]
