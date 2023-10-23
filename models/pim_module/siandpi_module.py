import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from typing import Union
import numpy as np
import math

import copy
# 使用Vit_Det
class Vit_FPN(nn.Module):
    def __init__(self, inputs: dict, fpn_size: int):
        super(Vit_FPN, self).__init__()
        scale_factors=[4.0,2.0,1.0]
        # 得到最后一层的输出
        inp_names = [name for name in inputs]
        inp_name=inp_names[-1]
        # 排除cls_token
        block_output=inputs[inp_name][:,1:,:]

        dim=block_output.size(-1)
        # 从最后一层开始下采样
        for idx,scale in enumerate(scale_factors):
            if scale==4.0:
                # S放大4倍，通道数缩小4倍
                layers=[
                    nn.ConvTranspose1d(dim,dim//2,kernel_size=2,stride=2),
                    nn.BatchNorm1d(dim//2),
                    nn.GELU(),
                    nn.ConvTranspose1d(dim // 2, dim // 4, kernel_size=2, stride=2),
                ]
                out_dim = dim // 4
                m = nn.Sequential(
                    nn.Linear(out_dim, out_dim, 1),
                    nn.ReLU(),
                    nn.Linear(out_dim, fpn_size, 1)
                )
            elif scale == 2.0:
                # S放大2倍，通道数缩小2倍
                layers = [
                    nn.ConvTranspose1d(dim, dim // 2, kernel_size=2, stride=2),
                    nn.BatchNorm1d(dim // 2),
                    nn.GELU(),
                    nn.ConvTranspose1d(dim // 2, dim // 2, kernel_size=1),
                ]
                out_dim = dim // 2
                m = nn.Sequential(
                    nn.Linear(out_dim, out_dim, 1),
                    nn.ReLU(),
                    nn.Linear(out_dim, fpn_size, 1)
                )
            elif scale == 1.0:
                # S和C不变
                layers = []
                out_dim = dim
                m = nn.Sequential(
                    nn.Linear(out_dim, out_dim, 1),
                    nn.ReLU(),
                    nn.Linear(out_dim, fpn_size, 1)
                )
            else:layers=[]
            layers = nn.Sequential(*layers)
            self.add_module(f"Up_layer{idx+1}", layers)
            self.add_module(f"Proj_layer{idx+1}", m)
    def forward(self, x):
        # 得到最后一层特征图，去掉cls类
        inp_names = [name for name in x]
        inp_name = inp_names[-1]
        input = x[inp_name][:, 1:, :]
        # B,C,S
        input=input.transpose(1, 2).contiguous()
        # 根据最后一层进行下采样，得到三层
        hs = ['layer1', 'layer2', 'layer3']
        outputs = {}
        for i in range(len(hs)):
            name=hs[i]
            # B,S，fpn_size
            outputs[f"layer{i+1}"]=getattr(self, "Proj_"+name)(getattr(self, "Up_"+name)(input).transpose(1, 2).contiguous())
        return outputs
# 使用blcok选择器
class Selector(nn.Module):

    def __init__(self, inputs: dict, num_classes: int, num_select: dict, fpn_size: Union[int, None] = None):
        super(Selector, self).__init__()
        # 分类器数，以及对应的分类器的输出维度
        self.num_select = num_select
        self.fpn_size = fpn_size
        ### build classifier
        # 构建分类器，如果不适用fpn_size统一维度，直接对每层的特征进行分类
        if self.fpn_size is None:
            # 图片类别
            self.num_classes = num_classes
            for name in inputs:
                # 每层的输出大小
                fs_size = inputs[name].size()
                # 如果是三维
                if len(fs_size) == 3:
                    in_size = fs_size[2]
                # 如果是四维
                elif len(fs_size) == 4:
                    in_size = fs_size[1]
                # 定义该层的分类器
                m = nn.Linear(in_size, num_classes)
                # 添加该模块
                # print("当前的in_size为:",in_size)
                self.add_module("classifier_l_"+name, m)
        # 定义阈值
        self.thresholds = {}
        # 初始化每层的阈值
        for name in inputs:
            self.thresholds[name] = []
    def forward(self, x, logits=None):
        """
        x :
            dictionary contain the features maps which
            come from your choosen layers.
            size must be [B, HxW, C] ([B, S, C]) or [B, C, H, W].
            [B,C,H,W] will be transpose to [B, HxW, C] automatically.
        """
        selections = {}
        # 遍历x,仅对下采样的样本通过选择器,不选择上采样的样本
        for name in x:
            # print("[selector]", name, x[name].size())
            # 如果名字中为FPN1_即跳到下一循环,判断是不是上采样样本
            # 进行修改,目前仅对上采样样本进行选择
            if "FPN1_" in name:
                continue
            # 得到通道数
            C = x[name].size(-1)
            #从通道维度进行softmax得到每一个类别概率
            # torch.Size([1, 144, 200])  torch.Size([1, 576, 200])
            probs = torch.softmax(logits[name], dim=-1)

            # print("probs的形状:",probs.shape)
            # logits[name].mean(1)即对每一层的分类的预测值在H*W维度进行平均，得到(B,class类)
            # 相当于对每一列进行平均得到一个值，最终得到200列的平均值
            # 之后在对其class维度进行softmax得到0-1范围内的平均预测概率分布
            # 得到每个图片样本的平均预测概率分布(b,class)
            sum_probs = torch.softmax(logits[name].mean(1), dim=-1)
            # 设置选择器name的初始值
            selections[name] = []
            preds_1 = []
            preds_0 = []
            # 根据名称获取到当前层选择器的输出维度
            num_select = self.num_select[name]
            # 遍历logits[name]的每一批次
            for bi in range(logits[name].size(0)):
                # 取到当前批次的最大值索引,如第二个特征最大则取值为1
                _, max_ids = torch.max(sum_probs[bi], dim=-1)
                # 从当前批次的所有样本按第max_ids的特征进行排序,返回排序后的张量confs,以及对应的未排序前的索引位置
                # 索引是当前批次中样本的索引
                # 排序是从大到下
                # rank: torch.Size([2304]) torch.Size([576])
                # confs:torch.Size([2304]) torch.Size([576])，confs的形状为(H*W)
                # confs为对应的每一个样本中对应max_ids的值，这些值为经过排序后的
                confs, ranks = torch.sort(probs[bi, :, max_ids], descending=True)

                # num_select:256
                # 按num_select截取排名前num_select的第name层的第bi个批次的样本
                # sf torch.Size([256, 1536]) torch.Size([128, 1536])
                # sf即代表这个选择器选择的特征
                sf = x[name][bi][ranks[:num_select]]
                # nf torch.Size([2048, 1536]) torch.Size([448, 1536])
                nf = x[name][bi][ranks[num_select:]]  # calculate
                # 将选择器选择后的样本存入selections[name]
                selections[name].append(sf) # [num_selected, C]

                # preds_1的大小: torch.Size([256, 200])
                # 把对应的选择器选择的分类概率保存进preds_1
                preds_1.append(logits[name][bi][ranks[:num_select]])
                # 未被选中的则保存进preds_0中
                # preds_0的大小:torch.Size([2048, 200])
                preds_0.append(logits[name][bi][ranks[num_select:]])

                # 如果batch数>len(self.thresholds[name])
                # 更新选择器阈值,将选择器选择的最小值进行保存
                # 如果batch数>len(self.thresholds[name]),代表当前batch的阈值还未添加则进行增加
                # 反之则对当前batch的阈值进行更新操作
                if bi >= len(self.thresholds[name]):
                    # 取得排序后对应的第num_select个值作为阈值
                    self.thresholds[name].append(confs[num_select]) # for initialize
                else:
                    self.thresholds[name][bi] = confs[num_select]
            # 将多个批次的张量进行堆叠,得到该name的总张量
            # torch.Size([2, 32, 1536])  (B,num_select,C)
            selections[name] = torch.stack(selections[name])
            # torch.Size([2, 32, 200])
            preds_1 = torch.stack(preds_1)
            # torch.Size([2, 112, 200])
            preds_0 = torch.stack(preds_0)
            logits["select_"+name] = preds_1
            logits["drop_"+name] = preds_0
        # 返回选择器
        return selections
class Part_Attention(nn.Module):
    def __init__(self):
        super(Part_Attention, self).__init__()
    def forward(self,x):
        length = len(x)
        last_map = x[0]
        for i in range(1, length):
            last_map = torch.matmul(x[i], last_map)
        last_map = last_map[:, :, 0, 1:]
        B, C = last_map.size(0), last_map.size(1)
        patch_num = last_map.size(-1)
        H = patch_num ** 0.5
        H = int(H)
        attention_map = last_map.view(B, C, H, H)
        # last_map(batch_size, num_attention_heads, S-1)
        # 注意力特征图(batch_size,num_attention_heads,H,H)
        return last_map, attention_map
# 得到相对距离，相对反正切值，锚点，以及位置权重
class RelativeCoordPredictor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # 得到图片的 N(Batch_Size),C,H,W
        N, C, H, W = x.shape
        # 计算掩码mask
        # 得到形状为(N,H*W)的mask,表示每个像素上的通道数量
        mask = torch.sum(x, dim=1)
        size = H
        mask = mask.view(N, H * W)
        # 计算掩码阈值,计算所有像素的平均通道数,得到thresholds的形状为(N,1)
        thresholds = torch.mean(mask, dim=1, keepdim=True)
        # 将掩码阈值与掩码进行比较,生成一个二进制掩码二维张量binary_mask
        # 其中元素值为0或1，表示是否超过阈值,形状为(N,H*W)
        binary_mask = (mask > thresholds).float()
        # 将binary_mask调整为(N, H, W)的形状
        binary_mask = binary_mask.view(N, H, W)
        # 将输入张量x与二进制掩码张量binary_mask相乘，
        # 得到一个掩码后的张量masked_x，形状为(N, C, H*W)
        masked_x = x * binary_mask.view(N, 1, H, W)
        # 将masked_x进行维度转置操作，形状变为(N, H*W, C) H*W=S
        masked_x = masked_x.view(N, C, H * W).transpose(1, 2).contiguous()  # (N, S, C)
        # torch.mean(masked_x, dim=-1) 将C维平均得到 (N, H*W)
        # 再对其进行取最大值,得到具有最大值像素的索引即reduced_x_max_index
        # reduced_x_max_index形状为(N)
        _, reduced_x_max_index = torch.max(torch.mean(masked_x, dim=-1), dim=-1)
        # 构建一个基础索引张量basic_index，其元素值为0到N-1的整数,并将其移动到GPU上
        basic_index = torch.from_numpy(np.array([i for i in range(N)])).cuda().long()
        # 生成一个形状为(size,size,2)的张量 size=H,(H,H,2)
        basic_label = torch.from_numpy(self.build_basic_label(size)).float()
        # Build Label
        label = basic_label.cuda()
        # 通过扩展操作得到与masked_x形状相匹配的张量形状(N, H, W, 2)
        # 将其转换为张量形状(N, H*W, 2)
        label = label.unsqueeze(0).expand((N, H, W, 2)).view(N, H * W, 2)  # (N, S, 2)
        basic_anchor = label[basic_index, reduced_x_max_index, :].unsqueeze(1)  # (N, 1, 2)
        # 计算相对坐标，label - basic_anchor并再除以size
        # 即当前样本的每个像素相对于锚点像素的相对坐标，并再除以size(N,S,2)
        relative_coord = label - basic_anchor
        relative_coord = relative_coord / size
        # 计算相对距离，通过对相对坐标的每个分量进行平方和再开方得到，(N, S)
        relative_dist = torch.sqrt(torch.sum(relative_coord ** 2, dim=-1))  # (N, S)
        # 计算相对角度，通过调用torch.atan2函数计算相对坐标的反正切值，得到角度值范围在(-pi, pi)，(N, S)
        relative_angle = torch.atan2(relative_coord[:, :, 1], relative_coord[:, :, 0])  # (N, S) in (-pi, pi)
        # 将相对角度的值转换到0到1的范围内，通过将角度值除以np.pi，加1，再除以2
        relative_angle = (relative_angle / np.pi + 1) / 2  # (N, S) in (0, 1)
        # 调整掩码的形状与相对距离和相对角度匹配(N, H*W)
        binary_relative_mask = binary_mask.view(N, H * W).cuda()
        # 将相对距离和相对角度乘以掩码，将非掩码的位置置零
        relative_dist = relative_dist * binary_relative_mask
        relative_angle = relative_angle * binary_relative_mask
        # 调整基本锚点的形状(N, 2)
        basic_anchor = basic_anchor.squeeze(1)  # (N, 2)
        relative_coord_total = torch.cat((relative_dist.unsqueeze(2), relative_angle.unsqueeze(2)), dim=-1)
        position_weight = torch.mean(masked_x, dim=-1)
        position_weight = position_weight.unsqueeze(2)
        position_weight = torch.matmul(position_weight, position_weight.transpose(1, 2))
        # 返回
        # 相对坐标总和(N,S,2)
        # 锚点(N,2)
        # 位置权重 (N, H*W,H*W) (N, S,S)
        return relative_coord_total, basic_anchor,position_weight
    # 得到每个patch的坐标信息
    def build_basic_label(self, size):
        basic_label = np.array([[(i, j) for j in range(size)] for i in range(size)])
        return basic_label
# 图卷积层模块
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=False, dropout = 0.1):
        super(GraphConvolution, self).__init__()
        # 输入特征
        self.in_features = in_features
        # 输出特征
        self.out_features = out_features
        # 权重矩阵
        self.weight =  nn.Parameter(torch.zeros(in_features, out_features))
        # 是否设置偏置向量
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        # 定义激活函数，即带有负斜率的泄漏线性整流单元
        self.relu = nn.LeakyReLU(0.2)
        # 丢弃率
        self.dropout = nn.Dropout(p=dropout)
        # 重置权重和偏置的初始化
        self.reset_parameters()

    def reset_parameters(self):
        #得到输出特征的标准差的倒数
        stdv = 1. / math.sqrt(self.weight.size(1))
        # 将权重 weight 从均匀分布中采样，并将其值限制在 (-stdv, stdv) 的范围内
        self.weight.data.uniform_(-stdv, stdv)
        # 如果存在偏置向量 bias，同样从均匀分布中采样
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    # input(N,S,2)  adj (N,S,S)
    def forward(self, input, adj):
        # weight为(N,2,512)
        weight = self.weight.float()
        # 将输入 input 与权重矩阵 weight 相乘，得到支撑信息 support
        # support为(N,S,512)
        support = torch.matmul(input, weight)
        # 将邻接矩阵 adj 与支撑信息 support 相乘，得到输出特征 output
        # output为(N,S,512)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return self.dropout(self.relu(output + self.bias))
        else:
            return self.dropout(self.relu(output))

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
class ClassMlp(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(ClassMlp, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.1)
        self.act_fn = nn.ReLU()
        # 初始化权重和偏置
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        hs = []
        names = []
        for name in x:
            # 暂时不使用从下到上的fpn
            if "FPN1_" in name:
                continue
            #  如果未使用fpn网络
            if self.fpn_size is None:
                # 使用x[name]进行投影
                _tmp = getattr(self, "proj_" + name)(x[name])
            else:
                # 否则直接获得x[name]
                _tmp = x[name]
            # x[name]添加入hs中
            hs.append(_tmp)
            # names添加name以及对应的大小
            names.append([name, _tmp.size()])
        hs = torch.cat(hs, dim=1).transpose(1, 2).contiguous()
        hs = torch.flatten(hs, start_dim=1)
        hs = self.fc1(hs)
        hs = self.act_fn(hs)
        hs = self.dropout(hs)
        hs = self.fc2(hs)
        hs = self.act_fn(hs)
        hs = self.dropout(hs)
        hs = self.fc3(hs)
        return hs
class Mlp(nn.Module):
    def __init__(self, fpn_size,num_class,mlp_dim):
        super(Mlp, self).__init__()
        #设置第一个全连接层，维度从config.hidden_size变换为config.transformer["mlp_dim"]
        self.fc1 = nn.Linear(fpn_size, mlp_dim)
        # 设置第二个全连接层，维度从config.transformer["mlp_dim"]变换回config.hidden_size
        self.fc2 = nn.Linear(mlp_dim, mlp_dim)
        self.fc3 = nn.Linear(mlp_dim, num_class)

        # 使用了GELU激活函数
        self.act_fn = nn.GELU()
        # 按照指定的config.transformer["dropout_rate"]概率将输入的部分元素置为零
        self.dropout = nn.Dropout(0.1)
        # 对全连接层的参数进行初始化
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)
        nn.init.normal_(self.fc3.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x=self.fc3(x)
        return x
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        #
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout)
        x = self.gc2(x, adj)
        return x
# GCN结合器
class GCNCombiner(nn.Module):
    def __init__(self,
                 total_num_selects: int,
                 num_classes: int,
                 inputs: Union[dict, None] = None,
                 proj_size: Union[int, None] = None,
                 fpn_size: Union[int, None] = None):
        super(GCNCombiner, self).__init__()
        # inputs不为空且fpn_size不为空
        # 确保输入参数inputs和fpn_size中至少有一个不为None
        assert inputs is not None or fpn_size is not None, \
            "To build GCN combiner, you must give one features dimension."
        # 当前fpn_size
        self.fpn_size = fpn_size
        # fpn_size为空即不使用fpn
        if fpn_size is None:
            # 由于不使用fpn则将每层的特征进行投影统一为proj_size大小
            for name in inputs:
                # 遍历每层输出,并在每层构建proj
                if len(name) == 4:
                    in_size = inputs[name].size(1)
                elif len(name) == 3:
                    in_size = inputs[name].size(2)
                else:
                    raise RuntimeError("The size of output dimension of previous must be 3 or 4.")
                # 进行投影,将输入通道调整为投影大小
                m = nn.Sequential(
                    nn.Linear(in_size, proj_size),
                    nn.ReLU(),
                    nn.Linear(proj_size, proj_size)
                )
                self.add_module("proj_" + name, m)
            self.proj_size = proj_size
        else:
            # proj_size=fpn_size
            self.proj_size = fpn_size

        # 总的选择器通道数除于64
        num_joints = total_num_selects // 64
        # 使用nn.Linear时，会判断total_num_selects与x的哪个维度相等，则在其维度进行变换
        self.param_pool0 = nn.Linear(total_num_selects, num_joints)
        # 创建一个单位矩阵A,大小为num_jointsxnum_joints,并将矩阵的值除以100后加上1/100
        A = torch.eye(num_joints) / 100 + 1 / 100
        # 并将A进行拷贝设置为可训练参数adj1
        self.adj1 = nn.Parameter(copy.deepcopy(A))
        #
        self.conv1 = nn.Conv1d(self.proj_size, self.proj_size, 1)
        # 正则化
        self.batch_norm1 = nn.BatchNorm1d(self.proj_size)
        # 用于信息融合
        self.conv_q1 = nn.Conv1d(self.proj_size, self.proj_size // 4, 1)
        self.conv_k1 = nn.Conv1d(self.proj_size, self.proj_size // 4, 1)
        # 控制信息融合中的权重
        self.alpha1 = nn.Parameter(torch.zeros(1))

        ### merge information
        # 将num_joints维度的信息汇总为1维
        self.param_pool1 = nn.Linear(num_joints, 1)

        #### class predict
        self.dropout = nn.Dropout(p=0.1)
        # 分类器进行分类
        self.classifier = nn.Linear(self.proj_size, num_classes)
        # 激活函数
        self.tanh = nn.Tanh()

    # 输入的x为选择器选择的样本
    def forward(self, x):
        """
        """
        hs = []
        names = []
        #
        for name in x:
            # 暂时不使用从下到上的fpn
            if "FPN1_" in name:
                continue
            #  如果未使用fpn网络
            if self.fpn_size is None:
                # 使用x[name]进行投影
                _tmp = getattr(self, "proj_" + name)(x[name])
            else:
                # 否则直接获得x[name]
                _tmp = x[name]
            # x[name]添加入hs中
            hs.append(_tmp)
            # names添加name以及对应的大小
            names.append([name, _tmp.size()])
        # 将B, S', C --> B, C, S
        # torch.cat(hs, dim=1)将各个层的选择器输出的张量(B,a,1536),(B,b,1536),(B,c,1536)
        # 按第一个维度拼接起来变为(B,a+b+c,1536)a+b+c=各层选择器输出维度之和
        # hs等于（B,1536,a+b+c）
        hs = torch.cat(hs, dim=1).transpose(1, 2).contiguous()  # B, S', C --> B, C, S
        # print(hs.size(), names)
        # （B,1536,7）
        hs = self.param_pool0(hs)

        ### adaptive adjacency
        # self.conv_q1(hs).shape大小为torch.Size([2, 384, 7])=（B,1536//4,7）
        # torch.Size([2, 7])
        q1 = self.conv_q1(hs).mean(1)
        # self.conv_k1(hs).shape大小为torch.Size([2, 384, 7])=（B,1536//4,7）
        # torch.Size([2, 7])
        k1 = self.conv_k1(hs).mean(1)
        # q1.unsqueeze(-1) - k1.unsqueeze(1)的形状为[2,7,7]
        # A1形状为[2,7,7]
        A1 = self.tanh(q1.unsqueeze(-1) - k1.unsqueeze(1))
        # 更新A1矩阵
        A1 = self.adj1 + A1 * self.alpha1
        ### graph convolution
        # 进行图卷积
        # （B,1536,7）
        hs = self.conv1(hs)
        # hs与可变邻接矩阵A1进行矩阵乘法
        hs = torch.matmul(hs, A1)
        # 正则化
        hs = self.batch_norm1(hs)
        ## predict
        # （B, 1536, 1）
        hs = self.param_pool1(hs)
        # print("池化后的hs大小",hs.shape)
        hs = self.dropout(hs)
        # torch.Size([B, 1536])
        hs = hs.flatten(1)
        # torch.Size([B, 200])
        hs = self.classifier(hs)
        return hs
# 处理部分结构信息
class Part_Structure(nn.Module):
    def __init__(self,hidden_size):
        super(Part_Structure, self).__init__()
        # 相对坐标信息
        self.relative_coord_predictor = RelativeCoordPredictor()
        # 设置GCN
        self.gcn =  GCN(2, 512, hidden_size, dropout=0.1)

    # 输入hidden_states制计即该层的输出(B, S, head_size)
    # 最大值索引(batch_size, num_attention_heads, 1)
    # 注意力特征图(B,C,H,H)C=注意力头数
    # struc_tokens (B,1,hidden_size)
    def forward(self, hidden_states, attention_map):
        # 注意力特征图的形状信息
        # C=注意力头数
        B,C,H,W = attention_map.shape
        # 得到注意力特征图的结构信息，基本锚点，位置权重，最大索引
        # 用于描述注意力图的结构特征和相对坐标
        # 相对坐标总和(N,S,2) S=H*W
        # 基础锚点(N,2)
        # 位置权重 (N,S,S)
        # 具有最大均值值像素的索引(N,)
        structure_info, basic_anchor, position_weight = self.relative_coord_predictor(attention_map)
        # structure_info为(N,S,512)
        structure_info = self.gcn(structure_info, position_weight)
        hidden_states_clone=hidden_states.clone()
        for i in range(B):
            index = int(basic_anchor[i,0]*H + basic_anchor[i,1])
            hidden_states_clone[i,0] = hidden_states_clone[i,0] + structure_info[i, index, :]
        hidden_states=hidden_states_clone
        return hidden_states



# 主干网络
class PluginMoodel(nn.Module):
    def __init__(self,
                 backbone: torch.nn.Module,
                 return_nodes: Union[dict, None],
                 img_size: int,
                 use_fpn: bool,
                 use_vit_fpn: bool,
                 fpn_size: Union[int, None],
                 proj_type: str,
                 upsample_type: str,
                 use_selection: bool,
                 num_classes: int,
                 num_selects: dict,
                 use_combiner: bool,
                 use_struct: bool,
                 comb_proj_size: Union[int, None]
                 ):
        super(PluginMoodel, self).__init__()
        self.return_nodes = return_nodes
        # 是否使用结构选择器
        self.use_struct=use_struct
        self.backbone=backbone
        # 是否使用fpn
        self.use_fpn=use_fpn
        # 是否使用选择器
        self.use_selection=use_selection
        # 是否使用结合器
        self.use_combiner=use_combiner
        rand_in = torch.randn(1, 3, img_size, img_size)
        outs, weights = self.backbone(rand_in)
        self.block_feature_map = create_feature_extractor(backbone, return_nodes=return_nodes)
        outs = self.block_feature_map(rand_in)
        last_key = list(return_nodes.keys())[-1]
        last_value = return_nodes[last_key]
        fs_size = outs[last_value].size()
        # print("得到最后的特征的大小",fs_size[-1])
        self.fpn_size = fs_size[-1]
        self.fpn_down=Vit_FPN(outs,self.fpn_size)
        self.build_fpn_classifier(outs,self.fpn_size,num_classes)
        self.num_selects=num_selects
        w_fpn_size = self.fpn_size
        self.selector = Selector(outs, num_classes,num_selects, w_fpn_size)
        self.part_select = Part_Attention()
        self.part_structure=Part_Structure(self.fpn_size)
        self.part_norm = nn.LayerNorm(self.fpn_size, eps=1e-6)

        gcn_inputs, gcn_proj_size = None, None
        total_num_selects = sum([num_selects[name] for name in num_selects])+len(num_selects) # sum
        # 设置图卷积聚合器
        # self.combiner = GCNCombiner(total_num_selects, num_classes, gcn_inputs, gcn_proj_size, self.fpn_size)
        input_dim=self.fpn_size*total_num_selects
        self.combiner=ClassMlp(input_dim=input_dim,hidden_dim=256,num_classes=num_classes)

    # 每个block的分类器
    def build_fpn_classifier(self, inputs: dict, fpn_size: int, num_classes: int):
        """
        Teh results of our experiments show that linear classifier in this case may cause some problem.
        """
        # 获取每层特征的名称
        for name in inputs:
            m=Mlp(fpn_size,num_classes,3072)
            # 用于每层的fpn进行分类
            self.add_module("fpn_classifier_" + name, m)
        # x为输入的图片
    def fpn_predict_vit_down(self, x: dict, logits: dict):
        for name in x:
            if "FPN1_" in name:
                continue
            logit = x[name].contiguous()
            # [1, 2304, 200]
            logits[name] = getattr(self, "fpn_classifier_" + name)(logit)

            # logits[name] = logits[name] # transpose
    def forward_backbone(self, x):
        return self.backbone(x)
    def forward(self, x: torch.Tensor):
        # 存储各个模块的结果
        logits = {}
        # x输入主干网络中得到每层的输出特征，输出outs和weights
        _, weights = self.forward_backbone(x)
        weights = weights[-3:]
        # 最后一层输出结果
        outs=self.block_feature_map(x)
        # 如果使用vit_fpn
        x = self.fpn_down(outs)
        self.fpn_predict_vit_down(x, logits)
        selects = self.selector(x, logits)
        # 从后三层开始
        i=0
        class_token=[]
        last_token=[]
        for name in outs:
            if i==2:
                part_states=outs[name]
            _,attention_map=self.part_select(weights[0:i+1])
            hidden_states=self.part_structure(outs[name],attention_map)
            class_token.append(torch.unsqueeze(self.part_norm(hidden_states[:,0]), 1))
            i+=1
        last_token.append(weights[-1])
        _,attention_map=self.part_select(last_token)
        part_states = self.part_structure(part_states, attention_map)
        part_encoded = self.part_norm(part_states)[:,0]
        # part_encoded=nn.functional.normalize(part_encoded)
        j=0
        for name in selects:
            selects[name]=torch.cat([class_token[j],selects[name]],dim=1)
            j+=1
        # 使用聚合器
        comb_outs = self.combiner(selects)
        logits['part_encoded']=part_encoded
        logits['comb_outs'] = comb_outs
        return logits

