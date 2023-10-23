import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from typing import Union
import copy
from .SICE import SICE
class GCNCombiner(nn.Module):

    def __init__(self, 
                 total_num_selects: int,
                 num_classes: int, 
                 inputs: Union[dict, None] = None, 
                 proj_size: Union[int, None] = None,
                 fpn_size: Union[int, None] = None):
        """
        If building backbone without FPN, set fpn_size to None and MUST give 
        'inputs' and 'proj_size', the reason of these setting is to constrain the 
        dimension of graph convolutional network input.
        """
        super(GCNCombiner, self).__init__()
        # inputs不为空且fpn_size不为空
        # assert inputs is not None or fpn_size is not None, \
        #     "To build GCN combiner, you must give one features dimension."
        # 确保输入参数inputs和fpn_size中至少有一个不为None
        assert inputs is not None or fpn_size is not None, \
            "To build GCN combiner, you must give one features dimension."

        ### auto-proj
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
                self.add_module("proj_"+name, m)
            self.proj_size = proj_size
        else:
            # proj_size=fpn_size
            self.proj_size = fpn_size

        ### build one layer structure (with adaptive module)
        # 总的选择器通道数除于64
        num_joints = total_num_selects // 64
        # 使用nn.Linear时，会判断total_num_selects与x的哪个维度相等，则在其维度进行变换
        self.param_pool0 = nn.Linear(total_num_selects, num_joints)
        # 创建一个单位矩阵A,大小为num_jointsxnum_joints,并将矩阵的值除以100后加上1/100
        A = torch.eye(num_joints) / 100 + 1 / 100
        #并将A进行拷贝设置为可训练参数adj1
        self.adj1 = nn.Parameter(copy.deepcopy(A))
        #
        self.conv1 = nn.Conv1d(self.proj_size, self.proj_size, 1)
        # 正则化
        self.batch_norm1 = nn.BatchNorm1d(self.proj_size)
        # 用于信息融合
        self.conv_q1 = nn.Conv1d(self.proj_size, self.proj_size//4, 1)
        self.conv_k1 = nn.Conv1d(self.proj_size, self.proj_size//4, 1)
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
                _tmp = getattr(self, "proj_"+name)(x[name])
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
        hs = torch.cat(hs, dim=1).transpose(1, 2).contiguous() # B, S', C --> B, C, S
        # print(hs.size(), names)
        #（B,1536,7）
        hs = self.param_pool0(hs)

        ### adaptive adjacency
        #self.conv_q1(hs).shape大小为torch.Size([2, 384, 7])=（B,1536//4,7）
        #torch.Size([2, 7])
        q1 = self.conv_q1(hs).mean(1)
        #self.conv_k1(hs).shape大小为torch.Size([2, 384, 7])=（B,1536//4,7）
        # torch.Size([2, 7])
        k1 = self.conv_k1(hs).mean(1)
        # q1.unsqueeze(-1).shape后torch.Size([2, 7, 1])
        # k1.unsqueeze(1).shape后torch.Size([2, 1, 7])
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

class WeaklySelector(nn.Module):

    def __init__(self, inputs: dict, num_classes: int, num_select: dict, fpn_size: Union[int, None] = None):
        """
        inputs: dictionary contain torch.Tensors, which comes from backbone
                [Tensor1(hidden feature1), Tensor2(hidden feature2)...]
                Please note that if len(features.size) equal to 3, the order of dimension must be [B,S,C],
                S mean the spatial domain, and if len(features.size) equal to 4, the order must be [B,C,H,W]
        """
        super(WeaklySelector, self).__init__()
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

    # def select(self, logits, l_name):
    #     """
    #     logits: [B, S, num_classes]
    #     """
    #     probs = torch.softmax(logits, dim=-1)
    #     scores, _ = torch.max(probs, dim=-1)
    #     _, ids = torch.sort(scores, -1, descending=True)
    #     sn = self.num_select[l_name]
    #     s_ids = ids[:, :sn]
    #     not_s_ids = ids[:, sn:]
    #     return s_ids.unsqueeze(-1), not_s_ids.unsqueeze(-1)
    # 输入长度为8的经过上采样和下采样的x以及对应的预测logits
    def forward(self, x, logits=None):
        """
        x : 
            dictionary contain the features maps which 
            come from your choosen layers.
            size must be [B, HxW, C] ([B, S, C]) or [B, C, H, W].
            [B,C,H,W] will be transpose to [B, HxW, C] automatically.
        """
        # 如果fpn_size即不使用fpn网络则logits为空
        if self.fpn_size is None:
            logits = {}
        selections = {}
        # 遍历x,仅对下采样的样本通过选择器,不选择上采样的样本
        for name in x:
            # print("[selector]", name, x[name].size())
            # 如果名字中为FPN1_即跳到下一循环,判断是不是上采样样本
            # 进行修改,目前仅对上采样样本进行选择
            if "FPN1_" in name:
                continue
            # 如果x[name]为四维
            if len(x[name].size()) == 4:
                B, C, H, W = x[name].size()
                # 变为三维并将第一维和第二维进行交换变为(B, H*W, C)
                x[name] = x[name].view(B, C, H*W).permute(0, 2, 1).contiguous()
            # 得到通道数
            C = x[name].size(-1)
            # 如果fpn为空,不适用fpn
            if self.fpn_size is None:
                # 对应名称的logits由分类器得到
                # model_name = name.replace("FPN1_", "")
                logits[name] = getattr(self, "classifier_l_"+name)(x[name])
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

class Vit_FPN(nn.Module):
    def __init__(self, inputs: dict, fpn_size: int):
        super(Vit_FPN, self).__init__()
        scale_factors=[4.0,2.0,1.0,0.5]
        inp_names = [name for name in inputs]
        inp_name=inp_names[-1]
        input=inputs[inp_name][:,1:,:]
        dim=input.size(-1)
        for idx,scale in enumerate(scale_factors):
            if scale==4.0:
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
                layers = [
                    nn.ConvTranspose1d(dim, dim // 2, kernel_size=2, stride=2),
                ]
                out_dim = dim // 2
                m = nn.Sequential(
                    nn.Linear(out_dim, out_dim, 1),
                    nn.ReLU(),
                    nn.Linear(out_dim, fpn_size, 1)
                )
            elif scale == 1.0:
                layers = []
                out_dim = dim
                m = nn.Sequential(
                    nn.Linear(out_dim, out_dim, 1),
                    nn.ReLU(),
                    nn.Linear(out_dim, fpn_size, 1)
                )
            elif scale == 0.5:
                layers = [nn.Conv1d(dim,dim*2,kernel_size=2,stride=2)]
                out_dim = dim*2
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

        hs = []
        outputs = {}
        #遍历x,i为索引,name为对应的层号key
        for i, name in enumerate(x):
            #如果字符串 "FPN1_" 在 name 中存在
            if "FPN1_" in name:
                continue
            # hs ['layer1', 'layer2', 'layer3', 'layer4']
            hs.append(name)
        for i in range(len(hs)):
            name=hs[i]
            # B,S,C
            outputs[f"layer{i+1}"]=getattr(self, "Proj_"+name)(getattr(self, "Up_"+name)(input).transpose(1, 2).contiguous())
        return outputs
class FPN(nn.Module):
    # upsample_type上采样类型
    # fpn_size=fpn大小
    def __init__(self, inputs: dict, fpn_size: int, proj_type: str, upsample_type: str):
        """
        inputs : dictionary contains torch.Tensor
                 which comes from backbone output
        fpn_size: integer, fpn 
        proj_type: 
            in ["Conv", "Linear"]

        upsample_type:
            in ["Bilinear", "Conv", "Fc"]
            # 卷积神经网络推荐插值
            # Vit推荐FC上采样
            # Swin-T推荐卷积上采样
            for convolution neural network (e.g. ResNet, EfficientNet), recommand 'Bilinear'. 
            for Vit, "Fc". and Swin-T, "Conv"
        """
        super(FPN, self).__init__()
        # 投影类型分卷积或线性层采样
        assert proj_type in ["Conv", "Linear"], \
            "FPN projection type {} were not support yet, please choose type 'Conv' or 'Linear'".format(proj_type)
        assert upsample_type in ["Bilinear", "Conv"], \
            "FPN upsample type {} were not support yet, please choose type 'Bilinear' or 'Conv'".format(proj_type)

        self.fpn_size = fpn_size
        # 上采样类型
        self.upsample_type = upsample_type
        # 获得主干网络输出的各层特征的名曾
        inp_names = [name for name in inputs]
        # 对主干网络的输出特征进行遍历
        for i, node_name in enumerate(inputs):
            ### projection module
            # 如果为卷积类型的投影方法
            # 讲H*W的转为fpn_size
            if proj_type == "Conv":
                m = nn.Sequential(
                    nn.Conv2d(inputs[node_name].size(1), inputs[node_name].size(1), 1),
                    nn.ReLU(),
                    nn.Conv2d(inputs[node_name].size(1), fpn_size, 1)
                )
            #  如果为线性回归
            elif proj_type == "Linear":
                # inputs[node_name]的形状为([1, 144, 1536])
                # 将所有的通道数均改为fpn_size的大小
                m = nn.Sequential(
                    nn.Linear(inputs[node_name].size(-1), inputs[node_name].size(-1)),
                    nn.ReLU(),
                    nn.Linear(inputs[node_name].size(-1), fpn_size),
                )
            self.add_module("Proj_"+node_name, m)

            ### upsample module
            # 上采样模块
            if upsample_type == "Conv" and i != 0:
                # 如果是三维的
                assert len(inputs[node_name].size()) == 3 # B, S, C
                # 获取当层的图片大小 H*W
                in_dim = inputs[node_name].size(1)
                # 获取下一层图片大小 H*W
                out_dim = inputs[inp_names[i-1]].size(1)
                # if in_dim != out_dim:
                # 将当层图片大小放大为上一层大小
                m = nn.Conv1d(in_dim, out_dim, 1) # for spatial domain
                # else:
                #     m = nn.Identity()
                self.add_module("Up_"+node_name, m)

        if upsample_type == "Bilinear":
            # 使用插值进行上采样，大小变为原来的两倍
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
    # 上采样
    def upsample_add(self, x0: torch.Tensor, x1: torch.Tensor, x1_name: str):
        """
        return Upsample(x1) + x1
        """


        # 如果是双插值法
        if self.upsample_type == "Bilinear":
            #如果x1和x0通道数不一样
            # 则将x1进行上采样
            if x1.size(-1) != x0.size(-1):
                x1 = self.upsample(x1)
        else:
            # 否则将x1进行上采样
            # x1上采样后与x0的形状一致
            x1 = getattr(self, "Up_"+x1_name)(x1)
        # 将同样形状的x0和x1进行相加
        return x1 + x0

    def forward(self, x):
        """
        x : dictionary
            {
                "node_name1": feature1,
                "node_name2": feature2, ...
            }
        """
        ### project to same dimension
        # 从主干网络中得到的x为
        # x : dictionary{
        #     "层号(layer1)": 特征1,
        #     "层号(layer2)": 特征2, ...
        # }
        hs = []
        #遍历x,i为索引,name为对应的层号key
        for i, name in enumerate(x):
            #如果字符串 "FPN1_" 在 name 中存在
            if "FPN1_" in name:
                continue
            # 输入函数前的数据[1, 144, 1536],[1, 2304, 384]
            # 将对应的x[name]作为输入,输入到函数名为"Proj_"+name的函数中
            # 将x的特征维度全部投影为fpn_size大小
            x[name] = getattr(self, "Proj_"+name)(x[name])
            # 经过函数后得到的数据,将通道数改为fpn_size[1, 144, 1536],[1, 2304, 1536]
            # hs ['layer1', 'layer2', 'layer3', 'layer4']
            hs.append(name)

        # 将最后一层("layer4")的的特征名设为"FPN1_" + "layer4"
        x["FPN1_" + "layer4"] = x["layer4"]
        # 倒数迭代,从索引len(hs)-1开始到1
        # range(len(hs)-1, 0, -1)
        # -1 表示从最后一个元素开始迭代,0 表示迭代到索引 0
        for i in range(len(hs)-1, 0, -1):

            # print("最上层大小为",x[hs[i]].shape)
            # 获得本层如果layer4
            x1_name = hs[i]
            # 获得上一层layer3
            x0_name = hs[i-1]
            # x[layer3]的特征变为x[layer3]+x[layer4](上采样后)
            x[x0_name] = self.upsample_add(x[x0_name], 
                                           x[x1_name], 
                                           x1_name)
            # x[layer3]的特征进行存储
            x["FPN1_" + x0_name] = x[x0_name]

        return x
# 暂时不用
# 自底向上融合特征
class FPN_UP(nn.Module):

    def __init__(self, 
                 inputs: dict, 
                 fpn_size: int):
        super(FPN_UP, self).__init__()

        inp_names = [name for name in inputs]

        for i, node_name in enumerate(inputs):
            ### projection module
            m = nn.Sequential(
                nn.Linear(fpn_size, fpn_size),
                nn.ReLU(),
                nn.Linear(fpn_size, fpn_size),
            )
            self.add_module("Proj_"+node_name, m)

            ### 下采样模块,如果不为最后一层
            if i != (len(inputs) - 1):
                assert len(inputs[node_name].size()) == 3 # B, S, C
                # 本层大小
                in_dim = inputs[node_name].size(1)
                # 下一层大小
                out_dim = inputs[inp_names[i+1]].size(1)
                # 本层大小缩小为下一层大小
                m = nn.Conv1d(in_dim, out_dim, 1) # for spatial domain
                self.add_module("Down_"+node_name, m)
                # print("Down_"+node_name, in_dim, out_dim)
                """
                Down_layer1 2304 576
                Down_layer2 576 144
                Down_layer3 144 144
                """

    def downsample_add(self, x0: torch.Tensor, x1: torch.Tensor, x0_name: str):
        """
        return Upsample(x1) + x1
        """
        # print("[downsample_add] Down_" + x0_name)
        # 将当前层进行下采样
        x0 = getattr(self, "Down_" + x0_name)(x0)
        # 将当前层和下一层进行相加
        return x1 + x0

    def forward(self, x):
        """
        x : dictionary
            {
                "node_name1": feature1,
                "node_name2": feature2, ...
            }
        """
        ### project to same dimension
        # x为每层输出的特征
        hs = []
        for i, name in enumerate(x):
            if "FPN1_" in name:
                continue
            # 将对应的x[name]作为输入,输入到函数名为"Proj_"+name的函数中
            x[name] = getattr(self, "Proj_"+name)(x[name])
            # 将输入通道改为统一的fpn_size的通道
            hs.append(name)

        # print(hs)
        # 从索引0开始遍历到索引len(hs) - 1(不包括)
        for i in range(0, len(hs) - 1):

            # print("当前层大小为",x[hs[i]].shape)
            # print("下一层大小为",x[hs[i+1]].shape)
            # 得到当前层的层号名
            x0_name = hs[i]
            # 得到下一层层号名
            x1_name = hs[i+1]
            # print(x0_name, x1_name)
            # print(x[x0_name].size(), x[x1_name].size())
            # 进行下采样得到从下到上的fpn特征
            x[x1_name] = self.downsample_add(x[x0_name], 
                                             x[x1_name], 
                                             x0_name)

        return x




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
                 use_sice: bool,
                 comb_proj_size: Union[int, None]
                 ):
        """
        * backbone: 
            torch.nn.Module class (recommand pretrained on ImageNet or IG-3.5B-17k(provided by FAIR))
        * return_nodes:
            e.g.
            return_nodes = {
                # node_name: user-specified key for output dict
                'layer1.2.relu_2': 'layer1',
                'layer2.3.relu_2': 'layer2',
                'layer3.5.relu_2': 'layer3',
                'layer4.2.relu_2': 'layer4',
            } # you can see the example on https://pytorch.org/vision/main/feature_extraction.html
            !!! if using 'Swin-Transformer', please set return_nodes to None
            !!! and please set use_fpn to True
        * feat_sizes: 
            tuple or list contain features map size of each layers. 
            ((C, H, W)). e.g. ((1024, 14, 14), (2048, 7, 7))
        * use_fpn: 
            boolean, use features pyramid network or not
        * fpn_size: 
            integer, features pyramid network projection dimension
        * num_selects:
            num_selects = {
                # match user-specified in return_nodes
                "layer1": 2048,
                "layer2": 512,
                "layer3": 128,
                "layer4": 32,
            }
        Note: after selector module (WeaklySelector) , the feature map's size is [B, S', C] which 
        contained by 'logits' or 'selections' dictionary (S' is selection number, different layer 
        could be different).
        """
        super(PluginMoodel, self).__init__()
        self.use_sice=use_sice
        if use_sice:
            self.representation = SICE(iterNum=8, is_sqrt=True, is_vec=True, input_dim=1536, dimension_reduction=256,
                                   sparsity_val=0.01, sice_lrate=5.0)
            self.fpn_classifier = nn.Linear(self.representation.output_dim, num_classes)

        ### = = = = = Backbone = = = = =
        # 主干网络
        # 如果返回节点不为空则从主干网络中指定的层数获得特征
        # return_nodes的格式为{'layer1': 'feat1', 'layer3': 'feat2'}
        self.return_nodes = return_nodes
        if return_nodes is not None:
            self.backbone = create_feature_extractor(backbone, return_nodes=return_nodes)
        else:
            self.backbone = backbone
        
        ### get hidden feartues size
        # 1为批次大小，3为通道数
        # 张量中的元素从标准正态分布（mean=0, std=1）中随机采样得到
        rand_in = torch.randn(1, 3, img_size, img_size)
        # 将随机得到的张量输入主干网络（swin-T）中得到每个stage的输出
        # 维度均为(b,h*w,c)
        # 第一层通过输入的(1,3,384,384)经过大小为4的patch之后变为(1,96,384/4,384/4)
        # 再通过patch_merge之后得到(1,96*4,384/4/2,384/4/2)得到(1,384,48,48)
        # 最后转换位置并转为三维得到(1,2043,48,48)
        # "layer1":(1,2304,384)
        # 对layer1输出进行patch_merge,H和W均除以2，C乘以2得到(1,576,768)
        # "layer2":(1,576,768)
        # "layer3":(1,144,1536)
        # "layer4":(1,144,1536)
        outs = self.backbone(rand_in)
        ### just original backbone
        # 如果未使用fpn网络并且未使用选择器和结合器,直接经由主干网络以及线性层分类器得到对应的输出
        if not use_fpn and (not use_selection and not use_combiner):
            # 遍历输出特征
            for name in outs:
                # 获取当前输出特征的形状
                fs_size = outs[name].size()
                # 如果为三维则为（B,S=H*W,C）
                if len(fs_size) == 3:
                    # 获得输出特征的最后一个维度即（通道数）
                    out_size = fs_size.size(-1)
                # 为四维则为(B,C,H,W)
                elif len(fs_size) == 4:
                    # 获取当前输出特征的第二个维度大小即通道数
                    out_size = fs_size.size(1)
                else:
                    raise ValusError("The size of output dimension of previous must be 3 or 4.")

            # 建立分类器输入维度根据上面得到，输出维度为图片类别数
            self.classifier = nn.Linear(out_size, num_classes)

        ### = = = = = FPN = = = = =

        self.use_fpn = use_fpn
        self.use_vit_fpn=use_vit_fpn
        # 如果要使用fpn
        if self.use_vit_fpn:
            self.fpn_down=Vit_FPN(outs,fpn_size)
            self.build_fpn_classifier_down(outs, fpn_size, num_classes)

        elif self.use_fpn:
            # 设置fpn网络自顶向下融合特征
            # 输入backone中的输出特征，fpn_size，投影类型，上采样类型
            self.fpn_down = FPN(outs, fpn_size, proj_type, upsample_type)
            # 自顶向下对特征融合后进行分类,输入干网络各层输出特征,fpn通道大小,图片类别大小
            # 创建分类器n1，输入各层输出特征，fpn_size,类别数
            self.build_fpn_classifier_down(outs, fpn_size, num_classes)
            # 设置自底向上融合特征
            self.fpn_up = FPN_UP(outs, fpn_size)
            # 自低向上对特征融合后进行分类,输入干网络各层输出特征,fpn通道大小,图片类别大小
            # 创建分类器n2
            self.build_fpn_classifier_up(outs, fpn_size, num_classes)

        self.fpn_size = fpn_size

        ### = = = = = Selector = = = = =
        self.use_selection = use_selection
        if self.use_selection:
            w_fpn_size = self.fpn_size if self.use_fpn else None # if not using fpn, build classifier in weakly selector
            self.selector = WeaklySelector(outs, num_classes, num_selects, w_fpn_size)

        ### = = = = = Combiner = = = = =
        self.use_combiner = use_combiner
        # 如果使用聚合器
        if self.use_combiner:
            # 使用聚合器之前要先使用选择器
            assert self.use_selection, "Please use selection module before combiner"
            # 如果使用了fpn网络，图神经网络输出和图神经网络投影大小均为空
            if self.use_fpn:
                gcn_inputs, gcn_proj_size = None, None\
            # 如果未使用fpn，则要设置gcn的输入的大小以及对应的gcn投影大小
            else:
                # comb_proj_size聚合器投影大小，
                gcn_inputs, gcn_proj_size = outs, comb_proj_size # redundant, fix in future
            # num_selects = {
            # 'layer1':32,
            # 'layer2':32,
            # 'layer3':32,
            # 'layer4':32
            # }
            # 计算所有层的输出通道之和
            total_num_selects = sum([num_selects[name] for name in num_selects]) # sum
            # 设置图卷积聚合器
            self.combiner = GCNCombiner(total_num_selects, num_classes, gcn_inputs, gcn_proj_size, self.fpn_size)

    def build_fpn_classifier_up(self, inputs: dict, fpn_size: int, num_classes: int):
        """
        Teh results of our experiments show that linear classifier in this case may cause some problem.
        """
        for name in inputs:
            m = nn.Sequential(
                    nn.Conv1d(fpn_size, fpn_size, 1),
                    nn.BatchNorm1d(fpn_size),
                    nn.ReLU(),
                    nn.Conv1d(fpn_size, num_classes, 1)
                )
            self.add_module("fpn_classifier_up_"+name, m)
    # 分类器
    def build_fpn_classifier_down(self, inputs: dict, fpn_size: int, num_classes: int):
        """
        Teh results of our experiments show that linear classifier in this case may cause some problem.
        """
        # 获取每层特征的名称
        for name in inputs:
            m = nn.Sequential(
                    nn.Conv1d(fpn_size, fpn_size, 1),
                    nn.BatchNorm1d(fpn_size),
                    nn.ReLU(),
                    nn.Conv1d(fpn_size, num_classes, 1)
                )
            # 用于每层的fpn进行分类
            self.add_module("fpn_classifier_down_" + name, m)

    def forward_backbone(self, x):
        return self.backbone(x)
    def fpn_predict_vit_down(self, x: dict, logits: dict):
        """
        x: [B, C, H, W] or [B, S, C]
           [B, C, H, W] --> [B, H*W, C]
        """
        # X[B, H*W, C]
        for name in x:
            if "FPN1_" in name:
                continue
            ### predict on each features point
            if len(x[name].size()) == 4:
                B, C, H, W = x[name].size()
                logit = x[name].view(B, C, H*W)
            elif len(x[name].size()) == 3:
                logit = x[name].transpose(1, 2).contiguous()
                # if self.use_sice:
                #     sice_output=self.representation(logit)
                #     sice_output = sice_output.view(sice_output.size(0), -1)
                #     sice_output=self.fpn_classifier(sice_output)
                #     sice_name="sice_"+name
                #     # print("sice的名字",sice_name)
                #     logits[sice_name]=sice_output
            logits[name] = getattr(self, "fpn_classifier_down_" + name)(logit)
            # [1, 2304, 200]
            logits[name] = logits[name].transpose(1, 2).contiguous() # transpose
    # 将自上到下的fpn进行分类,通道数变为200
    def fpn_predict_down(self, x: dict, logits: dict):
        """
        x: [B, C, H, W] or [B, S, C]
           [B, C, H, W] --> [B, H*W, C]
        """
        for name in x:
            # 跳过没有"FPN1_"开头的特征，表示仅对经过上采样后的特征进行处理
            if "FPN1_" not in name:
                continue 
            ### predict on each features point
            # 如果x[name]是四维[B, C, H, W]
            # 四维转为三维[B, H*W, C]
            if len(x[name].size()) == 4:
                B, C, H, W = x[name].size()
                logit = x[name].view(B, C, H*W)
            # 如果是3维
            elif len(x[name].size()) == 3:
                # 将第一维和第二维进行替换并保存在连续内存中
                # x从(B,H*W,C)转为(B, C, H*W)
                logit = x[name].transpose(1, 2).contiguous()
            # 将字符串 name 中的子字符串 "FPN1_" 替换为空字符串
            model_name = name.replace("FPN1_", "")
            # 对每层的fpn特征进行分类
            #  torch.Size([1, 200, 2304])
            # 保存每层经过上采样的特征进行分类器分配之后的张量
            logits[name] = getattr(self, "fpn_classifier_down_" + model_name)(logit)
            # 将第一个维度和第二个维度再次替换torch.Size([1, 2304, 200])
            logits[name] = logits[name].transpose(1, 2).contiguous() # transpose
    # 将自下到上的fpn进行分类,通道数变为200
    def fpn_predict_up(self, x: dict, logits: dict):
        """
        x: [B, C, H, W] or [B, S, C]
           [B, C, H, W] --> [B, H*W, C]
        """
        for name in x:
            if "FPN1_" in name:
                continue
            ### predict on each features point
            if len(x[name].size()) == 4:
                B, C, H, W = x[name].size()
                logit = x[name].view(B, C, H*W)
            elif len(x[name].size()) == 3:
                logit = x[name].transpose(1, 2).contiguous()
            model_name = name.replace("FPN1_", "")
            logits[name] = getattr(self, "fpn_classifier_up_" + model_name)(logit)
            # [1, 2304, 200]
            logits[name] = logits[name].transpose(1, 2).contiguous() # transpose
    # x为输入的图片
    def forward(self, x: torch.Tensor):

        logits = {}
        # x输入主干网络中得到每层的输出特征
        x = self.forward_backbone(x)
        # 如果使用vit_fpn
        if self.use_vit_fpn:
            x=self.fpn_down(x)
            self.fpn_predict_vit_down(x, logits)
        # 如果使用fpn
        elif self.use_fpn:
            # 将经由主干网络输出的特征输入fpn
            # 此时的x为{'layer1':'特征1',...."FPN1_layer4":'特征8'}
            # "FPN1_"开头代表的是经过上采样之后得到的各层特征
            x = self.fpn_down(x)
            # print([name for name in x])
            self.fpn_predict_down(x, logits)
            # print("经过fpn-up后得到的logits的长度:",len(logits))
            x = self.fpn_up(x)
            self.fpn_predict_up(x, logits)
            # logits长度为8,分别为FPN1_layer1,...,FPN1_layer4,layer1,...layer4
            # 前四个代表上采样后的预测,后四个代表下采样后的预测
            # print("经过fpn-down后得到的logits:",logits)
        # 使用选择器
        if self.use_selection:
            # 得到每层的经过选择器选择到的样本
            selects = self.selector(x, logits)
        # 使用聚合器
        if self.use_combiner:
            # 得到comb_outs=[B,200]
            comb_outs = self.combiner(selects)
            comb_outs=self.representation(comb_outs)
            comb_outs=comb_outs.view(comb_outs.size(0), -1)
            comb_outs = self.fpn_classifier(comb_outs)
            logits['comb_outs'] = comb_outs
            return logits
        # 如果仅使用了选择器或者fpn
        if self.use_selection or self.fpn:
            return logits

        ### original backbone (only predict final selected layer)
        # 如果仅使用主干网络
        for name in x:
            hs = x[name]

        if len(hs.size()) == 4:
            hs = F.adaptive_avg_pool2d(hs, (1, 1))
            hs = hs.flatten(1)
        else:
            hs = hs.mean(1)
        out = self.classifier(hs)
        logits['ori_out'] = logits

        return logits