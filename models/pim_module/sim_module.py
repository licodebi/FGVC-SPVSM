# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm, Parameter
from torch.nn.modules.utils import _pair
from scipy import ndimage

import models.pim_module.configs as configs


# 图卷积层模块
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=False, dropout=0.1):
        super(GraphConvolution, self).__init__()
        # 输入特征
        self.in_features = in_features
        # 输出特征
        self.out_features = out_features
        # 权重矩阵
        self.weight = nn.Parameter(torch.zeros(in_features, out_features))
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
        # 得到输出特征的标准差的倒数
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


'''Laplacian Matrix transorm'''


def gen_adj(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj


logger = logging.getLogger(__name__)

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


# 自注意力机制
class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        # 注意力头数
        self.num_attention_heads = config.transformer["num_heads"]
        # 每个头的维度大小=总维度大小/注意力头数
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # 经过线性层得到q,k,v
        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)
        # 将注意力计算后的特征进行线性变换的全连接层
        self.out = Linear(config.hidden_size, config.hidden_size)
        # 是用于进行随机失活的dropout层
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])
        # 用于进行注意力权重归一化的softmax函数，将注意力权重映射到概率分布
        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        # 用于将输入张量进行形状转换，以适应多头注意力计算的要求
        # 将输入张量x的形状为(batch_size, seq_length, hidden_size)
        # 转换为(batch_size, seq_length, num_attention_heads, attention_head_size)
        # 通过维度重排，得到(batch_size, num_attention_heads, seq_length, attention_head_size)的形状。
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    # 输入图片的embedding
    def forward(self, hidden_states):
        # 得到Q,K,V
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        # 将QKV转为 (batch_size, num_attention_heads, seq_length, attention_head_size)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Q和K进行矩阵乘法得到注意力权重大小为 (batch_size, num_attention_heads, seq_length, seq_length)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        # 得到注意力权重张量
        weights = attention_probs
        # 进行dropout (batch_size, num_attention_heads, seq_length, seq_length)
        attention_probs = self.attn_dropout(attention_probs)

        # 注意力权重与V进行矩阵乘法,得到上下文向量
        # (batch_size, num_attention_heads, seq_length, attention_head_size)
        context_layer = torch.matmul(attention_probs, value_layer)
        # 进行维度重排(batch_size, seq_length, num_attention_heads, attention_head_size)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # 制定形状为(batch_size, seq_length, all_head_size)
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        # 将上下文向量的形状进行更改变为 (batch_size, seq_length, all_head_size)
        context_layer = context_layer.view(*new_context_layer_shape)
        # 进行全连接层计算
        attention_output = self.out(context_layer)
        # 进行dropout
        attention_output = self.proj_dropout(attention_output)
        # 返回注意力机制计算后的张量以及对应的注意力权重
        # 注意力机制计算后的张量(batch_size, seq_length, all_head_size)
        # 注意力权重 (batch_size, num_attention_heads, seq_length, seq_length)
        return attention_output, weights


# 多层感知机类
class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        # 设置第一个全连接层，维度从config.hidden_size变换为config.transformer["mlp_dim"]
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        # 设置第二个全连接层，维度从config.transformer["mlp_dim"]变换回config.hidden_size
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        # 使用了GELU激活函数
        self.act_fn = ACT2FN["gelu"]
        # 按照指定的config.transformer["dropout_rate"]概率将输入的部分元素置为零
        self.dropout = Dropout(config.transformer["dropout_rate"])
        # 对全连接层的参数进行初始化
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


# 嵌入
class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """

    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        # 使得图片的高度和宽度一样
        # 得到patch_size
        img_size = _pair(img_size)
        patch_size = _pair(config.patches["size"])
        # 当图片分割是不重叠时
        if config.split == 'non-overlap':
            # 长和宽分别除以patch_size大小，得到n_patches即patch的数量
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            # 输出通道为hidden_size大小，将图片大小缩小patch_size倍
            self.patch_embeddings = Conv2d(in_channels=in_channels,
                                           out_channels=config.hidden_size,
                                           kernel_size=patch_size,
                                           stride=patch_size)
        # 当图片分割是重叠时
        elif config.split == 'overlap':
            # patch的数量为[((图片的高-patch的高)/滑动窗口)+1]*[((图片的高-patch的高)/滑动窗口)+1]
            n_patches = ((img_size[0] - patch_size[0]) // config.slide_step + 1) * (
                        (img_size[1] - patch_size[1]) // config.slide_step + 1)
            self.patch_embeddings = Conv2d(in_channels=in_channels,
                                           out_channels=config.hidden_size,
                                           kernel_size=patch_size,
                                           stride=(config.slide_step, config.slide_step))
        # 设置位置嵌入，B=1 C=patch的数量+1 S=hidden_size
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches + 1, config.hidden_size))
        # 设置cls嵌入用于最后的分类 B=1 C=1 S=hidden_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        # 设置dropout
        self.dropout = Dropout(config.transformer["dropout_rate"])

        # 图像结构嵌入 B=1 C=1 S=hidden_size
        self.struc_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        # 预测相对坐标信息
        self.relative_coord_predictor = RelativeCoordPredictor()
        # 定义模型的结构头部
        # 指神经网络模型的最后一层或几层
        self.struct_head = nn.Sequential(
            nn.BatchNorm1d(37 * 37 * 2 + 2),
            Linear(37 * 37 * 2 + 2, 1024),
            nn.BatchNorm1d(1024),
            nn.ELU(inplace=True),
            Linear(1024, config.hidden_size),
        )
        # 设置激活函数 act_fn 为 "relu"
        self.act_fn = ACT2FN["relu"]

    def get_attn(self, x, y):
        attn = F.normalize(x) * F.normalize(y)
        attn = torch.sum(attn, -1)

        H = attn.size(1) ** 0.5
        H = int(H)

        attn = attn.contiguous().view(attn.size(0), H, H)
        attn = attn.unsqueeze(dim=1)

        B, C, H, W = attn.shape
        structure_info = self.relative_coord_predictor(attn)
        structure_info = self.struct_head(structure_info)
        structure_info = self.act_fn(structure_info)
        structure_info = structure_info.unsqueeze(1)

        return structure_info

    # 输入图片
    def forward(self, x):
        # 根据第一个维度得到batch_size
        B = x.shape[0]
        # 得到cls_tokens (B,1,config.hidden_size)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        # 得到struc_tokens (B,1,config.hidden_size)
        struc_tokens = self.struc_token.expand(B, -1, -1)
        # 不使用
        if self.hybrid:
            x = self.hybrid_model(x)
        # 对x进行patch嵌入化,(B, config.hidden_size, OH, OW)
        x = self.patch_embeddings(x)
        # (B,config.hidden_size,n_patches)
        x = x.flatten(2)
        # (B,n_patches,config.hidden_size)
        x = x.transpose(-1, -2)
        # (B,n_patches+1,config.hidden_size)
        x = torch.cat((cls_tokens, x), dim=1)
        # (B,n_patches+1,config.hidden_size)
        embeddings = x + self.position_embeddings
        # (B,n_patches+1,config.hidden_size)
        embeddings = self.dropout(embeddings)
        # 返回x的embeddings,以及struc_tokens
        return embeddings, struc_tokens


# Transformer中的每个Block
class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        # 设置隐藏层大小
        self.hidden_size = config.hidden_size
        # 设置层归一化，设置eps防止除零
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        # 设置层归一化，用于对前馈神经网络（FFN）的输出进行层归一化操作
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        # 设置多层感知机（MLP）
        self.ffn = Mlp(config)
        # 设置多头注意力机制
        self.attn = Attention(config)

    # 输入图片的embedding即x
    def forward(self, x):
        h = x
        # 对x进行层归一化
        x = self.attention_norm(x)

        # 进行注意力计算，得到计算后的张量以及对应的权重张量
        x, weights = self.attn(x)

        # 与之前进行残差相加
        x = x + h
        # 记录残差之和
        h = x

        # 进行层归一化
        x = self.ffn_norm(x)

        # 前馈网络进行MLP计算
        x = self.ffn(x)
        # MLP计算结果和残差之和进行相加
        x = x + h
        # 返回改Block的输出以及Block中的注意力权重张量
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size,
                                                                                   self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size,
                                                                                   self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size,
                                                                                   self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


# 用于计算部分注意力
class Part_Attention(nn.Module):
    def __init__(self):
        super(Part_Attention, self).__init__()

    # 输入对应的注意力权重张量 (batch_size, num_attention_heads, S, S)数组
    # 即第9层以后该Transformer的每层的注意力权重张量数组
    def forward(self, x):
        # 得到注意力权重张量的数量,即第9层以后的层数
        length = len(x)
        # 得到第十层的注意力权重张量
        last_map = x[0]
        # 十层之后的每一层的注意力权重张量均和上一层的注意力权重相乘
        # (batch_size, num_attention_heads, S, S)
        for i in range(1, length):
            last_map = torch.matmul(x[i], last_map)
        # (batch_size, num_attention_heads, 1, S-1)
        last_map = last_map[:, :, 0, 1:]
        # 得到last_map 在第二个维度上的最大值和对应的索引
        # (batch_size, num_attention_heads, S-1)
        max_value, max_inx = last_map.max(2)

        # 得到B,C
        B, C = last_map.size(0), last_map.size(1)
        # 得到patch数
        patch_num = last_map.size(-1)
        # 根据patch数得到高
        H = patch_num ** 0.5
        H = int(H)
        # C=注意力头数
        attention_map = last_map.view(B, C, H, H)
        # last_map(batch_size, num_attention_heads, 1, S-1)
        # 最大值索引(batch_size, num_attention_heads, 1)
        # 最大值(batch_size, num_attention_heads, 1)
        # 注意力特征图(B,C,H,H)
        return last_map, max_inx, max_value, attention_map


# 预测相对坐标信息
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
        # reduced_x_max_index形状为(N,)
        _, reduced_x_max_index = torch.max(torch.mean(masked_x, dim=-1), dim=-1)

        # 构建一个基础索引张量basic_index，其元素值为0到N-1的整数,并将其移动到GPU上
        basic_index = torch.from_numpy(np.array([i for i in range(N)])).cuda()

        # 生成一个形状为(size,size,2)的张量 size=H
        basic_label = torch.from_numpy(self.build_basic_label(size)).float()

        # Build Label
        label = basic_label.cuda()
        # 通过扩展操作得到与masked_x形状相匹配的张量形状(N, H, W, 2)
        # 将其转换为张量形状(N, H*W, 2)
        label = label.unsqueeze(0).expand((N, H, W, 2)).view(N, H * W, 2)  # (N, S, 2)
        # 通过索引操作label[basic_index, reduced_x_max_index, :]得到形状为(N, 1, 2)的三维张量。
        # reduced_x_max_index为之前得到的最大值的像素的索引
        # 得到每个样本的基础锚点（anchor）坐标
        basic_anchor = label[basic_index, reduced_x_max_index, :].unsqueeze(1)  # (N, S, 2)

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
        binary_relative_mask = binary_mask.view(N, H * W)
        # 将相对距离和相对角度乘以掩码，将非掩码的位置置零
        relative_dist = relative_dist * binary_relative_mask
        relative_angle = relative_angle * binary_relative_mask
        # 调整基本锚点的形状(N, 2)
        basic_anchor = basic_anchor.squeeze(1)  # (N, 2)
        # 将相对距离和相对角度的维度新增一维均变为(N,S,1)
        # 再将最后一维进行拼接得到(N,S,2)
        relative_coord_total = torch.cat((relative_dist.unsqueeze(2), relative_angle.unsqueeze(2)), dim=-1)

        # 将具有掩码操作后的x的最后一维进行平均得到 (N, H*W)的张量position_weight
        position_weight = torch.mean(masked_x, dim=-1)
        # 将position_weight转为(N, H*W,1)
        position_weight = position_weight.unsqueeze(2)
        # 再将position_weight以及其转置相乘得到(N, H*W,H*W)的张量作为位置权重
        position_weight = torch.matmul(position_weight, position_weight.transpose(1, 2))
        # 返回
        # 相对坐标总和(N,S,2)
        # 基础锚点(N,S,2)
        # 位置权重 (N, H*W,H*W) (N, S,S)
        # 具有最大均值值像素的索引(N,)
        return relative_coord_total, basic_anchor, position_weight, reduced_x_max_index

    def build_basic_label(self, size):
        basic_label = np.array([[(i, j) for j in range(size)] for i in range(size)])
        return basic_label


# 图卷积神经网络
# nfeat输入特征的维度，nhid隐藏层维度，nclass
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


# 处理部分结构信息
class Part_Structure(nn.Module):
    def __init__(self, config):
        super(Part_Structure, self).__init__()
        # 全连接层fc1
        self.fc1 = Linear(37 * 37 * 2 + 2, config.hidden_size)
        # 激活函数
        self.act_fn = ACT2FN["relu"]
        # dropout
        self.dropout = Dropout(config.transformer["dropout_rate"])
        # 相对坐标信息
        self.relative_coord_predictor = RelativeCoordPredictor()
        # 结构头
        self.struct_head = nn.Sequential(
            nn.BatchNorm1d(37 * 37 * 2 + 2),
            Linear(37 * 37 * 2 + 2, config.hidden_size),
        )
        # 新的结构头，进一步处理结构信息。
        self.struct_head_new = nn.Sequential(
            nn.BatchNorm1d(config.hidden_size * 2),
            Linear(config.hidden_size * 2, 1024),
            nn.BatchNorm1d(1024),
            nn.ELU(inplace=True),
            Linear(1024, config.hidden_size),
        )
        # 设置GCN
        self.gcn = GCN(2, 512, config.hidden_size, dropout=0.1)

    # 输入hidden_states制计即该层的输出(B, S, head_size)
    # 最大值索引(batch_size, num_attention_heads, 1)
    # 注意力特征图(B,C,H,H)C=注意力头数
    # struc_tokens (B,1,hidden_size)
    def forward(self, hidden_states, part_inx, part_value, attention_map, struc_tokens):
        # 注意力特征图的形状信息
        # C=注意力头数
        B, C, H, W = attention_map.shape
        # 得到注意力特征图的结构信息，基本锚点，位置权重，最大索引
        # 用于描述注意力图的结构特征和相对坐标
        # 相对坐标总和(N,S,2) S=H*W
        # 基础锚点(N,S,2)
        # 位置权重 (N,S,S)
        # 具有最大均值值像素的索引(N,)
        structure_info, basic_anchor, position_weight, reduced_x_max_index = self.relative_coord_predictor(
            attention_map)
        # structure_info为(N,S,512)
        structure_info = self.gcn(structure_info, position_weight)

        for i in range(B):
            index = int(basic_anchor[i, 0] * H + basic_anchor[i, 1])
            hidden_states[i, 0] = hidden_states[i, 0] + structure_info[i, index, :]

        return hidden_states


# 编码器
class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        # 设置Encoder的层，每个层都具有各自的块
        self.layer = nn.ModuleList()
        # 根据设置的层数，进行迭代，创建各个层
        for _ in range(config.transformer["num_layers"] - 1):
            layer = Block(config)
            self.layer.append(copy.deepcopy(layer))
        # 选择输入序列的一部分进行处理，用于关注输入序列的某个子集
        self.part_select = Part_Attention()
        # 对部分选择的输入进行特征抽取和表示学习，使用的是与其他层相同的Block模块
        self.part_layer = Block(config)
        # 对部分选择后的输入进行归一化，以便在不同模块之间保持一致的特征尺度
        self.part_norm = LayerNorm(config.hidden_size, eps=1e-6)
        # 处理部分结构（part structure）信息
        self.part_structure = Part_Structure(config)

    # 输入x的embedding和struc_tokens
    def forward(self, hidden_states, struc_tokens):
        # 注意力权重
        attn_weights = []
        hid_ori = []
        structure = []
        # 读取每一层
        for i, layer in enumerate(self.layer):
            # 输入上一层的输出，得到对应的层输出和层中的注意力权重张量
            hidden_states, weights = layer(hidden_states)
            # 记录注意力权重张量
            attn_weights.append(weights)

            # 如果该层为第9层以下
            if i > 8:
                # 记录9层以后的每层的注意力权重张量
                temp_weight = []
                temp_weight.append(weights)
                # last_map(batch_size, num_attention_heads, 1, S-1)
                # 最大值索引(batch_size, num_attention_heads, 1)
                # 最大值(batch_size, num_attention_heads, 1)
                # 注意力特征图(B,C,H,H)
                _, part_inx, part_value, a_map = self.part_select(temp_weight)
                # 输入hidden_states制计即该层的输出(B, S, head_size)
                # 最大值索引(batch_size, num_attention_heads, 1)
                # 注意力特征图(B,C,H,H)C=注意力头数
                # struc_tokens (B,1,hidden_size)
                # hidden_states(B, S, head_size)
                # hidden_states的[:,0]保存结构信息
                hidden_states = self.part_structure(hidden_states, part_inx, part_value, a_map, struc_tokens)
                # 保存每层的结构信息
                hid_ori.append(self.part_norm(hidden_states[:, 0]))
        # 最后一层输出再通过一个block得到相应的输出和权重矩阵
        part_states, part_weights = self.part_layer(hidden_states)
        # 将权重矩阵添加进数组中
        attn_weights.append(part_weights)
        # 将数组清空后再将权重矩阵加入其中
        temp_weight = []
        temp_weight.append(part_weights)
        _, part_inx, part_value, a_map = self.part_select(temp_weight)
        # 得到最后一层的输出
        part_states = self.part_structure(part_states, part_inx, part_value, a_map, struc_tokens)
        part_encoded = self.part_norm(part_states)
        # 返回最后一层的输出以及之前层的结构信息数组
        return part_encoded, hid_ori


# config：关于Vit的参数
# img_size：图片大小
class Transformer(nn.Module):
    def __init__(self, config, img_size):
        super(Transformer, self).__init__()

        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config)

    # 输入图片
    def forward(self, input_ids):
        # 输入图片到embeddings中得到x的embedding以及struc_tokens
        embedding_output, struc_tokens = self.embeddings(input_ids)
        # 输入x的embedding以及struc_tokens
        # 得到最后一层的输出以及之前的结构信息
        part_encoded, hid = self.encoder(embedding_output, struc_tokens)
        # print(hid.shape)
        return part_encoded, hid


# zero_head是否将模型的最后一层分类头置零
class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, smoothing_value=0, zero_head=False):
        super(VisionTransformer, self).__init__()
        # 标签类别数量
        self.num_classes = num_classes
        # 标签平滑的平滑值。默认为0，表示不进行标签平滑。如果需要进行标签平滑，可以将其设置为一个小于1的正数，用于平滑标签分配
        self.smoothing_value = smoothing_value
        self.zero_head = zero_head
        # 设置分类器
        self.classifier = config.classifier
        # 设置transformer
        self.transformer = Transformer(config, img_size)
        # 设置part_head
        self.part_head = nn.Sequential(
            nn.BatchNorm1d(config.hidden_size * 3),
            Linear(config.hidden_size * 3, 1024),
            nn.BatchNorm1d(1024),
            nn.ELU(inplace=True),
            Linear(1024, num_classes),
        )

    # x:图片,lables:标签,
    #
    def forward(self, x, labels=None, step=0, global_step=10000):
        # 图片输入到transformer中，返回最后一层输出以及之前层的结构信息数组
        part_tokens, hid = self.transformer(x)
        # 将后三层的结构信息进行连接得到(B,1,hidden_size*3)
        final_hid = torch.cat((hid[-2], hid[-1], part_tokens[:, 0]), dim=-1)
        # 得到(B,1,num_classes)
        part_logits = self.part_head(final_hid)

        if labels is not None:
            if self.smoothing_value == 0:
                loss_fct = CrossEntropyLoss()
            else:
                loss_fct = LabelSmoothing(self.smoothing_value)

            part_loss = loss_fct(part_logits.view(-1, self.num_classes), labels.view(-1))
            # 输入最后一层的结构信息，(B,1,hidden_size)
            # labels(B,)
            contrast_loss = con_loss_new(part_tokens[:, 0], labels.view(-1), step, global_step)

            loss = part_loss + contrast_loss

            return loss, part_logits
        else:
            return part_logits

    def load_from(self, weights):
        with torch.no_grad():
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.transformer.encoder.part_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.part_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            for bname, block in self.transformer.encoder.named_children():
                if bname.startswith('part') == False:
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(
                    np2th(weights["conv_root/kernel"], conv=True))
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)
                gn_bias = np2th(weights["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname)


def cosine_anneal_schedule(t, nb_epoch, lr):
    cos_inner = np.pi * (t % (nb_epoch))  # t - 1 is used when t has 1-based indexing.
    cos_inner /= (nb_epoch)
    cos_out = np.cos(cos_inner) + 1

    return float(lr / 2 * cos_out)


# 计算对比损失（contrastive loss）的函数。
# 对比损失通常用于训练具有特征嵌入的模型，以使相同类别的样本在嵌入空间中更加接近，而不同类别的样本则更加分散
def con_loss_new(features, labels, step, global_step, ):
    eps = 1e-6

    B, _ = features.shape
    features = F.normalize(features)
    # # 计算归一化特征之间的余弦相似性矩阵
    cos_matrix = features.mm(features.t())

    # 这些行创建了一个大小为 (B, B) 的正标签矩阵和负标签矩阵
    # 相同类别的样本对应位置为1，不同类别的样本对应位置为0
    pos_label_matrix = torch.stack([labels == labels[i] for i in range(B)]).float()
    # 得到一个负标签矩阵，相同的位置为0，不同的为1
    neg_label_matrix = 1 - pos_label_matrix
    # 这行创建了一个与 pos_label_matrix 形状相同的新负标签矩阵 neg_label_matrix_new
    neg_label_matrix_new = 1 - pos_label_matrix

    # 正余弦相似性矩阵，用于衡量正样本对应位置的相似性
    pos_cos_matrix = 1 - cos_matrix
    # 负余弦相似性矩阵，用于衡量负样本对应位置的相似性
    neg_cos_matrix = 1 + cos_matrix

    # 定义了一个边界值（边界间隔），用于控制正样本之间的最小相似性要求
    margin = 0.3
    # 将矩阵元素的取值范围从[0, 2]映射到[0, 1]
    sim = (1 + cos_matrix) / 2.0
    # 得分矩阵，1减去相似性矩阵（sim）中对应位置上的值，表示两个特征之间的差异程度的得分矩阵
    # 得分值表示较高的相似性或较低的差异性，较高的得分值表示较低的相似性或较高的差异性
    # scores范围在(0,1)
    scores = 1 - sim
    # pos_label_matrix中等于1的位置返回得分矩阵中对应位置的值，不等于1的位置则返回0
    # positive_scores作为正对比分数矩阵
    positive_scores = torch.where(pos_label_matrix == 1.0, scores, scores - scores)
    # mask为(B,B)的对角矩阵，作为掩码
    mask = torch.eye(features.size(0)).cuda()
    # 运用掩码将正对比分数矩阵中对角线元素设置为零
    positive_scores = torch.where(mask == 1.0, positive_scores - positive_scores, positive_scores)
    # 先对positive_scores中的每个样本的得分进行求和，得到a
    # 再计算每个样本中正样本（标签为1）的数量，得到b，b=-1+eps,-1是为了减去与自身匹配的1，+eps是为了防止除以0
    # 将a/b得到每个样本中正样本的平均得分
    positive_scores = torch.sum(positive_scores, dim=1, keepdim=True) / (
                (torch.sum(pos_label_matrix, dim=1, keepdim=True) - 1) + eps)
    # 将positive_scores(B,1)在列方向上重复B次得到(B,B)
    positive_scores = torch.repeat_interleave(positive_scores, B, dim=1)

    # 得到平均正对比分数与样本得分之间的相对距离
    relative_dis1 = margin + positive_scores - scores
    # 找出relative_dis1中小于 0 的元素，即负样本与正样本的距离比边界值还小
    # 代码将相应位置在 neg_label_matrix_new 中的元素设置为 0。
    # 这样，neg_label_matrix_new 中的 0 值表示该位置上的样本为正样本
    neg_label_matrix_new[relative_dis1 < 0] = 0
    # 更新 neg_label_matrix
    neg_label_matrix = neg_label_matrix * neg_label_matrix_new

    loss = (pos_cos_matrix * pos_label_matrix).sum() + (neg_cos_matrix * neg_label_matrix).sum()
    loss /= B * B

    return loss


CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'testing': configs.get_testing(),
}
