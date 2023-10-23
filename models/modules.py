# coding=utf-8

import copy
import math
from os.path import join as pjoin
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn import Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair

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


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.mlp_dim)
        self.fc2 = Linear(config.mlp_dim, config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.dropout_rate)

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


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """

    def __init__(self, config, img_size,split,in_channels=3):
        super(Embeddings, self).__init__()
        img_size = _pair(img_size)
        patch_size = _pair(config.patches)
        if split=='non-overlap':
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        elif split=='overlap':
            slide_step=12
            n_patches = ((img_size[0] - patch_size[0]) // slide_step + 1) * (
                        (img_size[1] - patch_size[1]) //slide_step + 1)
            self.patch_embeddings = Conv2d(in_channels=in_channels,
                                           out_channels=config.hidden_size,
                                           kernel_size=patch_size,
                                           stride=(slide_step, slide_step))
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches + 1, config.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.dropout = Dropout(config.dropout_rate)

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = self.patch_embeddings(x)

        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = torch.cat((cls_tokens, x), dim=1)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList()
        # for _ in range(config.num_layers):
        for _ in range(config.num_layers + 1):
            layer = Block(config)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        # attmap = []
        for layer in self.layer:
            hidden_states, weights = layer(hidden_states)
        # print(weights.shape)
        # attmap.append(weights)
        return hidden_states


class Transformer(nn.Module):
    def __init__(self, config, img_size):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        part_encoded = self.encoder(embedding_output)
        return part_encoded


class Attention(nn.Module):
    def __init__(self, config,coeff_max=0.25):
        super(Attention, self).__init__()
        self.coeff_max = coeff_max
        # 注意力头数
        self.num_attention_heads = config.num_heads
        # 每个头大小
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        # 总维度大小
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # 投影qkv维度
        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)
        # 注意力后的输出
        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.att_dropout)
        self.proj_dropout = Dropout(config.att_dropout)
        self.softmax = Softmax(dim=-1)
        self.softmax2 = Softmax(dim=-2)
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states,mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        # qkv转为(B,Head,S,C/num_heads)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        # 权重矩阵
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # 对注意力权重进行缩放
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if mask is not None:
            # if debug_mode:
            #     # 如果随机生成的浮点数小于0.000001，否则被设置为False
            #     print_info = True if (random.random() < 0.000001) else False
            #     x = random.random()
            #     # x是否在区间(0.00005, 0.00007)内，如果是，则print_info设置为True，否则设置为False
            #     if (x > 0.00005) and (x < 0.00007):
            #         print_info = True
            #     else:
            #         print_info = False
            # else:
            #     print_info = False
            # attention_scores(B,C,S)
            # max_as大小为(B,C),取到每个头的最大值
            max_as = torch.max(attention_scores[:, :, 0, :], dim=2, keepdim=False)[0]
            max_as = max_as.to(device='cuda')
            mask_626 = torch.zeros(mask.size(0), (mask.size(1) + 1))  # , dtype=torch.float64) # dtype=torch.double)
            mask_626 = mask_626.to(device='cuda')
            mask_626[:, 1:] = mask[:, :]
            mask_626[:, 0] = 0
            # positive only, obj + (max * coeff):
            attention_scores[:, :, 0, :] = \
                torch.where(mask_626[:, None, :] < 0.5, torch.add(attention_scores[:, :, 0, :],
                                                                  torch.mul(max_as[:, :, None],
                                                                            torch.tensor(self.coeff_max).cuda())), \
                            attention_scores[:, :, 0, :]  # .float()
                            )

        attention_scores = self.dropkey(attention_scores,0.1)
        # 将权重转为概率形式
        attention_probs = self.softmax(attention_scores)
        # B,C,S(H+1)
        contribution=self.softmax2(attention_scores[:,:,:,:])[:,:,:,0]
        weights = attention_probs
        # attention_probs = self.attn_dropout(attention_probs)
        # print("weights的形状",weights.shape)

        context_layer = torch.matmul(attention_probs, value_layer)
        # 转为(B,S,Head,C/num_heads)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # 转为(B,S,C)
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        # 返回注意力输出，权重矩阵概率分布
        return attention_output, weights,contribution
    def dropkey(self,attention,mask_ratio):
        m_r=torch.ones_like(attention)*mask_ratio
        attention=attention+torch.bernoulli(m_r)*(-1e-12)
        return attention

class Block(nn.Module):
    def __init__(self, config,coeff_max):
        super(Block, self).__init__()
        # 隐藏层大小
        self.hidden_size = config.hidden_size
        # 注意力层norm
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        # fw的norm
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        # mlp
        self.ffn = Mlp(config)
        self.attn = Attention(config,coeff_max)

    def forward(self, x, mask=None):
        h = x
        x = self.attention_norm(x)
        x, weights,contribution = self.attn(x)
        x = x + h
        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights,contribution

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
# 多头投票模块

# if __name__ == '__main__':
# from core.vit import *
# config = get_b16_config()
# config.hidden_size = 4
