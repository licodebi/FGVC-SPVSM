import time
import numpy as np
import torch
from scipy import ndimage
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import torch.nn as nn
from models.modules import *
from models.vit import get_b16_config

class SPTransformer(nn.Module):
    def __init__(self,config,num_classes,img_size,update_warm,patch_num,total_num,split='non-overlap',coeff_max=0.25):
        super(SPTransformer, self).__init__()
        self.num_classes=num_classes
        self.img_size=img_size
        self.embeddings=Embeddings(config,img_size=img_size,split=split)
        self.encoder=SAPEncoder(config,update_warm,patch_num,total_num,split,coeff_max)
        # self.head = Linear(config.hidden_size, num_classes)
        self.head = nn.Sequential(
            nn.BatchNorm1d(config.hidden_size),
            Linear(config.hidden_size , 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            Linear(512, num_classes),
        )
        self.softmax = Softmax(dim=-1)
        self.part_head = nn.Sequential(
            nn.BatchNorm1d(config.hidden_size * 4),
            Linear(config.hidden_size * 4, 1024),
            nn.BatchNorm1d(1024),
            nn.ELU(inplace=True),
            Linear(1024, num_classes),
        )

    def forward(self, x, test_mode=False, mask=None):
        logits={}
        x = self.embeddings(x)
        # [2, 1369, 768] [2, 1370, 768]
        # [2, 784, 768] [2, 785, 768]
        x, xc,cls_token_list = self.encoder(x, test_mode, mask)
        final_hid=torch.cat(cls_token_list,dim=-1)
        struct_outs=self.part_head(final_hid)
        complement_logits = self.head(xc)
        probability_assist = self.softmax(complement_logits)
        weight_assist = list(self.head.parameters())[-1]
        assist_logit = probability_assist * (weight_assist.sum(-1))
        probability_struct=self.softmax(struct_outs)
        weight_struct = list(self.part_head.parameters())[-1]
        assist_struct = probability_struct * (weight_struct.sum(-1))
        # weight = self.head.weight
        # assist_logit = probability * (weight.sum(-1))
        comb_outs = self.head(x) + assist_logit+assist_struct
        logits['last_token']=cls_token_list[3]
        logits['struct_outs']=struct_outs
        logits['comb_outs']=comb_outs
        logits['assist_outs']=complement_logits
        return logits
    def load_from(self,weights):
        with torch.no_grad():
            self.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"],conv=True))
            self.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.encoder.part_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.encoder.part_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))
            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new=self.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.embeddings.position_embeddings.copy_(posemb)
            else:
                ntok_new = posemb_new.size(1)
                posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                ntok_new -= 1
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.embeddings.position_embeddings.copy_(np2th(posemb))
                for bname, block in self.encoder.named_children():
                    for uname, unit in block.named_children():
                        if not bname.startswith('key') and not bname.startswith('clr') and not bname.startswith('part') and not bname.startswith('stru') and  \
                                not uname.startswith('conv') and not uname.startswith('relative') and not uname.startswith('gcn'):
                            # print(uname)
                            unit.load_from(weights, n_block=uname)
class SAPEncoder(nn.Module):
    def __init__(self,config,update_warm,patch_num,total_num,split,coeff_max):
        super(SAPEncoder, self).__init__()
        self.warm_steps = update_warm
        self.patch_num=patch_num
        self.layer = nn.ModuleList()
        self.layer_num=config.num_layers
        # 前11层
        for _ in range(config.num_layers-1):
            layer=Block(config,coeff_max)
            self.layer.append(copy.deepcopy(layer))
        self.clr_layer = Block(config,coeff_max)
        self.key_layer = Block(config,coeff_max)
        # self.stru_atten=Block(config)
        self.part_layer=Block(config,coeff_max)
        # self.part_layer=self.layer[-1]

        self.key_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.part_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.patch_select=MultiHeadSelector(config.hidden_size,self.patch_num)
        self.part_attention=Part_Attention()
        # 总共选择层的大小一般为126，得到后三层每层选择的大小
        self.total_num=total_num
        self.select_num=torch.tensor([16, 14, 12, 10, 8, 6, 8, 10, 12, 14, 16], device='cuda')
        self.select_rate = self.select_num/ torch.sum(self.select_num)
        self.select_num = self.select_rate * self.total_num
        self.clr_encoder = CrossLayerRefinement(config, self.clr_layer)
        self.part_structure = Part_Structure(config.hidden_size)
        self.count = 0

    def forward(self,hidden_states, test_mode=False, mask=None):
        if not test_mode:
            self.count += 1
        B, N, C = hidden_states.shape
        selected_hidden_list=[]
        class_token_list = []
        for i,layer in enumerate(self.layer):
            hidden_states,weights,contribution=layer(hidden_states, mask)
            select_num = torch.round(self.select_num[i]).int()
            select_idx, select_score,hidden_states=self.patch_select(hidden_states,weights,contribution,select_num)
            selected_hidden = hidden_states[torch.arange(B).unsqueeze(1), select_idx]
            selected_hidden_list.append(selected_hidden)
            class_token_list.append(self.part_norm(hidden_states[:, 0]))
            # # 取9,10,11层
            # if i>7:
            #     j=i-8
            #     select_num = torch.round(self.select_num[j]).int()
            #     select_idx, select_score = self.patch_select(weights,contribution,select_num)
            #     # select_idx, select_score=self.patch_select(select_weights,select_num)
            #     # selected_hidden = select_hidden_states[torch.arange(B).unsqueeze(1),select_idx]
            #     selected_hidden = hidden_states[torch.arange(B).unsqueeze(1), select_idx]
            #     selected_hidden_list.append(selected_hidden)
            #     _,attention_map=self.part_attention(weights)
            #     # 此时的cls_token具有结构信息
            #     hidden_states=self.part_structure(hidden_states,attention_map)
            #     # select_hidden_states,select_weights=self.stru_atten(hidden_states)
            #     class_token_list.append(self.part_norm(hidden_states[:,0]))
        cls_token=hidden_states[:, 0].unsqueeze(1)
        clr, weights,contribution= self.clr_encoder(selected_hidden_list, cls_token)
        stru_states, stru_weights,_=self.part_layer(hidden_states)
        # _,attention_map=self.part_attention(stru_weights)
        attention_map=stru_weights[:,:,0,1:]
        stru_states= self.part_structure(stru_states,attention_map,stru_weights)
        # cls_token=stru_states[:,0].unsqueeze(1)
        class_token_list.append(self.part_norm(stru_states)[:,0])
        # select_hidden_states, select_weights = self.stru_atten(part_states)
        # clr,weights=self.clr_encoder(selected_hidden_list,cls_token)
        # clr, weights,contribution= self.clr_encoder(selected_hidden_list, cls_token)
        # clr, weights,contribution= self.clr_encoder(selected_hidden_list, last_token)
        sort_idx,_,_ = self.patch_select(clr,weights,contribution, last=True)
        if not test_mode and self.count >= self.warm_steps:
            layer_count=self.count_patch(sort_idx)
            self.update_layer_select(layer_count)
        out=clr[torch.arange(B).unsqueeze(1),sort_idx]
        out = torch.cat((cls_token, out), dim=1)
        out,_,_ = self.key_layer(out)
        key = self.key_norm(out)
        class_token_list=class_token_list[-4:]
        return key[:, 0], clr[:, 0],class_token_list
    def count_patch(self, sort_idx):
        # layer_count 将输出 [16, 30, 42, 52, 60, 66, 74, 84, 96, 110, 126]
        # 表示上层的选择数加本层的选择数
        layer_count = torch.cumsum(self.select_num, dim=-1)
        num=len(self.select_num)
        sort_idx = (sort_idx - 1).reshape(-1)
        for i in range(num):
            # print("当前每层的mask",mask_idx.sum())
            mask = (sort_idx < layer_count[i])
            layer_count[i] = mask.sum()
        cum_count = torch.cat((torch.tensor([0], device='cuda'), layer_count[:-1]))
        # print("cum_count的值为",cum_count)
        layer_count -= cum_count
        # print("每次减完之后的layer_count:",layer_count)

        return layer_count.int()

    def update_layer_select(self, layer_count):
        alpha = 1e-3  # if self.dataset != 'dog' and self.dataset == 'nabirds' else 1e-4
        new_rate = layer_count / layer_count.sum()

        self.select_rate = self.select_rate * (1 - alpha) + alpha * new_rate
        self.select_rate /= self.select_rate.sum()
        self.select_num = self.select_rate * self.total_num
# 多头选择器以及部分注意特征图
class MultiHeadSelector(nn.Module):
    def __init__(self,hidden_size,patch_num=84):
        super(MultiHeadSelector, self).__init__()
        # 得到每个头的需要的patch数
        self.patch_num = patch_num
        # 创建一个大小为(1,1,3,3)的平滑滤波器
        # self.kernel = torch.tensor([[1, 2, 1],
        #                             [2, 4, 2],
        #                             [1, 2, 1]], device='cuda').unsqueeze(0).unsqueeze(0).half()
        self.conv = nn.Conv2d(1, 1, 3, 1, 1).half()
        self.relative = RelativeCoordPredictor()
        self.gcn =  GCN(12, 512, hidden_size, dropout=0.1)
    # 得到的权重的值为(B,Head,S+1,S+1)
    def forward(self,hidden_states, x,contribution,select_num=None, last=False):
        # 得到B、头数、patch大小
        B,C,S = x.shape[0],x.shape[1],x.shape[3] - 1
        H = math.ceil(math.sqrt(S))
        select_num = self.patch_num if select_num is None else select_num

        count = torch.zeros((B, S), dtype=torch.int, device='cuda').half()
        row_score = x[:, :, 0,:]
        # col_score=contribution[:,:,:]
        score=(row_score*contribution)[:, :, 1:]
        # 得到结构信息
        if not last:
            structure_info, basic_index=self.relative(score)
            position_weight = x[:, :, 1:, 1:]
            position_weight = torch.mean(position_weight, dim=1)
            structure_info = self.gcn(structure_info, position_weight)
            hidden_states_clone = hidden_states.clone()
            for i in range(B):
                index = int(basic_index[i])
                hidden_states_clone[i, 0] = hidden_states_clone[i, 0] + structure_info[i, index, :]
            hidden_states = hidden_states_clone

        # 判别区域选择
        _, select = torch.topk(score, self.patch_num, dim=-1)
        select = select.reshape(B, -1)
        # new_score=torch.sum(score,dim=1)
        for i, b in enumerate(select):
            count[i, :] += torch.bincount(b, minlength=S)
        if not last:
            count = self.enhace_local(count)
            pass
        patch_value, patch_idx = torch.sort(count, dim=-1, descending=True)
        patch_idx += 1
        return patch_idx[:, :select_num], count,hidden_states

    def enhace_local(self, count):
        # 得到B和H
        B, H = count.shape[0], math.ceil(math.sqrt(count.shape[1]))
        # 将count转为B,H,H
        count = count.reshape(B, H, H)
        count = self.conv(count.unsqueeze(1)).reshape(B, -1)
        return count
# 跨层增强模块
class CrossLayerRefinement(nn.Module):
    def __init__(self, config, clr_layer):
        super(CrossLayerRefinement, self).__init__()
        # 跨层增强快
        self.clr_layer = clr_layer
        # 跨层增强快的norm
        self.clr_norm = LayerNorm(config.hidden_size, eps=1e-6)

    def forward(self, x, cls):
        out = torch.cat(x, dim=1)
        # B,total_num+1,hidden_size
        out = torch.cat((cls, out), dim=1)
        # 得到out(B,total_num+1,hidden_size)
        # weights (B,Head,total_num+1,total_num+1)
        out, weights,contribution = self.clr_layer(out)
        out = self.clr_norm(out)
        return out, weights,contribution
# 处理部分结构信息
class Part_Structure(nn.Module):
    def __init__(self,hidden_size):
        super(Part_Structure, self).__init__()
        # 相对坐标信息
        self.relative_coord_predictor = RelativeCoordPredictor()
        # 设置GCN
        self.gcn =  GCN(12, 512, hidden_size, dropout=0.1)

    # 输入hidden_states制计即该层的输出(B, S, head_size)
    # 最大值索引(batch_size, num_attention_heads, 1)
    # 注意力特征图(B,C,H,H)C=注意力头数
    # struc_tokens (B,1,hidden_size)
    def forward(self, hidden_states, attention_map,weight):
        # 注意力特征图的形状信息
        # C=注意力头数
        B,C,S = attention_map.shape
        H=math.ceil(math.sqrt(S))
        # 得到注意力特征图的结构信息，基本锚点，位置权重，最大索引
        # 用于描述注意力图的结构特征和相对坐标
        # 相对坐标总和(N,S,2) S=H*W
        # 基础锚点(N,2)
        # 位置权重 (N,S,S)
        # 具有最大均值值像素的索引(N,)
        position_weight = weight[:, :, 1:, 1:]
        position_weight = torch.mean(position_weight, dim=1)
        structure_info, basic_index = self.relative_coord_predictor(attention_map)
        # structure_info为(N,S,512)
        structure_info = self.gcn(structure_info, position_weight)
        hidden_states_clone=hidden_states.clone()
        for i in range(B):
            index = int(basic_index[i])
            hidden_states_clone[i, 0] = hidden_states_clone[i, 0] + structure_info[i, index, :]
        hidden_states=hidden_states_clone
        return hidden_states
class RelativeCoordPredictor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # 得到图片的 N(Batch_Size),C,H,W
        # H=28
        N, C, S = x.shape
        structure_info = x.view(N, C, S).transpose(1, 2).contiguous()  # (N, S, C)
        # torch.mean(masked_x, dim=-1) 将C维平均得到 (N, H*W)
        # 再对其进行取最大值,得到具有最大值像素的索引即reduced_x_max_index
        # reduced_x_max_index形状为(N)
        _, reduced_x_max_index = torch.max(torch.mean(structure_info, dim=-1), dim=-1)

        return structure_info, reduced_x_max_index
    # 得到每个patch的坐标信息
    def build_basic_label(self, size):
        basic_label = np.array([[(i, j) for j in range(size)] for i in range(size)])
        return basic_label
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
# 用于计算部分注意力
class Part_Attention(nn.Module):
    def __init__(self):
        super(Part_Attention, self).__init__()

    # 输入对应的注意力权重张量 (batch_size, num_attention_heads, S, S)数组
    # 即第9层以后该Transformer的每层的注意力权重张量数组
    def forward(self, x):
        last_map = x[:,:,0,1:]
        # 得到B,C
        B,C = last_map.size(0),last_map.size(1)
        # 得到patch数
        patch_num = last_map.size(-1)
        # 根据patch数得到高
        H = patch_num ** 0.5
        H = int(H)
        # C=注意力头数
        # H=28
        attention_map = last_map.view(B,C,H,H)
        # last_map(batch_size, num_attention_heads, S-1)
        # 最大值索引(batch_size, num_attention_heads)
        # 最大值(batch_size, num_attention_heads)
        # 注意力特征图(batch_size,num_attention_heads,H,H)
        return last_map, attention_map
if __name__ == '__main__':
    start = time.time()
    config = get_b16_config()
    # com = clrEncoder(config,)
    # com.to(device='cuda')
    net = SPTransformer(config,200,448,500,84,129,split='non-overlap').cuda()
    # hidden_state = torch.arange(400*768).reshape(2,200,768)/1.0
    x = torch.rand(2, 3, 448, 448, device='cuda')
    # print(x[:, :, 0, :].shape)
    # y=torch.max(x[:, :, 0, :], dim=2, keepdim=False)[0]
    # print(y.shape)
    y = net(x)
    # print(y.keys())
    # for name, param in net.state_dict().items():
    #     print(name)
    # pretrained_weights = np.load('ViT-B_16.npz')
    # net.load_from(pretrained_weights)

    # for name, param in pretrained_weights.items():
    #     print(name)