import torch
import torch.nn as nn
import torch.nn.functional as F
# torch.autograd.set_detect_anomaly(True)
def con_loss_new(features, labels):
    eps = 1e-6
    # 获得样本数
    # B,200
    B= features.shape[0]
    # 规范化
    features = F.normalize(features)
    # 样本间的余弦相似性(B,B)
    cos_matrix = features.mm(features.t())
    # 创建一个矩阵，其中对角线上的元素表示同一类别的样本，其他元素为零,形状为(B,B)
    pos_label_matrix = torch.stack([labels == labels[i] for i in range(B)]).float()
    # 负样本矩阵，对角线为0其他为1
    neg_label_matrix = 1 - pos_label_matrix
    neg_label_matrix_new = 1 - pos_label_matrix
    # 正样本间的余弦相似度，余弦相似度的取值范围是 [-1, 1]，其中 1 表示完全相似，0 表示不相关，-1 表示完全不相似
    # 这样得到的相似度的取值范围变为 [0, 2]，其中 0 表示完全相似，2 表示完全不相似
    pos_cos_matrix = 1 - cos_matrix
    # 通过将余弦相似度加上 1，得到了负样本之间的相似度矩阵。
    # 这样得到的相似度的取值范围也是 [0, 2]，其中 0 表示完全相似，2 表示完全不相似
    neg_cos_matrix = 1 + cos_matrix
    # 定义一个间隔（margin）值，它用于区分正样本和负样本的相似性
    margin = 0.3
    # 计算相似性分数，将余弦相似度映射到区间 [0, 1]
    sim = (1 + cos_matrix) / 2.0
    # 计算相似性分数的反向，用于度量距离
    scores = 1 - sim
    #计算正样本之间的相似性分数，保留正样本的分数，而负样本的分数被设为零
    positive_scores = torch.where(pos_label_matrix == 1.0, scores, scores - scores)
    mask = torch.eye(features.size(0)).cuda()
    positive_scores = torch.where(mask == 1.0, positive_scores - positive_scores, positive_scores)
    positive_scores = torch.sum(positive_scores, dim=1, keepdim=True) / (
                (torch.sum(pos_label_matrix, dim=1, keepdim=True) - 1) + eps)
    positive_scores = torch.repeat_interleave(positive_scores, B, dim=1)
    relative_dis1 = margin + positive_scores - scores
    neg_label_matrix_new[relative_dis1 < 0] = 0
    neg_label_matrix = neg_label_matrix * neg_label_matrix_new

    loss = (pos_cos_matrix * pos_label_matrix).sum() + (neg_cos_matrix * neg_label_matrix).sum()
    loss /= B * B

    return loss