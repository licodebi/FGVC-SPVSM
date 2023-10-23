import torch
import torch.nn as nn
from torch.autograd import Function
from numpy import linalg as LA
import numpy as np
# 稀疏逆协方差估计
class SICE(nn.Module):
     def __init__(self, iterNum=3, is_sqrt=True, is_vec=True, input_dim=2048, dimension_reduction=None, sparsity_val=0.0, sice_lrate=0.0):

         super(SICE, self).__init__()
         # 一个epoch中的迭代次数
         self.iterNum=iterNum
         #是否应用平方根操作
         self.is_sqrt = is_sqrt
         # 是否是向量形式
         self.is_vec = is_vec
         #降维的大小
         self.dr = dimension_reduction
         # 稀疏值
         self.sparsity = sparsity_val
         # 学习率
         self.learingRate = sice_lrate
         # 如果将为大小不为空
         # 将其进行降维
         if self.dr is not None:
             self.conv_dr_block = nn.Sequential(
               nn.Conv1d(input_dim, self.dr, kernel_size=1, stride=1, bias=False),
               nn.BatchNorm1d(self.dr),
               nn.ReLU(inplace=True)
             )

         output_dim = self.dr if self.dr else input_dim
         #如果是向量
         # 具体计算式output_dim*(output_dim+1)/2是用来计算向量的上三角矩阵的元素数量
         if self.is_vec:
             self.output_dim = int(output_dim*(output_dim+1)/2)
         # 不是向量
         else:
             self.output_dim = int(output_dim*output_dim)
         self._init_weight()

     def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

     def _cov_pool(self, x):
         return Covpool.apply(x)
     

     def _inv_sqrtm(self, x, iterN):
         return Sqrtm.apply(x, iterN)

     def _sqrtm(self, x, iterN):
         batchSize = x.shape[0]
         dim = x.shape[1]
         dtype = x.dtype
         I3 = 3.0 * torch.eye(dim, dim, device=x.device).view(1, dim, dim).repeat(batchSize, 1, 1).type(dtype)
         normA = (1.0 / 3.0) * x.mul(I3).sum(dim=1).sum(dim=1)
         A = x.div(normA.view(batchSize, 1, 1).expand_as(x))
         ZY = 0.5 * (I3 - A)
         if iterN < 2:
             ZY = 0.5*(I3 - A)
             YZY = A.bmm(ZY)
         else:
             Y = A.bmm(ZY)
             Z = ZY
             for _ in range(iterN - 2):
                 ZY = 0.5 * (I3 - Z.bmm(Y))
                 Y = Y.bmm(ZY)
                 Z = ZY.bmm(Z)
             YZY = 0.5 * Y.bmm(I3 - Z.bmm(Y))
         y = ZY * torch.sqrt(normA).view(batchSize, 1, 1).expand_as(x)
         return y
     # mfx是输入的特征向量数据
     # fLR是学习率
     # fSparsity是稀疏度
     # nSteps是迭代次数
     def _sice(self, mfX, fLR=5.0, fSparsity=0.07, nSteps=10000):
         # 得到特征之间的协方差矩阵
         mfC = self._cov_pool(mfX)
         # torch.diagonal(mfC, dim1=-2, dim2=-1).sum(-1).view(-1,1,1)为从每个batch的矩阵中提取对角线之和
         # 得到对应的(batch_size,1,1)大小的对角线之和的张量
         # 将mfc除以其对角线之和得到归一化协方差矩阵
         # torch.diagonal(mfC, dim1=-2, dim2=-1)得到mfC的对角线(B,dim)
         # 对对角线进行求和(B,)，并进行张量变换得到，(B,1,1)
         # mfC再对其进行除法操作得到对应的归一化协方差矩阵
         mfC=mfC/torch.diagonal(mfC, dim1=-2, dim2=-1).sum(-1).view(-1,1,1)
         #torch.rand(mfC.shape[1],device = mfC.device)生成一个形状为(mfC.shape[1],mfC.shape[1])的随机张量
         # torch.diag(torch.rand(mfC.shape[1],device = mfC.device))则是将随机张量作为对角线生成对角线矩阵
         # view(1, mfC.shape[1], mfC.shape[2])对角线张量的形状调整为 (1, mfC.shape[1], mfC.shape[2])
         # 最后得到的I是与mfC相同大小的对角矩阵
         # 这种正则化矩阵通常用于通过添加一个很小的正数
         # 来防止协方差矩阵的过度奇异性，并改善计算的稳定性
         # 确保在计算逆矩阵时避免除以零的错误
         I = 1e-10+1e-9*torch.diag(torch.rand(mfC.shape[1],device = mfC.device))\
             .view(1, mfC.shape[1], mfC.shape[2]).repeat(mfC.shape[0],1,1).type(mfC.dtype)
         #zz为mfC+I协方差矩阵的逆平方根矩阵
         zz=self._inv_sqrtm(mfC+I, 7)
         # 进行了逆平方根矩阵与自身的乘积运算，得到新的矩阵
         mfInvC=zz.bmm(zz)

         mfCov=mfC*1.0
         mfLLT=mfInvC*1.0 #+1
         # mfCov=mfCov
         # mfLLT=mfLLT
         # 创建一个和mfLLT相同大小且每个元素均为1e10的张量
         mfLLT_prev=1e10*torch.ones(mfLLT.size(), device=mfC.device)
         
         nCounter=0
         for i in range(nSteps):
             # 这两行代码分别计算了 mfLLT 中的正值和负值
             # mfLLT_plus 中的所有负值都被置为零
             # mfLLT_minus 中的所有正值都被置为零
             mfLLT_plus = torch.relu(mfLLT)
             mfLLT_minus = torch.relu(-mfLLT)
             # 计算 (mfLLT + I) 的逆平方根矩阵
             zz = self._inv_sqrtm(mfLLT+I, 7)
             #
             mfGradPart1=-zz.bmm(zz)
             #计算了 mfCov 矩阵的转置和自身相加的结果的一半，用于计算 mfCov 的一部分梯度
             mfGradPart2 = 0.5*(mfCov.transpose(1,2) + mfCov)
             # 这一行代码将 mfGradPart1 和 mfGradPart2 进行相加，得到组合的梯度
             mfGradPart12 = mfGradPart1+mfGradPart2
             # 这两行代码分别将 fSparsity 加到 mfGradPart12 和减去 mfGradPart12，
             # 得到两个不同方向的梯度
             mfGradPart3_plus = mfGradPart12 + fSparsity
             mfGradPart3_minus = -mfGradPart12 + fSparsity
             #这一行代码计算了一个衰减因子 fDec，其值随着迭代次数的增加而降低
             fDec=(1-i/(nSteps-1.0) )
             # 对 mfLLT_plus 和 mfLLT_minus 进行相应的更新，采用了梯度下降的方式进行更新
             mfLLT_plus = mfLLT_plus - fLR*fDec*mfGradPart3_plus
             mfLLT_minus = mfLLT_minus - fLR*fDec*mfGradPart3_minus
             # 这两行代码再次将更新后的 mfLLT_plus 和 mfLLT_minus 中的负值转换为零
             mfLLT_plus = torch.relu(mfLLT_plus)
             mfLLT_minus = torch.relu(mfLLT_minus)
             # 这一行代码将更新后的 mfLLT_plus 和 mfLLT_minus 相减，得到最终的 mfLLT
             mfLLT = mfLLT_plus-mfLLT_minus
             # 这一行代码将 mfLLT 和其转置相加后除以2，以确保它是对称的
             mfLLT = 0.5*(mfLLT+mfLLT.transpose(1,2))
             # 这一行代码计算了当前 mfLLT 和上一次迭代的 mfLLT_prev 之间的绝对差异的平均值。
             # 这用于判断算法是否收敛
             fSolDiff = (mfLLT-mfLLT_prev).abs().mean()
             # 这一行代码计算了 mfLLT 中大于2e-8的绝对值的元素数量的平均值
             # 用于评估 mfLLT 的稀疏程度
             fSparseCount = ((mfLLT.abs()>2e-8)*1.0).mean()

             mfLLT_prev = mfLLT*1.0
             mfLLT_prev = mfLLT_prev
         mfOut = mfLLT
         # 这一行代码对 mfOut 进行归一化操作，将其除以输入数据 mfX协方差矩阵的对角线元素的和
         mfOut = mfOut/torch.diagonal(self._cov_pool(mfX), dim1=-2, dim2=-1).sum(-1).view(-1,1,1) 
         return mfOut
    
     def _triuvec(self, x):
         return Triuvec.apply(x)     
     

     def forward(self, x):
         if self.dr is not None:
             x = self.conv_dr_block(x)
         x = self._sice(x, fLR=self.learingRate, fSparsity=self.sparsity, nSteps=self.iterNum)
         if self.is_vec:
             x = self._triuvec(x)
         return x



class Covpool(Function):
     @staticmethod
     def forward(ctx, input):
         x = input
         # 得到batch_size
         batchSize = x.data.shape[0]
         # dim为通道数
         dim = x.data.shape[1]
         M = x.data.shape[-1]
         x = x.reshape(batchSize,dim,M)
         #生成一个MxM的矩阵,对角线上元素为-1/M^2+1/M其余元素为-1/M^2
         I_hat = (-1./M/M)*torch.ones(M,M,device = x.device) + (1./M)*torch.eye(M,M,device = x.device)
         # 然后更改形状变为(batchSize,M,M)
         I_hat = I_hat.view(1,M,M).repeat(batchSize,1,1).type(x.dtype)
         # x先与I_hat进行矩阵乘法x的最后一个维度和I_hat的第二个维度相等得到(batchSize,dim,M)
         # 再和x的转置进行矩阵乘法,x的转置形状为(batchSize,M,dim)
         # 相乘后得到的y为(batchSize,dim,dim)
         # y表示样本内各个特征之间的协方差，可进行特征相关性分析，特征选择等
         y = x.bmm(I_hat).bmm(x.transpose(1,2))
         # 将x与I_hat保存进前向传播当中
         ctx.save_for_backward(input,I_hat)
         return y
     # 根据前向传播时保存的张量和梯度输出,计算出相应的梯度张量并返回。
     @staticmethod
     def backward(ctx, grad_output):
         # 获取前向传播过程中的input和I_hat
         input,I_hat = ctx.saved_tensors
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         M = x.data.shape[-1]
         x = x.reshape(batchSize,dim,M)
         # 对 grad_output 进行转置，并与原始 grad_output 相加，从而得到一个临时梯度张量
         grad_input = grad_output + grad_output.transpose(1,2)
         # 利用临时梯度张量、输入张量 x 和矩阵 I_hat 进行批次矩阵乘法操作
         # grad_input的维度为(B,dim,dim) x为(B,dim,M)
         # 计算后得到grad_input的维度为(B,dim,M)
         grad_input = grad_input.bmm(x).bmm(I_hat)
         # 重塑为形状为 (batchSize, dim, h, w) 的张量
         grad_input = grad_input.reshape(batchSize,dim,M)
         return grad_input

class Sqrtm(Function):
     @staticmethod
     def forward(ctx, input, iterN):
         # input形状为(B,dim,dim)的协方差矩阵
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         dtype = x.dtype
         # I3为一个3倍的单位矩阵，大小为(B,dim,dim)
         I3 = 3.0*torch.eye(dim,dim,device = x.device).view(1, dim, dim).repeat(batchSize,1,1).type(dtype)
         # normA为(B,)即归一化因子
         normA = (1.0/3.0)*x.mul(I3).sum(dim=1).sum(dim=1)
         # A为x除以normA得到的形状为(B,dim,dim)的归一化矩阵
         A = x.div(normA.view(batchSize,1,1).expand_as(x))
         # Y是一个大小为 (batchSize, iterN, dim, dim) 的零矩阵，用于保存中间计算结果
         Y = torch.zeros(batchSize, iterN, dim, dim, requires_grad = False, device = x.device).type(dtype)
         # 是一个大小为 (batchSize, iterN, dim, dim) 的单位矩阵，用于保存中间计算结果
         Z = torch.eye(dim,dim,device = x.device).view(1,dim,dim).repeat(batchSize,iterN,1,1).type(dtype)
         # 如果 iterN 小于 2，直接计算 ZY 和 YZY 的值
         if iterN < 2:
            ZY = 0.5*(I3 - A)
            YZY = A.bmm(ZY)
         else:
            ZY = 0.5*(I3 - A)
            Y[:,0,:,:] = A.bmm(ZY)
            Z[:,0,:,:] = ZY
            for i in range(1, iterN-1):
               ZY = 0.5*(I3 - Z[:,i-1,:,:].bmm(Y[:,i-1,:,:]))
               Y[:,i,:,:] = Y[:,i-1,:,:].bmm(ZY)
               Z[:,i,:,:] = ZY.bmm(Z[:,i-1,:,:])
            ZYZ = 0.5 * (I3 - Z[:,iterN-2,:,:].bmm(Y[:,iterN-2,:,:])).bmm(Z[:,iterN-2,:,:])
         # 得到协方差的逆平方根矩阵形状为(B,dim,dim)
         y = ZYZ * torch.pow(normA,-0.5).view(batchSize, 1, 1).expand_as(x)
         ctx.save_for_backward(input, A, ZYZ, normA, Y, Z)
         ctx.iterN = iterN
         return y
     @staticmethod
     def backward(ctx, grad_output):
         input, A, ZY, normA, Y, Z = ctx.saved_tensors
         iterN = ctx.iterN
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         dtype = x.dtype
         der_postCom = grad_output*torch.pow(normA, -0.5).view(batchSize, 1, 1).expand_as(x)
         der_postComAux = -0.5*torch.pow(normA, -1.5)*((grad_output*ZY).sum(dim=1).sum(dim=1))
         I3 = 3.0*torch.eye(dim,dim,device = x.device).view(1, dim, dim).repeat(batchSize,1,1).type(dtype)
         if iterN < 2:
            der_NSiter = 0.5*(der_postCom.bmm(I3 - A) - A.bmm(der_postCom))
         else:
            dldZ = 0.5*((I3 - Y[:,iterN-2,:,:].bmm(Z[:,iterN-2,:,:])).bmm(der_postCom) -
                          der_postCom.bmm(Z[:,iterN-2,:,:]).bmm(Y[:,iterN-2,:,:]))
            dldY = -0.5*Z[:,iterN-2,:,:].bmm(der_postCom).bmm(Z[:,iterN-2,:,:])
            for i in range(iterN-3, -1, -1):
               YZ = I3 - Y[:,i,:,:].bmm(Z[:,i,:,:])
               ZY = Z[:,i,:,:].bmm(Y[:,i,:,:])
               dldY_ = 0.5*(dldY.bmm(YZ) -
                         Z[:,i,:,:].bmm(dldZ).bmm(Z[:,i,:,:]) -
                             ZY.bmm(dldY))
               dldZ_ = 0.5*(YZ.bmm(dldZ) -
                         Y[:,i,:,:].bmm(dldY).bmm(Y[:,i,:,:]) -
                            dldZ.bmm(ZY))
               dldY = dldY_
               dldZ = dldZ_
            der_NSiter = 0.5*(dldY.bmm(I3 - A) - dldZ - A.bmm(dldY))
         der_NSiter = der_NSiter.transpose(1, 2)
         grad_input = der_NSiter.div(normA.view(batchSize,1,1).expand_as(x))
         grad_aux = der_NSiter.mul(x).sum(dim=1).sum(dim=1)
         for i in range(batchSize):
             grad_input[i,:,:] += (der_postComAux[i] \
                                   - grad_aux[i] / (normA[i] * normA[i])) \
                                   *torch.ones(dim,device = x.device).diag().type(dtype)
         return grad_input, None

class Triuvec(Function):
     @staticmethod
     def forward(ctx, input):
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         dtype = x.dtype
         x = x.reshape(batchSize, dim*dim)
         I = torch.ones(dim,dim).triu().reshape(dim*dim)
         index = I.nonzero()
         y = torch.zeros(batchSize,int(dim*(dim+1)/2),device = x.device).type(dtype)
         y = x[:,index]
         ctx.save_for_backward(input,index)
         return y
     @staticmethod
     def backward(ctx, grad_output):
         input,index = ctx.saved_tensors
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         dtype = x.dtype
         grad_input = torch.zeros(batchSize,dim*dim,device = x.device,requires_grad=False).type(dtype)
         grad_input[:,index] = grad_output
         grad_input = grad_input.reshape(batchSize,dim,dim)
         return grad_input

def CovpoolLayer(var):
    return Covpool.apply(var)

def SqrtmLayer(var, iterN):
    return Sqrtm.apply(var, iterN)

# def InvcovpoolLayer(var):
#     return InverseCOV.apply(var)

def TriuvecLayer(var):
    return Triuvec.apply(var)
