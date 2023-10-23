import torch
from typing import Union
from torchvision.models.feature_extraction import get_graph_node_names

# from .pim_module import pim_module
from .pim_module import siandpi_module
from models import vit
from models.SAPIV import SPTransformer


"""
[Default Return]
Set return_nodes to None, you can use default return type, all of the model in this script 
return four layers features.

[Model Configuration]
if you are not using FPN module but using Selector and Combiner, you need to give Combiner a 
projection  dimension ('proj_size' of GCNCombiner in pim_module.py), because graph convolution
layer need the input features dimension be the same.

[Combiner]
You must use selector so you can use combiner.

[About Costom Model]
This function is to building swin transformer. timm swin-transformer + torch.fx.proxy.Proxy 
could cause error, so we set return_nodes to None and change swin-transformer model script to
return features directly.
Please check 'timm/models/swin_transformer.py' line 541 to see how to change model if your costom
model also fail at create_feature_extractor or get_graph_node_names step.
"""


def load_model_weights_vit(model, model_path):
    ### reference https://github.com/TACJu/TransFG
    ### thanks a lot.
    state = torch.load(model_path, map_location='cpu')
    for key in model.state_dict():
        if 'num_batches_tracked' in key:
            continue
        p = model.state_dict()[key]
        if key in state.keys():
            ip = state[key]
            if p.shape == ip.shape:
                p.data.copy_(ip.data)  # Copy the data of parameters
            else:
                print('could not load layer: {}, mismatch shape {} ,{}'.format(key, (p.shape), (ip.shape)))
        else:
            print('could not load layer: {}, not in checkpoint'.format(key))
    return model


def build_vit16(pretrained: str = "models/vit_base_patch16_224_in21k_miil.pth",
                return_nodes: Union[dict, None] = None,
                num_selects: Union[dict, None] = None, 
                img_size: int = 448,
                use_fpn: bool = False,
                fpn_size: int = 512,
                proj_type: str = "Linear",
                upsample_type: str = "Conv",
                use_selection: bool = True,
                num_classes: int = 200,
                use_combiner: bool = True,
                comb_proj_size: Union[int, None] = None):

    import timm
    # 得到主干网络
    backbone = timm.create_model('vit_base_patch16_224_miil_in21k', pretrained=pretrained)
    ### original pretrained path "./models/vit_base_patch16_224_miil_21k.pth"
    # 如果提供预训练权重的路径，则使用load_model_weights加载权重
    if pretrained != "":
        backbone = load_model_weights_vit(backbone, pretrained)

    backbone.train()

    # print(backbone)
    # print(get_graph_node_names(backbone))
    # 0~11 under blocks

    if return_nodes is None:
        return_nodes = {
            'blocks.9': 'layer1',
            'blocks.10': 'layer2',
            'blocks.11': 'layer3',
        }
    if num_selects is None:
        num_selects = {
            'layer1':32,
            'layer2':32,
            'layer3':32
        }
    # 导入数学库和图像处理库
    import math
    from scipy import ndimage
    # 从主干模型中获取位置嵌入
    # posemb_tok得到每个图片中cls的位置嵌入 即(B,1,embed_size)
    # posemb_grid得到图片的位置嵌入，输入图片大小一致，故均相同 (H*W,embed_size)
    posemb_tok, posemb_grid = backbone.pos_embed[:, :1], backbone.pos_embed[0, 1:]
    # 将posemb_grid转为numpy数组
    posemb_grid = posemb_grid.detach().numpy()
    # 计算posemb_grid的尺寸，即图片的patch数,即H或W
    gs_old = int(math.sqrt(len(posemb_grid)))
    #根据现有的图片大小除以patch得到新的网格尺寸
    gs_new = img_size//16
    # 将H*W拆分为(H,W,-1)
    posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
    # 计算缩放因子，用于将网格缩放到新的尺寸
    zoom = (gs_new / gs_old, gs_new / gs_old, 1)
    # 使用 ndimage.zoom 函数按指定的缩放因子对网格进行缩放
    posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
    # 重新调整 posemb_grid 的形状（1,H*W,embed_size）
    posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
    # 转换为 PyTorch 张量
    posemb_grid = torch.from_numpy(posemb_grid)
    # 将 posemb_tok 和 posemb_grid 沿指定维度进行拼接，得到完整的位置嵌入
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    # 将位置嵌入作为模型的参数，并赋值给 backbone.pos_embed
    backbone.pos_embed = torch.nn.Parameter(posemb)

    return siandpi_module.PluginMoodel(backbone = backbone,
                                   return_nodes = return_nodes,
                                   img_size = img_size,
                                   use_fpn = use_fpn,
                                   use_vit_fpn=True,
                                   use_struct=True,
                                   fpn_size = fpn_size,
                                   proj_type = proj_type,
                                   upsample_type = upsample_type,
                                   use_selection = use_selection,
                                   num_classes = num_classes,
                                   num_selects = num_selects, 
                                   use_combiner = num_selects,
                                   comb_proj_size = comb_proj_size)

def build_vitb16(num_classes,num_selects,img_size,update_warm,patch_num,split='non-overlap',coeff_max=0.25):
    # 加载vit配置
    config=backbone['ViT-B_16']
    model = SPTransformer(config=config,
                          num_classes=num_classes,
                          img_size=img_size,
                          update_warm=update_warm,
                          patch_num=patch_num,
                          total_num=num_selects,
                          split=split,
                          coeff_max=coeff_max
                          )
    return model


backbone = {
	'ViT-B_16': vit.get_b16_config(),
	'ViT-B_32': vit.get_b32_config(),
	'ViT-L_16': vit.get_l16_config(),
	'ViT-L_32': vit.get_l32_config(),
	'ViT-H_14': vit.get_h14_config(),
	'testing': vit.get_testing(),
}
if __name__ == "__main__":
   a=1

MODEL_GETTER = {
    "vit":build_vit16,
    "vit-b_16":build_vitb16
}
