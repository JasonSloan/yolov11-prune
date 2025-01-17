import contextlib
from copy import deepcopy

import torch
import torch.nn as nn

from ultralytics.nn.tasks import BaseModel
from ultralytics.nn.modules.conv import Conv, Concat
from ultralytics.nn.modules.head_pruned import DetectPruned
from ultralytics.nn.modules.block_pruned import C3k2Pruned, SPPFPruned, C2PSAPruned

from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.loss import v8DetectionLoss, E2EDetectLoss
from ultralytics.utils.torch_utils import initialize_weights,scale_img
from ultralytics.utils.ops import make_divisible


class DetectionModelPruned(BaseModel):
    """
        forward继承BaseModel
    """
    def __init__(self, maskbndict, cfg, ch=3, nc=None, verbose=True):  # model, input channels, number of classes
        """Initialize the YOLOv8 detection model with the given config and parameters."""
        super().__init__()
        self.yaml = cfg
        # 注意这里一定要deepcopy一下cfg, 因为当模型剪枝过程全部结束时我们要把模型保存下来, 包括模型配置信息
        # 所以我们并不想改变模型配置信息, 如果改变, 当再次用配置信息去构建网络的时候就会出错
        self.model, self.save, self.current_to_prev = parse_model_pruned(maskbndict, deepcopy(cfg), ch)
        self.names = {i: f'{i}' for i in range(self.yaml['nc'])}  # default names dict
        self.inplace = self.yaml.get('inplace', True)

        # Build strides
        m = self.model[-1]  # Detect()
        if isinstance(m, (DetectPruned)):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            forward = lambda x: self.forward(x)
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # forward
            self.stride = m.stride
            m.bias_init()  # only run once
        else:
            self.stride = torch.Tensor([32])  # default stride for i.e. RTDETR

        # Init weights, biases
        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info('')
            
    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference and train outputs."""
        if getattr(self, "end2end", False) or self.__class__.__name__ != "DetectionModel":
            LOGGER.warning("WARNING ⚠️ Model does not support 'augment=True', reverting to single-scale prediction.")
            return self._predict_once(x)
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = super().predict(xi)[0]  # forward
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, -1), None  # augmented inference, train

    @staticmethod
    def _descale_pred(p, flips, scale, img_size, dim=1):
        """De-scale predictions following augmented inference (inverse operation)."""
        p[:, :4] /= scale  # de-scale
        x, y, wh, cls = p.split((1, 1, 2, p.shape[dim] - 4), dim)
        if flips == 2:
            y = img_size[0] - y  # de-flip ud
        elif flips == 3:
            x = img_size[1] - x  # de-flip lr
        return torch.cat((x, y, wh, cls), dim)

    def _clip_augmented(self, y):
        """Clip YOLO augmented inference tails."""
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4**x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[-1] // g) * sum(4**x for x in range(e))  # indices
        y[0] = y[0][..., :-i]  # large
        i = (y[-1].shape[-1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][..., i:]  # small
        return y

    def init_criterion(self):
        """Initialize the loss criterion for the DetectionModel."""
        return E2EDetectLoss(self) if getattr(self, "end2end", False) else v8DetectionLoss(self)
    
def parse_model_pruned(maskbndict, d, ch, verbose=True):  # model_dict, input_channels(3)
    """    
    网络构建(ch是个列表, 记录着每一层的输出通道数; current_to_prev是一个字典, 记录着{某一bn层的名字: 该bn层连接的上一(多)bn层的名字}:
        这里要重写C2f模块、SPPF模块和Detect模块:
            为什么要重写C2f是因为C2f中的Bottleneck存在残差连接; 
            为什么要重写SPPF和Detect是因为如果不重些SPPF和Detect, 那么SPPF和Detect的内部的结构将不被剪枝
            
        对于Conv结构: 输入通道数来自ch[f], 输出通道数通过取出maskbndict中的bn层mask掩码矩阵计算得到
        
        对于C2f结构(使用netron查看onnx更清晰): 
        需要计算这几个参数: cv1in, cv1out, cv1_split_sections, inner_cv1outs, inner_cv2outs, cv2out, bottle_args
            其中model.{}.cv1的输入通道数cv1in来自ch[f], 输出通道数cv1out通过maskbndict中cv1层的mask掩码矩阵计算得到;
            如果是由残差连接的C2f:
                model.{}.m.0.cv1的输入通道数inner_cv1in等于cv1out/2, 输出通道数inner_cv1out通过maskbndict中对应的mask掩码矩阵计算得到;
                model.{}.m.0.cv2的输入通道数inner_cv2in等于inner_cv1outs[0], 输出通道数inner_cv2out等于inner_cv1in;
                model.{}.cv2的输入通道数cv2in等于(2 + n)*(cv1out/2), 输出通道数cv2out通过maskbndict中对应的mask掩码矩阵计算得到;
            如果是没有残差连接的C2f:
                model.{}.m.0.cv1的输入通道数inner_cv1in等于cv1_split_sections[1], 
                输出通道数inner_cv1out通过maskbndict中对应的mask掩码矩阵计算得到;
                model.{}.m.0.cv2的输入通道数inner_cv2in等于inner_cv1outs[0], 
                输出通道数inner_cv2out通过maskbndict中对应的mask掩码矩阵计算得到;
                model.{}.cv2的输入通道数cv2in等于(cv1_split_sections[0]+所有的inner_cv2out的通道数之和), 
                输出通道数cv2out通过maskbndict中对应的mask掩码矩阵计算得到;
            更详细的以代码为准!
                
        对于Detect结构:
            比较复杂, 看代码吧!
            三张特征图(80*80, 40*40, 20*20), 每张特征图是解耦的两个分支(回归分支, 分类分支), 
            每个分支有三个卷积层, 其中前两个卷积层是带BN的, 最后一个不带BN
    """
    import ast

    # Args
    max_channels = float('inf')
    nc, act, scales = (d.get(x) for x in ('nc', 'activation', 'scales'))
    #  宽度系数没有用上, 因为每一层的输出通道数都是由mask来定的
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ('depth_multiple', 'width_multiple', 'kpt_shape'))
    if scales:
        scale = d.get('scale')
        if not scale:
            scale = tuple(scales.keys())[0]
            LOGGER.warning(f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale]
    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")  # print
    
    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<50}{'arguments':<30}")
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    # 按照{某一bn层的名字: 该bn层连接的上一(多)bn层的名字}记录
    # 因为某一层的输出通道哪些保留哪些不保留是由mask来定的, 
    # 某一层的输入通道哪些保留哪些不保留是由上一层的mask来定的
    current_to_prev = {}
    # 按照{索引: 本层最后一个bn层的名字}记录
    idx_to_bn_layer_name = {}
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)

        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
        
        base_name = f'model.{i}'
        if m in [Conv]:
            # Conv层的f只有-1
            c1 = ch[f]
            bn_layer_name = base_name + '.bn'
            mask = maskbndict[bn_layer_name]
            c2 = torch.sum(mask).int().item()
            args[0] = c2
            args.insert(0, c1)
            # =============================================
            if i == 0:
                prev_bn_layer_name = bn_layer_name
            else:
                current_to_prev[bn_layer_name] = prev_bn_layer_name
                prev_bn_layer_name = bn_layer_name
            idx_to_bn_layer_name[i] = bn_layer_name
            # =============================================
        elif m in [C3k2Pruned]:
            # 这里比较复杂, 但是只要搞清楚每一个Conv的in_channels和out_channels怎么计算就清楚了
            # 这里将C3k2看成一个复杂版本的C2f, 外壳是C2f, 内部可能是C3k也可能只是简单的BottleNeck
            c2f_cv1in = ch[f]
            c2f_cv1_bn_layer_name = base_name + '.cv1.bn'
            c2f_cv1_mask = maskbndict[c2f_cv1_bn_layer_name]
            c2f_cv1out = torch.sum(c2f_cv1_mask).int().item()
            # 在v8版本的head部分的C2f层中, 由于没有shortcut残差结构, 因此C2f结构中的第一个cv1层是可以被剪枝的
            # 但是剪完以后是不一定对称的, 因此要重新计算比例
            # 例如, C2f结构中的第一个cv1层剪枝前输出通道数为256, chunck以后左右各式128,
            # 剪枝后, cv1层输出通道数为120, 但是可能其中80落在左半区, 40落在右半区
            # 在v11版本中的head的C3k2中均有残差结构, 所以这里为了和yolov8-prune保持一致, 也这么写一下
            c2f_cv1_split_sections = [torch.sum(c2f_cv1_mask.chunk(2, 0)[0]).int().item(), torch.sum(c2f_cv1_mask.chunk(2, 0)[1]).int().item()]
            c2f_cv2_bn_layer_name = base_name + '.cv2.bn'
            c2f_cv2_mask = maskbndict[c2f_cv2_bn_layer_name]
            c2f_cv2out = torch.sum(c2f_cv2_mask).int().item()
            
            use_c3k = args[1]
            if not use_c3k:
                bottle_inner_cv1_bn_layer_names = [base_name + f'.m.{i}.cv1.bn' for i in range(n)]
                bottle_inner_cv2_bn_layer_names = [base_name + f'.m.{i}.cv2.bn' for i in range(n)]
                bottle_inner_cv1_masks = [maskbndict[bottle_inner_cv1_bn_layer_name] for bottle_inner_cv1_bn_layer_name in bottle_inner_cv1_bn_layer_names]
                bottle_inner_cv2_masks = [maskbndict[bottle_inner_cv2_bn_layer_name] for bottle_inner_cv2_bn_layer_name in bottle_inner_cv2_bn_layer_names]
                bottle_inner_cv1outs = [torch.sum(bottle_inner_cv1_mask).int().item() for bottle_inner_cv1_mask in bottle_inner_cv1_masks]
                bottle_inner_cv2outs = [torch.sum(bottle_inner_cv2_mask).int().item() for bottle_inner_cv2_mask in bottle_inner_cv2_masks]
                c3k_cv1outs = c3k_cv2outs = c3k_inner_cv1outs = c3k_cv3outs = None # 占位用
            else:
                bottle_inner_cv1outs = bottle_inner_cv2outs = None # 占位用
                c3k_cv1_bn_layer_names = [base_name + f'.m.{i}.cv1.bn' for i in range(n)]
                c3k_cv1_bn_masks = [maskbndict[c3k_cv1_bn_layer_name] for c3k_cv1_bn_layer_name in c3k_cv1_bn_layer_names]
                c3k_cv1outs = [torch.sum(c3k_cv1_bn_mask).int().item() for c3k_cv1_bn_mask in c3k_cv1_bn_masks]
                c3k_cv2_bn_layer_names = [base_name + f'.m.{i}.cv2.bn' for i in range(n)]
                c3k_cv2_bn_masks = [maskbndict[c3k_cv2_bn_layer_name] for c3k_cv2_bn_layer_name in c3k_cv2_bn_layer_names]
                c3k_cv2outs = [torch.sum(c3k_cv2_bn_mask).int().item() for c3k_cv2_bn_mask in c3k_cv2_bn_masks]
                inner_c3k_cv3_bn_layer_names = [base_name + f'.m.{i}.cv3.bn' for i in range(n)]
                inner_c3k_cv3_bn_masks = [maskbndict[inner_c3k_cv3_bn_layer_name] for inner_c3k_cv3_bn_layer_name in inner_c3k_cv3_bn_layer_names]
                c3k_cv3outs = [torch.sum(inner_c3k_cv3_bn_mask).int().item() for inner_c3k_cv3_bn_mask in inner_c3k_cv3_bn_masks]
                c3k_inner_cv1_bn_layer_namess = [[base_name + f'.m.{i}.m.{j}.cv1.bn' for j in range(2)] for i in range(n)] # 原代码这里是写死的, 每个C3k中的Bottleneck重复2次
                c3k_inner_cv1_bn_masks = [[maskbndict[c3k_inner_cv1_bn_layer_name] for c3k_inner_cv1_bn_layer_name in c3k_inner_cv1_bn_layer_names] for c3k_inner_cv1_bn_layer_names in c3k_inner_cv1_bn_layer_namess]
                c3k_inner_cv1outs = [[torch.sum(c3k_inner_cv1_bn_mask).int().item() for c3k_inner_cv1_bn_mask in c3k_inner_cv1_bn_masks] for c3k_inner_cv1_bn_masks in c3k_inner_cv1_bn_masks]

            args = [
                c2f_cv1in, c2f_cv1out, c2f_cv1_split_sections, c2f_cv2out, 
                bottle_inner_cv1outs, bottle_inner_cv2outs,
                c3k_cv1outs, c3k_cv2outs, c3k_inner_cv1outs, c3k_cv3outs,
                n, *args[1:]
            ]
            c2 = c2f_cv2out
            # =============================================
            current_to_prev[c2f_cv1_bn_layer_name] = prev_bn_layer_name
            # 如果当前C2f的前一层是Concat层, 就要记录Concat层是由哪几个支路来的
            if prev_module in [Concat]:
                current_to_prev[c2f_cv1_bn_layer_name] = [idx_to_bn_layer_name[ix] for ix in idx_to_bn_layer_name[i + f]]
            prev_bn_layer_name = c2f_cv1_bn_layer_name
            if not use_c3k:
                prev_bn_layer_names_for_last_cv2 = [c2f_cv1_bn_layer_name]
                for i_inner in range(n):
                    inner_cv1_bn_layer_name = base_name + f'.m.{i_inner}.cv1.bn'
                    inner_cv2_bn_layer_name = base_name + f'.m.{i_inner}.cv2.bn'
                    current_to_prev[inner_cv1_bn_layer_name] = prev_bn_layer_name
                    prev_bn_layer_name = inner_cv1_bn_layer_name
                    current_to_prev[inner_cv2_bn_layer_name] = prev_bn_layer_name
                    prev_bn_layer_name = inner_cv2_bn_layer_name
                    prev_bn_layer_names_for_last_cv2.append(inner_cv2_bn_layer_name)
                current_to_prev[c2f_cv2_bn_layer_name] = prev_bn_layer_names_for_last_cv2
            else:
                for i_inner in range(n):
                    inner_c3k_cv1_bn_layer_name = base_name + f'.m.{i_inner}.cv1.bn'
                    inner_c3k_cv2_bn_layer_name = base_name + f'.m.{i_inner}.cv2.bn'
                    current_to_prev[inner_c3k_cv1_bn_layer_name] = prev_bn_layer_name
                    current_to_prev[inner_c3k_cv2_bn_layer_name] = prev_bn_layer_name
                    prev_bn_layer_name = inner_c3k_cv1_bn_layer_name
                    for j_inner in range(2):
                        inner_c3k_bottle_cv1_bn_layer_name = base_name + f'.m.{i_inner}.m.{j_inner}.cv1.bn'
                        inner_c3k_bottle_cv2_bn_layer_name = base_name + f'.m.{i_inner}.m.{j_inner}.cv2.bn'
                        current_to_prev[inner_c3k_bottle_cv1_bn_layer_name] = prev_bn_layer_name
                        prev_bn_layer_name = inner_c3k_bottle_cv1_bn_layer_name
                        current_to_prev[inner_c3k_bottle_cv2_bn_layer_name] = prev_bn_layer_name
                        prev_bn_layer_name = inner_c3k_bottle_cv2_bn_layer_name     
                    inner_c3k_cv3_bn_layer_name = base_name + f'.m.{i_inner}.cv3.bn' 
                    current_to_prev[inner_c3k_cv3_bn_layer_name] = [inner_c3k_bottle_cv2_bn_layer_name, inner_c3k_cv2_bn_layer_name]              
                    prev_bn_layer_name = inner_c3k_cv3_bn_layer_name
                current_to_prev[c2f_cv2_bn_layer_name] = [inner_c3k_cv3_bn_layer_name, c2f_cv1_bn_layer_name]
            prev_bn_layer_name = c2f_cv2_bn_layer_name
            idx_to_bn_layer_name[i] = c2f_cv2_bn_layer_name
            # =============================================
            n = 1 
        elif m in [SPPFPruned]:
            # SPPF层中不止是maxpool层, 还有两个卷积层, 所以要重写
            cv1in = ch[f]
            cv1_bn_layer_name = base_name + '.cv1.bn'
            cv1_mask = maskbndict[cv1_bn_layer_name]
            cv1out = torch.sum(cv1_mask).int().item()
            cv2_bn_layer_name = base_name + '.cv2.bn'
            cv2_mask = maskbndict[cv2_bn_layer_name]
            cv2out = torch.sum(cv2_mask).int().item()
            args = [cv1in, cv1out, cv2out, *args[1:]]
            c2 = cv2out
            # =============================================
            current_to_prev[cv1_bn_layer_name] = prev_bn_layer_name
            prev_bn_layer_name = cv1_bn_layer_name
            current_to_prev[cv2_bn_layer_name] = prev_bn_layer_name
            prev_bn_layer_name = cv2_bn_layer_name
            idx_to_bn_layer_name[i] = cv2_bn_layer_name
            # =============================================
        elif m in [C2PSAPruned]:
            args.insert(2, n)
            c1, c2 = ch[f], args[0]
            c2 = make_divisible(min(c2, max_channels) * width, 8)
            inner_ffn_cv1_bn_layer_names = [base_name + f'.m.{i}.ffn.0.bn' for i in range(n)]
            inner_ffn_cv1_masks = [maskbndict[inner_ffn_cv1_bn_layer_name] for inner_ffn_cv1_bn_layer_name in inner_ffn_cv1_bn_layer_names]
            inner_ffn_cv1outs = [torch.sum(inner_ffn_cv1_mask).int().item() for inner_ffn_cv1_mask in inner_ffn_cv1_masks]
            c2f_cv2_bn_layer_name = base_name + '.cv2.bn'
            c2f_cv2_mask = maskbndict[c2f_cv2_bn_layer_name]
            c2f_cv2out = torch.sum(c2f_cv2_mask).int().item()
            args = [c1, c2, inner_ffn_cv1outs, c2f_cv2out, *args[1:]]
            c2 = c2f_cv2out
            n = 1
            # =============================================
            # 这里将C2PSA看成一个复杂版本的C2f, 外壳是C2f, 内部是PSABlock
            c2f_cv1_bn_layer_name = base_name + '.cv1.bn'
            current_to_prev[c2f_cv1_bn_layer_name] = prev_bn_layer_name
            prev_bn_layer_name = c2f_cv1_bn_layer_name
            for i_inner in range(n):
                inner_attention_qkv_bn_layer_name = base_name + f'.m.{i_inner}.attn.qkv.bn' 
                current_to_prev[inner_attention_qkv_bn_layer_name] = prev_bn_layer_name
                inner_attention_pe_bn_layer_name = base_name + f'.m.{i_inner}.attn.pe.bn' 
                current_to_prev[inner_attention_pe_bn_layer_name] = inner_attention_qkv_bn_layer_name
                inner_attention_proj_bn_layer_name = base_name + f'.m.{i_inner}.attn.proj.bn' 
                current_to_prev[inner_attention_proj_bn_layer_name] = [inner_attention_pe_bn_layer_name]
                inner_ffn_cv1_bn_layer_name = base_name + f'.m.{i_inner}.ffn.0.bn' 
                current_to_prev[inner_ffn_cv1_bn_layer_name] = inner_attention_proj_bn_layer_name
                inner_ffn_cv2_bn_layer_name = base_name + f'.m.{i_inner}.ffn.1.bn'
                current_to_prev[inner_ffn_cv2_bn_layer_name] = inner_ffn_cv1_bn_layer_name
                prev_bn_layer_name = inner_ffn_cv2_bn_layer_name
            c2f_cv2_bn_layer_name = base_name + '.cv2.bn'
            current_to_prev[c2f_cv2_bn_layer_name] = [c2f_cv1_bn_layer_name, inner_ffn_cv2_bn_layer_name]
            c2f_cv2_bn_layer_name = c2f_cv2_bn_layer_name
            idx_to_bn_layer_name[i] = c2f_cv2_bn_layer_name
            # =============================================
        elif m in [nn.Upsample]:
            c2 = ch[f]
            idx_to_bn_layer_name[i] = idx_to_bn_layer_name[i-1]
        elif m in [Concat]:
            c2 = sum(ch[x] for x in f)
            # =============================================
            fx = []
            for ix in f:
                if ix == -1:
                    fx.append(i + ix)
                else:
                    fx.append(ix)
            idx_to_bn_layer_name[i] = fx
            # =============================================
        elif m in [DetectPruned]:
            # cv2x?是有所有的左侧分支, cv3x?是所有的右侧分支
            args.append([ch[x] for x in f])
            cv2x0_out_bn_layer_names = [base_name + f'.cv2.{i}.0.bn' for i in range(3)]
            cv2x1_out_bn_layer_names = [base_name + f'.cv2.{i}.1.bn' for i in range(3)]
            cv2x2_out_conv_layer_names = [base_name + f'.cv2.{i}.2' for i in range(3)]
            cv3x0x0_out_bn_layer_names = [base_name + f'.cv3.{i}.0.0.bn' for i in range(3)]
            cv3x0x1_out_bn_layer_names = [base_name + f'.cv3.{i}.0.1.bn' for i in range(3)]
            cv3x1x0_out_bn_layer_names = [base_name + f'.cv3.{i}.1.0.bn' for i in range(3)]
            cv3x1x1_out_bn_layer_names = [base_name + f'.cv3.{i}.1.1.bn' for i in range(3)]
            cv3x2_out_conv_layer_names = [base_name + f'.cv3.{i}.2' for i in range(3)]
            cv2x0_mask = [maskbndict[x] for x in cv2x0_out_bn_layer_names]
            cv2x1_mask = [maskbndict[x] for x in cv2x1_out_bn_layer_names]
            cv3x0_mask = [maskbndict[x] for sublist in zip(cv3x0x0_out_bn_layer_names, cv3x0x1_out_bn_layer_names) for x in sublist]
            cv3x1_mask = [maskbndict[x] for sublist in zip(cv3x1x0_out_bn_layer_names, cv3x1x1_out_bn_layer_names) for x in sublist]
            cv2x0_outs = [torch.sum(x).int().item() for x in cv2x0_mask]
            cv2x1_outs = [torch.sum(x).int().item() for x in cv2x1_mask]
            cv3x0_outs = [torch.sum(x).int().item() for x in cv3x0_mask]
            cv3x1_outs = [torch.sum(x).int().item() for x in cv3x1_mask]
            cv3x0_outs = [cv3x0_outs[:2], cv3x0_outs[2:4], cv3x0_outs[4:]]
            cv3x1_outs = [cv3x1_outs[:2], cv3x1_outs[2:4], cv3x1_outs[4:]]
            args = [cv2x0_outs, cv2x1_outs, cv3x0_outs, cv3x1_outs, *args]
            # =============================================
            for ix, (cv2x0_out_bn_layer_name, cv3x0x0_out_bn_layer_name) \
                in enumerate(zip(cv2x0_out_bn_layer_names, cv3x0x0_out_bn_layer_names)):
                current_to_prev[cv2x0_out_bn_layer_name] = idx_to_bn_layer_name[f[ix]]
                current_to_prev[cv3x0x0_out_bn_layer_name] = idx_to_bn_layer_name[f[ix]]
            for ix in range(3):
                current_to_prev[cv2x1_out_bn_layer_names[ix]] = cv2x0_out_bn_layer_names[ix]
                current_to_prev[cv2x2_out_conv_layer_names[ix]] = cv2x1_out_bn_layer_names[ix]
                current_to_prev[cv3x0x1_out_bn_layer_names[ix]] = cv3x0x0_out_bn_layer_names[ix]
                current_to_prev[cv3x1x0_out_bn_layer_names[ix]] = cv3x0x1_out_bn_layer_names[ix]
                current_to_prev[cv3x1x1_out_bn_layer_names[ix]] = cv3x1x0_out_bn_layer_names[ix]
                current_to_prev[cv3x2_out_conv_layer_names[ix]] = cv3x1x1_out_bn_layer_names[ix]
            # =============================================
        else:
            raise ValueError(f"ERROR ❌ module {m} not supported in parse_model.")
        prev_module = m
            
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        m.np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
        if verbose:
            LOGGER.info(f'{i:>3}{str(f):>20}{n_:>3}{m.np:10.0f}  {t:<50}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save), current_to_prev


if __name__ == "__main__":
    import prune
    dummies = torch.randn([4, 3, 640, 640], dtype=torch.float32).cuda()
    opt = prune.parse_opt()
    maskbndict, pruned_yaml = prune.main(opt)
    model = DetectionModelPruned(maskbndict, pruned_yaml, ch=3)
    model.train().cuda()
    out = model(dummies)