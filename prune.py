"""
新增的代码文件:
ultralytics/nn/modules/block_pruned.py: 新增Bottleneckpruned, C2fpruned, SPPFpruned 
ultralytics/nn/modules/head_pruned.py: 新增Detect_pruned
ultralytics/nn/tasks_pruend.py: 新增DetectionModelPruned, parse_model_pruned
"""
"""
编写此部分代码, 要同时结合yolov11.yaml模型结构文件, netron下的yolov11.onnx文件一起看
"""
import re
import os
import sys
from pathlib import Path

import yaml
import argparse
import onnx  
import onnxslim

import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.utils import colorstr
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.nn.modules import Conv, Concat, Bottleneck, Attention, PSABlock, C2PSA, DWConv, Detect

from ultralytics.nn.modules.block_pruned import SPPFPruned, C3k2Pruned, C2PSAPruned
from ultralytics.nn.modules.head_pruned import DetectPruned
from ultralytics.nn.tasks_pruned import DetectionModelPruned

import warnings
warnings.filterwarnings('ignore')
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def main(opt):
    weights, prune_ratio, cfg, model_size, save_dir = opt.weights, opt.prune_ratio, opt.cfg, opt.model_size, opt.save_dir
    model = AutoBackend(weights, fuse=False)
    model.eval()
    # =========================================step1=========================================
    """
    遍历所有module:
        将module中的所有bn层按照{名字: bn层}的形式存在一个字典中;
        同时将忽略剪枝的bn层的名字存在一个列表中(具有残差结构的位置);
    """
    print(colorstr('blue', 'Gather bn weights...'))
    # 聚集所有的bn层和忽略剪枝的bn层
    bn_dict = {}
    # 这三层是连接Detect的三个卷积层, 因为Detect中的右侧分支的第一个卷积是深度可分离卷积, 输入通道数和输出通道数必须一样, 所以连接其的前一个卷积忽略剪枝
    ignore_bn_list = ['model.16.cv2.bn', 'model.19.cv2.bn', 'model.22.cv2.bn'] 
    chunk_bn_list = []
    for name, module in model.model.named_modules():
        # 在Bottleneck结构中, 如果有残差连接, 那么Add相加的两个分支的通道数要保持一致, 所以不剪
        if isinstance(module, Bottleneck):
            if module.add:  # 虽然在v11中是C3k2模块, 但实际上上和v8中的C2f模块对忽略剪枝的bn层的处理是一样的
                ignore_bn_list.append(name.rsplit(".", 2)[0] + ".cv1.bn")  # C2f模块中的第一个卷积层的bn层
                ignore_bn_list.append(name + '.cv2.bn')                    # C2f模块中的BottleNeck模块中的第二个卷积层
            else:
                # yolov11与yolov8是由backbone+head构成的, 与v8不同的是, v8中的head中所有的c2f模块是没有add操作的, 所以也可以剪,
                # 但是由于C2f在forward时有chunck操作, 所以要保证chunck之前的通道数是偶数, 这里用chunk_bn_list保存一下
                # 后面剪枝的时候, 把chunck之前的通道数是奇数的调整为偶数
                # 但是v11的是有add操作的, 所以这个else代码其实不生效
                chunk_bn_list.append(f"{name[:-4]}.cv1.bn")
        if isinstance(module, Attention):
            ignore_bn_list.append(name + ".qkv.bn")                        # Attention模块中的qkv中的bn层
            ignore_bn_list.append(name + ".pe.bn")                         # Attention模块中的pe分支的卷积层的bn层
            ignore_bn_list.append(name + ".proj.bn")                       # Attention模块中的proj分支的卷积层的bn层
        if isinstance(module, PSABlock):
            ignore_bn_list.append(name + ".ffn.1.bn")                      # Attention模块中的ffn分支的第二个卷积层的bn层
        if isinstance(module, C2PSA):
            ignore_bn_list.append(name + ".cv1.bn")
        if isinstance(module, Detect):                                     # Detect模块右侧分支的中两个深度可分离卷积之间的卷积
            for i in range(3):
                ignore_bn_list.append(name + f".cv3.{i}.0.1.bn")
                ignore_bn_list.append(name + f".cv3.{i}.1.1.bn")
        if isinstance(module, DWConv):
            ignore_bn_list.append(name + ".bn")
        # 这里bn_dict保存了所有bn层的信息(包括忽略剪枝的bn层)
        if isinstance(module, nn.BatchNorm2d):
            bn_dict[name] = module     
    # 检查下忽略剪枝的bn层是否有命名错误的        
    for ignore_bn_name in ignore_bn_list:
        assert ignore_bn_name in bn_dict.keys()
    
    # =========================================step2========================================= 
    """按照忽略列表过滤已存储的bn层的字典"""
    # 按照ignore_bn_list过滤bn_dict
    bn_dict = {k : v for k, v in bn_dict.items() if k not in ignore_bn_list}
    
    # =========================================step3=========================================
    """遍历剩余的bn层, 将bn层的gamma值拿出来, 按照顺序依次放入同一个列表中(即将要剪枝的bn层的**gamma的绝对值**全都gather起来)"""
    bn_weights = []
    for name, module in bn_dict.items():
        # 注意这里一定要取绝对值
        bn_weights.extend(module.weight.data.abs().clone().cpu().tolist())
        
    # =========================================step4========================================= 
    """将step3中的列表按照数值大小排序得到一个新的已排序好的数组, 准备计算阈值"""
    sorted_bn = torch.sort(torch.tensor(bn_weights))[0]
    
    # ========================================step5=========================================
    """
    遍历bn_dict, 计算每一个bn层的gamma的绝对值的最大值(即当前层gamma的最大值), 存入highest_thre中
    计算highest_thre所有值的最小值, 剪枝的阈值不能超过该值, 因为如果超过该值, 那么就存在可能某一层整体被剪掉的情况
    按照指定剪枝比率计算剪枝阈值: 即按比率先计算sorted_bn对应索引, 然后取索引对应的值即为阈值
    """
    print(colorstr('blue', 'Calculating pruning threshold...'))
    highest_thre = []
    for name, module in bn_dict.items():
        # 计算每个bn层的weight(gamma)的最大值
        highest_thre.append(module.weight.data.abs().clone().cpu().max())
    # 所有最大值的最小值即为剪枝的最大阈值
    highest_thre = min(highest_thre)
    # 计算剪枝最大的比率
    percent_limit = (sorted_bn == highest_thre).nonzero()[0, 0].item() / len(sorted_bn)
    # 计算按照当前用户指定比率剪枝下的阈值
    thre = sorted_bn[int(len(sorted_bn) * prune_ratio)] if prune_ratio >= 0 else -1
    print(f'Pruning gamma values should be less than {colorstr(f"{highest_thre:.4f}")}, yours is {colorstr(f"{thre:.4f}")}')
    print(f'The corresponding pruing ratio should be less than {colorstr(f"{percent_limit:.3f}")}, yours is {colorstr(f"{prune_ratio:.3f}")}')
    if prune_ratio > percent_limit:
        prune_ratio = percent_limit
        print(f'Pruing ratio falling down to {colorstr(f"{prune_ratio:.3f}")}')
        
    # ========================================step6=========================================
    """将模型配置文件重新保存为一个字典(注意, 这里重写了C2f模块、SPPF模块和Detect模块)"""
    pruned_yaml = {}
    with open(cfg, encoding='ascii', errors='ignore') as f:
        model_yamls = yaml.safe_load(f)  # model dict
    # # Define model
    nc = model.model.nc
    pruned_yaml["nc"] = nc
    pruned_yaml["scales"] = model_yamls["scales"]
    pruned_yaml["scale"] = model_size
    pruned_yaml["backbone"] = [
        [-1, 1, Conv, [64, 3, 2]], # 0-P1/2
        [-1, 1, Conv, [128, 3, 2]], # 1-P2/4
        [-1, 2, C3k2Pruned, [256, False, 0.25]],
        [-1, 1, Conv, [256, 3, 2]], # 3-P3/8
        [-1, 2, C3k2Pruned, [512, False, 0.25]],
        [-1, 1, Conv, [512, 3, 2]], # 5-P4/16
        [-1, 2, C3k2Pruned, [512, True]],
        [-1, 1, Conv, [1024, 3, 2]], # 7-P5/32
        [-1, 2, C3k2Pruned, [1024, True]],
        [-1, 1, SPPFPruned, [1024, 5]], # 9
        [-1, 2, C2PSAPruned, [1024]], # 10
    ]
    pruned_yaml["head"] = [
        [-1, 1, nn.Upsample, [None, 2, "nearest"]],
        [[-1, 6], 1, Concat, [1]], # cat backbone P4
        [-1, 2, C3k2Pruned, [512, False]], # 13
        
        [-1, 1, nn.Upsample, [None, 2, "nearest"]],
        [[-1, 4], 1, Concat, [1]], # cat backbone P3
        [-1, 2, C3k2Pruned, [256, False]], # 16 (P3/8-small)
        
        [-1, 1, Conv, [256, 3, 2]],
        [[-1, 13], 1, Concat, [1]], # cat head P4
        [-1, 2, C3k2Pruned, [512, False]], # 19 (P4/16-medium)
        
        [-1, 1, Conv, [512, 3, 2]],
        [[-1, 10], 1, Concat, [1]], # cat head P5
        [-1, 2, C3k2Pruned, [1024, True]], # 22 (P5/32-large)
        
        [[16, 19, 22], 1, DetectPruned, [nc]] # Detect(P3, P4, P5)
    ]
    # ========================================step7=========================================
    """
    遍历模型的每一个module, 按照剪枝阈值计算每一个module的掩码mask, 然后将bn层对应的weight(gamma)和bias(beta)乘以掩码mask; 
    计算每一层的原本的channel数, 剪枝后的channel数, 打印出来; 
    按照{bn层名字: bn层mask掩码矩阵}保存出一个maskbndict
    """
    print(colorstr('blue', 'Pruning model.....'))
    print("=" * 94)
    print(f"|\t{'layer name':<25}{'|':<10}{'origin channels':<20}{'|':<10}{'remaining channels':<20}|")
    maskbndict = {}
    for name, module in model.model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            # 计算剪枝前的通道数
            origin_channels = module.weight.data.size()[0]
            # 初始化剪枝后剩余的通道数
            remaining_channels = origin_channels
            # 初始化掩码mask
            mask = torch.ones(origin_channels)
            if name not in ignore_bn_list:
                mask = module.weight.data.abs().gt(thre).float()
                # 如果剪枝后剩余通道数是奇数而且是C2f结构, 那么chunck的时候就会出问题, 这里把剩余通道数变成偶数
                if name in chunk_bn_list and mask.sum() % 2 == 1:
                    # 将所有的权重值展平并排序
                    flattened_sorted_weight = torch.sort(module.weight.data.abs().view(-1))[0]
                    # 找到第一个大于阈值的元素的索引
                    idx = torch.min(torch.nonzero(flattened_sorted_weight.gt(thre))).item()
                    # 重新计算阈值, 新阈值只要比第idx - 1个位置元素的值小一点就可以
                    thre_ = flattened_sorted_weight[idx - 1] - 1e-6
                    mask = module.weight.data.abs().gt(thre_).float()
                assert mask.sum() > 0, f"bn {name} has no active elements"
                module.weight.data.mul_(mask)
                module.bias.data.mul_(mask)
                remaining_channels = mask.sum().int()    
            maskbndict[name] = mask
            print(f"|\t{name:<25}{'|':<10}{origin_channels:<20}{'|':<10}{remaining_channels:<20}|")
    print("=" * 94)
    
    # ========================================step8========================================= 
    """
    剪枝后的网络拓扑构建
    """ 
    print(colorstr('blue', 'Building pruned model.....'), end='')
    pruned_model = DetectionModelPruned(maskbndict=maskbndict, cfg=pruned_yaml, ch=3).cuda()
    pruned_model.eval() 
    # ========================================step9========================================= 
    # """
    # 将剪枝后的模型保存为onnx格式, 可以使用netron查看
    # """
    print(colorstr('blue', 'Exporting pruned model to onnx.....'))
    f = os.path.join(save_dir, 'pruned.onnx')
    pruned_model = pruned_model.cpu()
    for m in pruned_model.modules():
        if isinstance(m, (DetectPruned)):
            m.export = True
            m.format = 'CPU'
            m.shape = None
    dummies = torch.randn([1, 3, 640, 640], dtype=torch.float32).cpu()
    torch.onnx.export(
        pruned_model,
        dummies,
        f,
        input_names=['images'],
        output_names=['output'],
        opset_version=13
    )
    model_onnx = onnx.load(f)
    model_onnx = onnxslim.slim(model_onnx)
    onnx.save(model_onnx, f)
    print(colorstr('blue', f'Onnx Model has been saved to {f}'))
    # ========================================step10=========================================      
    """
    将原模型与剪枝后构建好的模型zip起来, 然后遍历每一层的module, 将原模型对应于剪枝后的通道位置的参数赋值给剪枝后的模型
    """
    print(colorstr('blue', 'Assigning pruned model weights from original model.....'))
    current_to_prev = pruned_model.current_to_prev
    # 验证一下current_to_prev中所有的key和value都在maskbndict中
    for xks, xvs in current_to_prev.items():
        xvs = [xvs] if not isinstance(xvs, list) else xvs
        for xk, xv in zip(xks, xvs):
            assert all([xk, xv in maskbndict.keys()]), f"{xk, xv} from 'current_to_prev' not in maskbndict" 
    changed = []
    # 可匹配c3k2模块中的Bottleneck中的第一个卷积层(在不使用c3k的情况下) 以及 c3k2模块中的c3k中的cv1卷积层(在使用c3k的情况下)
    pattern_c3k2_general_cv1 = re.compile(r"model.\d+.m.0.cv1.bn")
    # 匹配c3k2模块中的c3k中的cv2卷积层(在使用c3k的情况下)
    pattern_c3k2_c3k_cv2 = re.compile(r"model.\d+.m.0.cv2.bn")
    # 匹配c2psa中的attention模块中的pe层
    pattern_attention_pe = re.compile(r"model.\d+.m.\d.attn.pe.bn")
    # 匹配Detect模块中的最后一个卷积层
    pattern_detect = re.compile(r"model.\d+.cv\d.\d.2")
    model_yamls = model_yamls['backbone'] + model_yamls['head']
    for (name_org, module_org), (name_pruned, module_pruned) in \
        zip(model.model.named_modules(), pruned_model.named_modules()): 
            
        assert name_org == name_pruned, f"name_org: {name_org} != name_pruned: {name_pruned}"
        layer_index = None if 'model.' not in name_org else int(name_org.split('.')[1])
        use_c3k = False if layer_index is None else 'C3k2' in model_yamls[layer_index] and 'True' in [str(model_yamls[layer_index][-1][i]) for i in range(len(model_yamls[layer_index][-1]))]
        
        # 如果是dfl层, 说明已经结束了
        if 'dfl' in name_org:
            break
        
        # 如果是Detect模块中的最后一个卷积(不带BN层的卷积)  
        # 只需要改变一下in_channels就可以了, 不需要改变out_channels  
        if pattern_detect.fullmatch(name_org) is not None:
            current_conv_layer_name = name_org
            prev_bn_layer_name = current_to_prev[current_conv_layer_name]
            in_channels_mask = maskbndict[prev_bn_layer_name].to(torch.bool)
            module_pruned.weight.data = module_org.weight.data[:, in_channels_mask, :, :]
            if module_org.bias is not None:
                assert module_pruned.bias.data is not None, f"{name_pruned} has no bias"
                module_pruned.bias.data = module_org.bias.data
            continue
            
        if isinstance(module_org, nn.Conv2d):
            currnet_bn_layer_name = name_org[:-4] + 'bn'
            state_dict_org = module_org.weight.data
            out_channels_mask = maskbndict[currnet_bn_layer_name].to(torch.bool)
            prev_bn_layer_name = current_to_prev.get(currnet_bn_layer_name, None)
            if isinstance(prev_bn_layer_name, list):
                in_channels_masks = [maskbndict[ni] for ni in prev_bn_layer_name]
                # 特殊处理: 此时是C2PSA结构中的最后一个卷积(C2PSA与C3k2虽然都是类似于C2f的结构, 但是对于最后一个卷积的输入不同)
                if 'model.10.cv2' in currnet_bn_layer_name:
                    in_channels_masks = [maskbndict[prev_bn_layer_name[0]].chunk(2)[0], maskbndict[prev_bn_layer_name[1]]]
                in_channels_mask = torch.cat(in_channels_masks, dim=0).to(torch.bool)
            elif prev_bn_layer_name is not None:
                in_channels_mask = maskbndict[prev_bn_layer_name].to(torch.bool)
                chunck_in_channels = any(
                    [   
                        # 匹配c3k2模块中的Bottleneck中的第一个卷积层(在不使用c3k的情况下) 以及 c3k2模块中的c3k中的cv1卷积层(在使用c3k的情况下)
                        pattern_c3k2_general_cv1.fullmatch(currnet_bn_layer_name) is not None, 
                        # 匹配c3k2模块中的c3k中的cv2卷积层(在使用c3k的情况下)
                        pattern_c3k2_c3k_cv2.fullmatch(currnet_bn_layer_name) is not None and use_c3k,
                        'model.10.m.0.attn.qkv' in currnet_bn_layer_name
                    ]
                )
                if chunck_in_channels:
                    in_channels_mask = in_channels_mask.chunk(2, 0)[1]
                # 特殊处理: 此时是SPPF结构中的第二个卷积
                if name_org == "model.9.cv2.conv":
                    in_channels_mask = torch.cat([in_channels_mask for _ in range(4)], dim=0)
                if pattern_attention_pe.fullmatch(currnet_bn_layer_name) is not None:
                    in_channels_mask = torch.ones(1, dtype=torch.bool)
            else:
                in_channels_mask = torch.ones(module_org.weight.data.shape[1], dtype=torch.bool)
            if module_org.groups > 1:
                in_channels_mask = torch.ones([1], dtype=torch.bool)
            state_dict_org = module_org.weight.data[out_channels_mask, :, :, :]
            state_dict_org = state_dict_org[:, in_channels_mask, :, :]
            module_pruned.weight.data = state_dict_org
            
            # 如果断言失败, 那么说明剪枝模型构建的有问题
            assert module_pruned.in_channels == state_dict_org.shape[1] * module_pruned.groups, \
                f"{name_org} module weight mismatch, module_pruned.in_channels: {module_pruned.in_channels}, " \
                f"state_dict_org.shape[1]: {state_dict_org.shape[1]} \n"
            assert module_pruned.out_channels == state_dict_org.shape[0], \
                f"{name_org} module weight mismatch, module_pruned.out_channels: {module_pruned.out_channels}, " \
                f"state_dict_org.shape[0]: {state_dict_org.shape[0]} \n"
                
            if module_org.bias is not None:
                assert module_pruned.bias.data is not None, f"{name_pruned} has no bias"
                module_pruned.bias.data = module_org.bias.data[out_channels_mask]
            changed.append(currnet_bn_layer_name)
            
        if isinstance(module_org, nn.BatchNorm2d):
            out_channels_mask = maskbndict[name_org].to(torch.bool)
            module_pruned.weight.data = module_org.weight.data[out_channels_mask]
            module_pruned.bias.data = module_org.bias.data[out_channels_mask]
            module_pruned.running_mean = module_org.running_mean[out_channels_mask]
            module_pruned.running_var = module_org.running_var[out_channels_mask]
    missing = [name for name in maskbndict.keys() if name not in changed]
    assert not missing, "mising: {missing}" 
    # ========================================step11=========================================
    """
    保存模型
    """
    print(colorstr('blue', 'Saving pruned model.....'))
    pruned_model.model[23].shape = None
    pruned_model.eval().cpu()
    save_path = os.path.join(save_dir, "pruned.pt")
    torch.save(
        {
            "model": pruned_model,
            "maskbndict": maskbndict
        },
        save_path
    )
    model = YOLO(save_path)
    dummies = torch.randn([1, 3, 640, 640], dtype=torch.float32)
    model.predict(dummies, verbose=False)
    print(colorstr('blue', f'Pruned model has been saved to {save_path}'))
    

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'ultralytics/cfg/datasets/coco.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'runs/train-sparsity/weights/last.pt', help='model.pt path(s)')
    parser.add_argument('--cfg', type=str, default=ROOT / 'ultralytics/cfg/models/11/yolo11.yaml', help='model.yaml path')
    parser.add_argument('--model-size', type=str, default='l', help='(yolov11)n, s, m, l or x?')
    parser.add_argument('--prune-ratio', type=float, default=0.2, help='prune ratio, leave it -1 if you dont want to prune(for example, debug purpose)')
    parser.add_argument('--save-dir', type=str, default=ROOT / 'weights', help='pruned model weight save dir')
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
