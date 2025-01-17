"""
修改的代码:
ultralytics/nn/modules/block.py: 对C3k2增加一个C3k布尔值属性
ultralytics/engine/trainer.py: 禁用amp, 梯度裁剪, 增加梯度惩罚项系数
ultralytics/engine/model.py: 主要是将sr参数绑定到self.trainer上
ultralytics/cfg/__init__.py: 对额外参数finetune的处理, 防止DDP下报错
ultralytics/engine/model.py: 对sr, maskbndict等额外参数的处理
"""
from ultralytics import YOLO

model = YOLO("runs/train-normal/weights/best.pt")
# L1正则的惩罚项系数sr
model.train(
    sr=1e-3, 
    data="ultralytics/cfg/datasets/coco.yaml", 
    cfg='ultralytics/cfg/default.yaml',
    project='.',
    name='runs/train-sparsity',
    device=0, # NOTE: 目前只能单卡训, DDP下多卡训不会产生稀疏效果(TODO)
    epochs=50, 
    batch=10,
    optimizer='SGD',
    lr0=1e-3, 
    patience=50 # 注意patience要比epochs大, 防止训练过早结束
)