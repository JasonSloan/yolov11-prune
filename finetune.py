"""
ultralytics/cfg/__init__.py中修改了增加'finetune'从overrides中pop和重赋值,防止参数检查报错
ultralytics/engine/model.py中增加了对'maskbndict'的加载
"""
from ultralytics import YOLO

weight = "weights/pruned.pt"

model = YOLO(weight)
# finetune设置为True
model.train(
    data='ultralytics/cfg/datasets/coco.yaml', 
    cfg='ultralytics/cfg/default.yaml',
    project='.',
    name='runs/finetune',
    epochs=200, 
    batch=16,
    imgsz=640,
    optimizer='Adam',
    lr0=1e-4,
    finetune=True, 
    device=1,
    resume=False,
    workers=8,
    multi_scale=True,
    label_smoothing=True
)