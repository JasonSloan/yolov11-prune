from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  

from ultralytics import YOLO

weight = "weights/officials/yolo11l.pt"
model = YOLO(weight)
model.train(
    sr=0,
    data='ultralytics/cfg/datasets/coco.yaml',
    cfg='ultralytics/cfg/default.yaml',
    project='.',
    name='runs/train-normal',
    epochs=200,
    batch=10,
    imgsz=640,
    device=[0, 1],
    resume=False,
    workers=8,
    optimizer='SGD',
    lr0=1e-4,
    patience=50,
    multi_scale=True,
    label_smoothing=True
)