from ultralytics import YOLO

model = YOLO("yolo12n.pt")   # or yolov12s.pt, yolov12m.pt...

model.train(
    data="dataset_yolo.yaml",   # path to your YAML file
    epochs=20,
    imgsz=640,
    batch=8,
)
