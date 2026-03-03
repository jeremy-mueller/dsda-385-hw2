from loader import load
from rcnn import train
from eval import eval
from ultralytics import YOLO

dataset = "oxfordiiit"
model = "rcnn"
epochs = 15
batch_size = 4

if __name__ == '__main__':
    train_loader, test_loader, val_loader = load(dataset, batch_size=batch_size)

    if model == "rcnn":
        train(epochs=epochs, train_loader=train_loader, val_loader=val_loader)

        eval(test_loader, model, dataset)
    elif model == "yolo":
        model = YOLO("yolov8n.pt")
        results = model.train(data=f"datasets/{dataset}.yaml", epochs=epochs, batch=batch_size, imgsz=512)

        model = YOLO("runs/detect/train/weights/best.pt")
        metrics = model.val(data=f"datasets/{dataset}.yaml", split="test", imgsz=512)
        print(metrics)