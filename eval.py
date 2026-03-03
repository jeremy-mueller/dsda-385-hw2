import torch
import time
import torchvision.utils as vutils
import matplotlib.pyplot as plt

from rcnn import get_model
from torchmetrics.detection.mean_ap import MeanAveragePrecision

def eval(test_loader, threshold=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes=11)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.to(device)

    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
    model.eval()

    total_images = 0
    start_time = time.time()

    with torch.no_grad():
        for images, targets in test_loader:
            images_gpu = list(img.to(device) for img in images)
            
            batch_start = time.time()
            preds = model(images_gpu)
            
            total_images += len(images)
            
            preds = [{k: v.cpu() for k, v in p.items()} for p in preds]
            targets = [{k: v.cpu() for k, v in t.items()} for t in targets]
            
            metric.update(preds, targets)
            
    end_time = time.time()
    
    results = metric.compute()
    
    total_time = end_time - start_time
    fps = total_images / total_time
    
    print("\n--- Final Report ---")
    print(f"mAP @ 0.5:     {results['map_50'].item():.4f}")
    print(f"Recall:         {results['mar_100'].item():.4f}") 
    print(f"Precision:       {results['map'].item():.4f}")
    print(f"Inference Speed: {fps:.2f} images per second")

    
    images, targets = next(iter(test_loader))
    dataset = test_loader.dataset
    
    if hasattr(dataset, 'dataset'):
        base_ds = dataset.dataset
    else:
        base_ds = dataset

    images_gpu = list(img.to(device) for img in images)
    with torch.no_grad():
        predictions = model(images_gpu)

    idx = 0
    img_tensor = (images[idx] * 255).to(torch.uint8)
    pred = predictions[idx]

    scores = pred['scores'].cpu()
    keep = scores > threshold
    
    boxes = pred['boxes'][keep].cpu()
    labels_idx = pred['labels'][keep].cpu()

    if hasattr(base_ds, 'subset_breeds'):
        class_names = base_ds.subset_breeds
    elif hasattr(base_ds, 'classes'):
        class_names = base_ds.classes
    else:
        class_names = ["Pedestrian"] 

    display_labels = [class_names[i.item()-1] for i in labels_idx]

    if len(boxes) > 0:
        res = vutils.draw_bounding_boxes(
            img_tensor, 
            boxes=boxes, 
            labels=display_labels, 
            colors="red", 
            width=3
        )
        plt.imshow(res.permute(1, 2, 0))
        plt.title(f"Penn-Fudan R-CNN Detections")
        plt.legend().remove()
        plt.axis('off')
        plt.show()
    else:
        print("No objects detected above threshold.")