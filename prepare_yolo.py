import torchvision
import torch
import numpy as np

from loader import load

penn_train, penn_test, penn_val = load("pennfudan", batch_size=8)
index = 0

for images, targets in penn_train:
    for i, image in enumerate(images):
        img_tensor = (image * 255).to(torch.uint8)
        # Now save
        torchvision.io.write_png(img_tensor, f"yolo/pennfudan/images/train/{index}.png")
        boxes = torchvision.ops.box_convert(targets[i]['boxes'], "xyxy", "cxcywh")
        boxes = boxes / 512.0
        pedestrians = targets[i]['labels']
        with open(f"yolo/pennfudan/labels/train/{index}.txt", "w") as f:
            for i, ped in enumerate(pedestrians):
                f.write(f"0 {boxes[i][0]} {boxes[i][1]} {boxes[i][2]} {boxes[i][3]}\n")
        index += 1

for images, targets in penn_test:
    for i, image in enumerate(images):
        img_tensor = (image * 255).to(torch.uint8)
        # Now save
        torchvision.io.write_png(img_tensor, f"yolo/pennfudan/images/test/{index}.png")
        boxes = torchvision.ops.box_convert(targets[i]['boxes'], "xyxy", "cxcywh")
        boxes = boxes / 512.0
        pedestrians = targets[i]['labels']
        with open(f"yolo/pennfudan/labels/test/{index}.txt", "w") as f:
            for i, ped in enumerate(pedestrians):
                f.write(f"0 {boxes[i][0]} {boxes[i][1]} {boxes[i][2]} {boxes[i][3]}\n")
        index += 1

for images, targets in penn_val:
    for i, image in enumerate(images):
        img_tensor = (image * 255).to(torch.uint8)
        # Now save
        torchvision.io.write_png(img_tensor, f"yolo/pennfudan/images/val/{index}.png")
        boxes = torchvision.ops.box_convert(targets[i]['boxes'], "xyxy", "cxcywh")
        boxes = boxes / 512.0
        pedestrians = targets[i]['labels']
        with open(f"yolo/pennfudan/labels/val/{index}.txt", "w") as f:
            for i, ped in enumerate(pedestrians):
                f.write(f"0 {boxes[i][0]} {boxes[i][1]} {boxes[i][2]} {boxes[i][3]}\n")
        index += 1

oxford_train, oxford_test, oxford_val = load("oxfordiiit", batch_size=1)
index = 0

for images, targets in oxford_train:
    for i, image in enumerate(images):
        img_tensor = (image * 255).to(torch.uint8)
        torchvision.io.write_png(img_tensor, f"yolo/oxfordiiit/images/train/{index}.png")
        boxes = torchvision.ops.box_convert(targets[i]['boxes'], "xyxy", "cxcywh")
        boxes = boxes / 512.0
        animals = targets[i]['labels']
        with open(f"yolo/oxfordiiit/labels/train/{index}.txt", "w") as f:
            for i, anim in enumerate(animals):
                f.write(f"{anim} {boxes[i][0]} {boxes[i][1]} {boxes[i][2]} {boxes[i][3]}\n")
        index += 1

for images, targets in oxford_test:
    for i, image in enumerate(images):
        img_tensor = (image * 255).to(torch.uint8)
        torchvision.io.write_png(img_tensor, f"yolo/oxfordiiit/images/test/{index}.png")
        boxes = torchvision.ops.box_convert(targets[i]['boxes'], "xyxy", "cxcywh")
        boxes = boxes / 512.0
        animals = targets[i]['labels']
        with open(f"yolo/oxfordiiit/labels/test/{index}.txt", "w") as f:
            for i, anim in enumerate(animals):
                f.write(f"{anim} {boxes[i][0]} {boxes[i][1]} {boxes[i][2]} {boxes[i][3]}\n")
        index += 1

for images, targets in oxford_val:
    for i, image in enumerate(images):
        img_tensor = (image * 255).to(torch.uint8)
        torchvision.io.write_png(img_tensor, f"yolo/oxfordiiit/images/val/{index}.png")
        boxes = torchvision.ops.box_convert(targets[i]['boxes'], "xyxy", "cxcywh")
        boxes = boxes / 512.0
        animals = targets[i]['labels']
        with open(f"yolo/oxfordiiit/labels/val/{index}.txt", "w") as f:
            for i, anim in enumerate(animals):
                f.write(f"{anim} {boxes[i][0]} {boxes[i][1]} {boxes[i][2]} {boxes[i][3]}\n")
        index += 1