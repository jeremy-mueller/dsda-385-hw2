import torchvision
import torch
import os

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors

from torchvision.transforms.v2 import functional as F

class PennFudanDataset(Dataset):
    def __init__(self):
        self.imgs = list(sorted(os.listdir(os.path.join("PennFudanPed", "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join("PennFudanPed", "PedMasks"))))
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img_path = os.path.join("PennFudanPed", "PNGImages", self.imgs[idx])
        mask_path = os.path.join("PennFudanPed", "PedMasks", self.masks[idx])
        img = read_image(img_path)
        img = torchvision.transforms.Resize((512, 512), interpolation=torchvision.transforms.InterpolationMode.BILINEAR)(img)
        img = torchvision.transforms.v2.ToDtype(torch.float32, scale=True)(img)
        mask = read_image(mask_path)
        mask = torchvision.transforms.Resize((512, 512), interpolation=torchvision.transforms.InterpolationMode.NEAREST)(mask)
        mask = torchvision.transforms.v2.ToDtype(torch.float32, scale=True)(mask)

        object_ids = torch.unique(mask)
        object_ids = object_ids[1:]
        num_objects = len(object_ids)

        masks = (mask == object_ids[:, None, None]).to(dtype=torch.uint8)
        boxes = masks_to_boxes(masks)
        labels = torch.ones((num_objects,), dtype=torch.int64)

        img = tv_tensors.Image(img)

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img))
        target["labels"] = labels

        return img, target

class OxfordIIITDataset(torchvision.datasets.OxfordIIITPet):
    def __init__(self, root, **kwargs):
        super().__init__(root, split="trainval", target_types=["category", "segmentation"], download=True, **kwargs)
        
        test_ds = torchvision.datasets.OxfordIIITPet(root, split="test", target_types=["category", "segmentation"], download=True, **kwargs)
        self._images.extend(test_ds._images)
        self._labels.extend(test_ds._labels)
        self._segs.extend(test_ds._segs)
        
        self.subset_breeds = [
            "Bengal", "Egyptian Mau", "Persian", "Ragdoll", "Sphynx",
            "German Shorthaired", "Japanese Chin", "Chihuahua", "Yorkshire Terrier", "Beagle"
        ]

        self.breeds_to_idx = {name: i for i, name in enumerate(self.classes)}
        target_original_indices = [self.breeds_to_idx[name] for name in self.subset_breeds]

        self.subset_indices = [
            i for i, label in enumerate(self._labels) if label in target_original_indices
        ]

        self.new_label_map = {orig_idx: i for i, orig_idx in enumerate(target_original_indices)}

        final_indices = []
        print("Verifying masks...")
        for idx in self.subset_indices:
            _, (_, trimap) = super().__getitem__(idx)
            trimap_tensor = torchvision.transforms.v2.functional.to_image(trimap)
            if torch.any(trimap_tensor == 1):
                final_indices.append(idx)

        self.subset_indices = final_indices
        
        self.transform = torchvision.transforms.v2.Compose([
            torchvision.transforms.v2.ToImage(),
            torchvision.transforms.v2.Resize(size=(512, 512)),
            torchvision.transforms.v2.ToDtype(torch.float32, scale=True)
        ])

    def __len__(self):
        return len(self.subset_indices)
    
    def __getitem__(self, idx):
        actual_idx = self.subset_indices[idx]
        
        image, (label, trimap) = super().__getitem__(actual_idx)

        trimap_tensor = torchvision.transforms.v2.functional.to_image(trimap)
        mask = (trimap_tensor == 1).to(torch.uint8)
        
        boxes_coords = masks_to_boxes(mask) 

        new_label = self.new_label_map[label]

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(
            boxes_coords, 
            format="XYXY", 
            canvas_size=image.size[::-1]
        )
        target["labels"] = torch.tensor([new_label], dtype=torch.int64)

        if self.transform:
            image, target = self.transform(image, target)

        target["labels"] = target["labels"].reshape(-1)
        target["boxes"] = target["boxes"].reshape(-1, 4)

        return image, target

def collate_fn(batch):
    return tuple(zip(*batch))

def load(dataset, batch_size):
    if dataset == "pennfudan":
        full_dataset = PennFudanDataset()
        train_size = int(round(0.7 * len(full_dataset)))
        test_size = int(round(0.15 * len(full_dataset)))
        val_size = len(full_dataset) - train_size - test_size

        train_dataset, test_dataset, val_dataset = random_split(full_dataset, [train_size, test_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    elif dataset == "oxfordiiit":
        full_dataset = OxfordIIITDataset(root="")
        train_size = int(round(0.7 * len(full_dataset)))
        test_size = int(round(0.15 * len(full_dataset)))
        val_size = len(full_dataset) - train_size - test_size

        train_dataset, test_dataset, val_dataset = random_split(full_dataset, [train_size, test_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    else:
        raise ValueError(f"Dataset {dataset} not recognized. Please choose 'pennfudan' or 'oxfordiiit'.")
    return train_loader, test_loader, val_loader