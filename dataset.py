# dataset.py
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

from config import (
    IMAGE_SIZE, BATCH_SIZE, SEED,
    TRAIN_RATIO, VAL_RATIO, EXCLUDE_TYPES
)


class MultiCancerDataset(Dataset):
    def __init__(self, root_dir, transform=None, exclude_types=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_names = []
        self.label_map = {}
        self.exclude_types = exclude_types or []

        cancer_types = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d)) and d.lower() not in self.exclude_types
        ])

        label_counter = 0
        for cancer_type in cancer_types:
            cancer_path = os.path.join(root_dir, cancer_type)
            class_folders = sorted([
                d for d in os.listdir(cancer_path)
                if os.path.isdir(os.path.join(cancer_path, d))
            ])

            for class_folder in class_folders:
                class_path = os.path.join(cancer_path, class_folder)
                image_files = [
                    f for f in os.listdir(class_path)
                    if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tif'))
                ]

                label_name = f"{cancer_type}_{class_folder}"
                self.label_map[label_name] = label_counter

                for img_name in image_files:
                    self.images.append(os.path.join(class_path, img_name))
                    self.labels.append(label_counter)

                self.class_names.append(label_name)
                label_counter += 1

        print(f" Total images loaded: {len(self.images)}")
        print(f" Total classes used: {len(self.class_names)}")
        print(f" Excluded types: {self.exclude_types}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


class TransformDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform


def prepare_dataloaders(root_dir, batch_size=BATCH_SIZE):
    full_dataset = MultiCancerDataset(root_dir, transform=None, exclude_types=EXCLUDE_TYPES)

    total_size = len(full_dataset)
    train_size = int(TRAIN_RATIO * total_size)
    val_size = int(VAL_RATIO * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_transform, val_transform = get_transforms()

    train_dataset = TransformDataset(train_dataset, train_transform)
    val_dataset = TransformDataset(val_dataset, val_transform)
    test_dataset = TransformDataset(test_dataset, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Training samples: {len(train_dataset)} ({train_size/total_size*100:.1f}%)")
    print(f"Validation samples: {len(val_dataset)} ({val_size/total_size*100:.1f}%)")
    print(f"Test samples: {len(test_dataset)} ({test_size/total_size*100:.1f}%)")

    return train_loader, val_loader, test_loader, full_dataset