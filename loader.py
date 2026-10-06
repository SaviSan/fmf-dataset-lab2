import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

class FMFDataset(Dataset):
    def __init__(self, pt_file, transform=None):
        d = torch.load(pt_file)
        self.images = d["images"]
        self.labels = d["labels"]
        self.class_names = d["class_names"]
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img = self.images[i]
        if self.transform:
            img = self.transform(img)
        return img, self.labels[i]

def create_dataloaders(root, batch_size=8):
    norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_ds = FMFDataset(f"{root}/processed/train.pt", transform=norm)
    val_ds = FMFDataset(f"{root}/processed/val.pt", transform=norm)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, train_ds.class_names

if __name__ == "__main__":
    train_loader, val_loader, class_names = create_dataloaders("./fmfdataset", batch_size=8)
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels: {labels}")
    print(f"Label names: {[class_names[l] for l in labels]}")