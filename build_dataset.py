import torch, os
from PIL import Image
from torchvision import transforms as T
from split_data import get_all_paths, split_data
from augment import Augmenter

TARGET = 50         
aug = Augmenter()
val_tf = T.Compose([T.Resize((224, 224)), T.ToTensor()])

paths, labels, class_names = get_all_paths()
tr_p, tr_l, va_p, va_l = split_data(paths, labels)

# train: eka class ekata 50 wenakan augment karanawa
imgs, labs, ps = [], [], []
for c in sorted(set(tr_l)):
    cls_paths = [p for p, l in zip(tr_p, tr_l) if l == c]
    for i in range(TARGET):
        p = cls_paths[i % len(cls_paths)]        
        imgs.append(aug.get_random_augs(p, 1)[0])
        labs.append(c)
        ps.append(p)

torch.save({
    "images": torch.stack(imgs),
    "labels": torch.tensor(labs),
    "paths": ps,
    "class_names": class_names
}, "fmfdataset/processed/train.pt")

# val (augmentation na)
imgs, labs, ps = [], [], []
for p, l in zip(va_p, va_l):
    imgs.append(val_tf(Image.open(p).convert("RGB")))
    labs.append(l)
    ps.append(p)

torch.save({
    "images": torch.stack(imgs),
    "labels": torch.tensor(labs),
    "paths": ps,
    "class_names": class_names
}, "fmfdataset/processed/val.pt")

# report ekata
for name in ["train", "val"]:
    f = f"fmfdataset/processed/{name}.pt"
    d = torch.load(f)
    print(name, "images:", d["images"].shape, "labels:", d["labels"].shape)
    print("  per class:", torch.unique(d["labels"], return_counts=True))
    print("  size MB:", round(os.path.getsize(f) / 1024 / 1024, 2))