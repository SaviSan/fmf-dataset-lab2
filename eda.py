import torch, random
import matplotlib.pyplot as plt

tr = torch.load("fmfdataset/processed/train.pt")
va = torch.load("fmfdataset/processed/val.pt")
names = tr["class_names"]

print("train:", tr["images"].shape, "val:", va["images"].shape)

tr_u, tr_c = torch.unique(tr["labels"], return_counts=True)
va_u, va_c = torch.unique(va["labels"], return_counts=True)
print("train counts:", dict(zip(tr_u.tolist(), tr_c.tolist())))
print("val counts:", dict(zip(va_u.tolist(), va_c.tolist())))

# bar chart
x = range(len(names))
plt.bar([i - 0.2 for i in x], tr_c.tolist(), width=0.4, label="train")
plt.bar([i + 0.2 for i in x], va_c.tolist(), width=0.4, label="val")
plt.xticks(x, names)
plt.legend()
plt.title("Class counts")
plt.savefig("class_counts.png")
plt.show()

# 16 random images
idx = random.sample(range(len(tr["images"])), 16)
fig, axes = plt.subplots(4, 4, figsize=(10, 10))
for a, i in zip(axes.flatten(), idx):
    a.imshow(tr["images"][i].permute(1, 2, 0).clamp(0, 1))
    a.set_title(names[tr["labels"][i]])
    a.axis("off")
plt.tight_layout()
plt.savefig("sample_grid.png")
plt.show()