import time
import torch
import torch.nn as nn
from torchvision import models
from loader import create_dataloaders

def run(bs, epochs=5):
    torch.manual_seed(42)
    train_loader, val_loader, names = create_dataloaders("./fmfdataset", batch_size=bs)
    model = models.mobilenet_v2(weights=None, num_classes=len(names))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    x0, _ = next(iter(train_loader))
    batch_mb = x0.element_size() * x0.nelement() / 1024 / 1024

    start = time.time()
    losses = []
    for ep in range(epochs):
        model.train()
        for x, y in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
    t = time.time() - start

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in val_loader:
            correct += (model(x).argmax(1) == y).sum().item()
            total += len(y)

    print(f"batch_size={bs}")
    print(f"  one input batch = {batch_mb:.1f} MB")
    print(f"  train time = {t:.1f} s")
    print(f"  loss_std = {torch.tensor(losses).std():.3f}")
    print(f"  final loss = {losses[-1]:.3f}")
    print(f"  val_acc = {correct}/{total}")

if __name__ == "__main__":
    run(8)
    run(64)