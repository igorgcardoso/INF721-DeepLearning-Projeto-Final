from datetime import datetime

import torch
from dataset.prepare import prepare
from datasets import Audio, load_dataset
from model import Supressor
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
# from torchmetrics import audio
from tqdm import tqdm


def train(loader, model, criterion, optim, epochs = 100, metrics = []):
    writer = SummaryWriter(f'tensorboard/{datetime.now()}')
    best_loss = float('inf')
    for epoch in range(epochs):
        for idx, batch in enumerate(tqdm(loader['train'], desc=f"Epoch {epoch + 1}", leave=False)):
            step = epoch * len(loader['train']) + idx
            optim.zero_grad()
            x = batch["input"]
            y = batch["output"]
            x = x.cuda()
            y = y.cuda()
            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optim.step()
            writer.add_scalar("Loss/train", loss, step)
            for metric in metrics:
                writer.add_scalar(f"{metric.__name__}/train", metric(y_hat, clean), step)
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(loader['validation'], desc="Validating", leave=False):
                noisy = batch["input"]
                clean = batch["output"]
                noisy = noisy.cuda()
                clean = clean.cuda()
                y_hat = model(noisy)
                loss = criterion(y_hat, clean)
                val_loss += loss
            val_loss /= len(loader['validation'])
            writer.add_scalar("Loss/validation", val_loss, epoch)
            # writer.add_audio("Audio/validation", y_hat[0], epoch, sample_rate=16000)
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save({
                    'model': model.state_dict(),
                    'optim': optim.state_dict()
                }, "out/best.pt")
            for metric in metrics:
                writer.add_scalar(f"{metric.__name__}/validation", metric(y_hat, clean), epoch)


if __name__ == "__main__":
    model = Supressor(384, 6, 4, 1500).cuda()
    train_dataset = load_dataset('dataset/dataset.py', split='train')
    val_dataset = load_dataset('dataset/dataset.py', split='validation')
    train_dataset = train_dataset.cast_column("input", Audio(16000))
    train_dataset = train_dataset.cast_column("output", Audio(16000))
    val_dataset = val_dataset.cast_column("input", Audio(16000))
    val_dataset = val_dataset.cast_column("output", Audio(16000))
    train_dataset = train_dataset.map(prepare, batched=True, batch_size=32)
    val_dataset = val_dataset.map(prepare, batched=True, batch_size=32)
    train_dataset.set_format("torch", columns=["input", "output"])
    val_dataset.set_format("torch", columns=["input", "output"])
    criterion = F.mse_loss
    # metrics = [audio.SignalNoiseRatio(), audio.SignalDistortionRatio(), audio.ScaleInvariantSignalNoiseRatio()]
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1)
    dataloader = {
        'train': train_dataloader,
        'validation': val_dataloader
    }
    # train(dataloader, model, criterion, optim, metrics=metrics)
    train(dataloader, model, criterion, optim, 4)
