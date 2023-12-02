
from datetime import datetime

import safetensors
import torch
from dataset.dataset import AudioDataset
from model import Model
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
# from torchmetrics import audio
from tqdm import tqdm


def train(loader, model, criterion, optim, epochs = 100, metrics = []):
    writer = SummaryWriter(f'tensorboard/{datetime.now()}')
    for epoch in range(epochs):
        for x, y in tqdm(loader['train'], desc=f"Epoch {epoch + 1}", leave=False):
            optim.zero_grad()
            x = x.cuda()
            y = y.cuda()
            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optim.step()
            writer.add_scalar("Loss/train", loss, epoch)
            for metric in metrics:
                writer.add_scalar(f"{metric.__name__}/train", metric(y_hat, clean), epoch)
        best_loss = float('inf')
        val_loss = 0
        for batch in tqdm(loader['validation'].shuffle(), desc="Validating", leave=False):
            noisy = batch["input"]
            clean = batch["output"]
            noisy = noisy.to("cuda")
            clean = clean.to("cuda")
            y_hat = model(noisy)
            loss = criterion(y_hat, clean)
            val_loss += loss
        val_loss /= len(loader['validation'])
        if val_loss < best_loss:
            best_loss = val_loss
            safetensors.save(model.state_dict(), "out/model.safetensor")
            torch.save(optim.state_dict(), "out/optim.pt")
        writer.add_scalar("Loss/validation", loss, epoch)
        for metric in metrics:
            writer.add_scalar(f"{metric.__name__}/validation", metric(y_hat, clean), epoch)


if __name__ == "__main__":
    model = Model().cuda()
    train_dataset = AudioDataset('train')
    val_dataset = AudioDataset('val')
    criterion = F.mse_loss
    # metrics = [audio.SignalNoiseRatio(), audio.SignalDistortionRatio(), audio.ScaleInvariantSignalNoiseRatio()]
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32)
    dataloader = {
        'train': train_dataloader,
        'validation': val_dataloader
    }
    # train(dataloader, model, criterion, optim, metrics=metrics)
    train(dataloader, model, criterion, optim)
