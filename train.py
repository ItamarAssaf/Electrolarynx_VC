
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torchaudio.transforms import MelSpectrogram
from IPython.display import Audio, display
from models import WaveformToWaveformModel
from losses import AudioLoss
from Electorilarynx_simulator import *


def playback_example(model, dataloader, device, rate=16000):
    model.eval()
    batch = next(iter(dataloader))
    src = batch['sim_waveform'].to(device)
    tgt = batch['target_waveform'].to(device)

    if src.dim() == 2:
        src = src.unsqueeze(1)
    if tgt.dim() == 2:
        tgt = tgt.unsqueeze(1)

    with torch.no_grad():
        out, mel, latent = model(src, tgt)

    # take first sample [T]
    src_np = src[0,0].cpu().numpy()
    tgt_np = tgt[0,0].cpu().numpy()
    out_np = out[0,0].cpu().numpy()

    print("🎵 Simulated electrolarynx:")
    display(Audio(src_np, rate=rate))
    print("🎯 Clean target:")
    display(Audio(tgt_np, rate=rate))
    print("🔁 Generated:")
    display(Audio(out_np, rate=rate))

    model.train()


def train_model(
    model,
    train_loader,
    test_dataloader=None,
    num_epochs=10,
    lr=1e-3,
    device='cuda',
    ckpt_path=None
):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    ) if test_dataloader else optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.9
    )
    criterion = AudioLoss().to(device)
    best_loss = float('inf')

    for epoch in range(1, num_epochs+1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            src = batch['sim_waveform'].to(device)
            tgt = batch['target_waveform'].to(device)
            if src.dim()==2: src = src.unsqueeze(1)
            if tgt.dim()==2: tgt = tgt.unsqueeze(1)

            out, mel, latent = model(src, tgt)
            loss = criterion(out, tgt)

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_train = total_loss / len(train_loader)
        print(f"Epoch {epoch}/{num_epochs} — Train Loss: {avg_train:.4f}")

        if test_dataloader:
            # validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in test_dataloader:
                    src = batch['sim_waveform'].to(device)
                    tgt = batch['target_waveform'].to(device)
                    if src.dim()==2: src = src.unsqueeze(1)
                    if tgt.dim()==2: tgt = tgt.unsqueeze(1)
                    out, mel, latent = model(src, tgt)
                    val_loss += criterion(out, tgt).item()
            avg_val = val_loss / len(test_dataloader)
            print(f"Epoch {epoch}/{num_epochs} — Val   Loss: {avg_val:.4f}")

            scheduler.step(avg_val)
            if ckpt_path and avg_val < best_loss:
                best_loss = avg_val
                torch.save(model.state_dict(), ckpt_path)
                print(f"💾 Saved best checkpoint: {ckpt_path}")

            # playback example
            playback_example(model, test_dataloader, device)

        else:
            scheduler.step()

    print("✅ Training complete!")
    return model
