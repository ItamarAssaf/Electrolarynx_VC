import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pyworld")
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
import warnings
import pyworld
# Import necessary libraries
import torch
from torch.utils.data import DataLoader
from speechbrain.inference.vocoders import HIFIGAN
from vocos import Vocos
from IPython.display import Audio, display
from models import *
from losses import *
from Electorilarynx_simulator import *
from train import *
from Utiles import *

debug = False  # Set to True for debugging purposes

# Main script to run the training and inference
if __name__ == "__main__":

# Load the datasets

    train_dataloader = DataLoader(
        ElectrolarynxChunkedDataset(
            segment_seconds=4.0,
            prepare_segments=False,
            dataloader_backup_path='./datasets/train_dataloader_dataset.pth'
        ),
        batch_size=8, shuffle=True, num_workers=2, drop_last=True
    )

    test_dataloader = DataLoader(
        ElectrolarynxChunkedDataset(
            segment_seconds=4.0,
            prepare_segments=False,
            dataloader_backup_path='./datasets/test_dataloader_dataset.pth'
        ),
        batch_size=8, shuffle=True, num_workers=2, drop_last=True
    )

    if debug:
        # Example: Fetch a batch
        batch = next(iter(train_dataloader))
        pair_display(test_dataloader, sample_idx=0)

    # Set device       
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate each submodule
    latent_extractor = LatentExtractor().to(device)        # your voice embedding model
    waveform_to_mel = WaveformToMelLSTM().to(device)       # LSTM-based mel generator
    hifigan = HIFIGAN.from_hparams(                        # pre-trained vocoder
        source="speechbrain/tts-hifigan-ljspeech",
        savedir="pretrained_models/tts-hifigan-ljspeech",
        run_opts={"device":device}
    ).to(device)

    # Combine into one model
    model = WaveformToWaveformModel(
        latent_extractor=latent_extractor,
        waveform_to_mel=waveform_to_mel,
        hifigan=hifigan
    ).to(device)

    for param in model.hifigan.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.shape}")


    # Run training loop with AudioLoss (previously updated)
    trained_model = train_model(
        model=model,
        train_loader=test_dataloader, # Change to your train dataloader
        test_dataloader=test_dataloader,
        num_epochs=200,
        lr=1e-3,
        device=device,
        ckpt_path = './models/diffvc_best.pth'
    )
