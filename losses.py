import torchaudio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T

class AudioLoss(nn.Module):
    def __init__(self,
                 weight_time=0.5,
                 weight_spec_conv=1.0,
                 weight_logmel=1.0,
                 n_fft=1024,
                 hop_length=256,
                 n_mels=80,
                 sample_rate=16000,
                 eps=1e-6):
        super().__init__()
        self.w1 = weight_time
        self.w2 = weight_spec_conv
        self.w3 = weight_logmel
        self.eps = eps

        # mel extractor
        self.mel_extractor = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            power=1.0,
            n_mels=n_mels,
        )
        # STFT window
        self.window = torch.hann_window(n_fft)

    def spectral_convergence(self, mag_pred, mag_true):
        num = torch.norm(mag_true - mag_pred, p='fro')
        den = torch.norm(mag_true, p='fro')
        return num / (den + self.eps)

    def forward(self, wav_pred, wav_true):
        """
        wav_pred, wav_true: [B, 1, T]
        returns: scalar loss
        """
        # 1) Time-domain L1
        l1_time = F.l1_loss(wav_pred, wav_true)

        # 2) STFT magnitude
        B, C, T = wav_true.shape
        window = self.window.to(wav_true.device)
        pred_stft = torch.stft(
            wav_pred.squeeze(1), n_fft=window.numel(), hop_length=self.mel_extractor.hop_length,
            win_length=window.numel(), window=window, return_complex=True
        )
        true_stft = torch.stft(
            wav_true.squeeze(1), n_fft=window.numel(), hop_length=self.mel_extractor.hop_length,
            win_length=window.numel(), window=window, return_complex=True
        )
        mag_pred = torch.abs(pred_stft)
        mag_true = torch.abs(true_stft)
        sc_loss = self.spectral_convergence(mag_pred, mag_true)

        # 3) Log-Mel L1
        mel_true = self.mel_extractor(wav_true)  # [B, n_mels, T']
        mel_pred = self.mel_extractor(wav_pred)
        logmel_pred = torch.log(mel_pred + self.eps)
        logmel_true = torch.log(mel_true + self.eps)
        l1_mel = F.l1_loss(logmel_pred, logmel_true)

        return self.w1 * l1_time + self.w2 * sc_loss + self.w3 * l1_mel
