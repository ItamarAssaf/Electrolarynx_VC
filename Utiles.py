import soundfile as sf
import IPython.display as ipd
import json
from tqdm import tqdm
import os
import soundfile as sf
import numpy as np


def extract_mel_spectrogram(y, sr, n_mels=80, n_fft=1024, hop_length=256):
    """
    Extracts a mel-spectrogram from a waveform.
    """
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, power=1.0
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db

def extract_ground_truth_f0(y, fs, frame_period=5.0):
    """
    Extracts the ground-truth F0 curve from a waveform using WORLD vocoder.

    Args:
        y (np.ndarray): Input waveform.
        fs (int): Sampling rate.
        frame_period (float): Frame period in ms (default 5ms).

    Returns:
        np.ndarray: Extracted F0 curve (frames,)
    """
    # Ensure the input is float64
    y = y.astype(np.float64)

    # Use WORLD harvest to extract F0
    f0, timeaxis = pw.harvest(y, fs, frame_period=frame_period)

    return f0

def normalize_audio(y):
    return y / np.max(np.abs(y)) if np.max(np.abs(y)) > 0 else y

def prepare_dataset(samples, save_dir, simulator):
    """
    Processes a list of natural samples and saves (input, target) pairs for training.
    Now saves also simulated waveforms as .npy, and saves everything directly to Google Drive.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    log_data = {}  # Collect hyperparameter logs

    os.makedirs(os.path.join(save_dir, 'target'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'GTf0'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'GTF'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'ELS'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'SWF'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'MelInput'), exist_ok=True)


    for idx, sample in enumerate(tqdm(samples, desc="Preparing dataset")):
        audio_array = sample['audio']['array']
        audio_array = normalize_audio(audio_array)
        sampling_rate = sample['audio']['sampling_rate']

        # Save target natural audio
        target_path = os.path.join(save_dir, f'target/target_wav_{idx}.wav')
        sf.write(target_path, audio_array, samplerate=sampling_rate)

        # Extract ground-truth F0 and filter
        f0, timeaxis = pw.harvest(audio_array, sampling_rate, frame_period=5.0)
        sp = pw.cheaptrick(audio_array, f0, timeaxis, sampling_rate)
        ap = pw.d4c(audio_array, f0, timeaxis, sampling_rate)

        # Save ground-truth F0
        gt_f0_path = os.path.join(save_dir, f'GTf0/ground_truth_f0_{idx}.npy')
        np.save(gt_f0_path, f0)

        # Save ground-truth filter (spectral envelope)
        gt_filter_path = os.path.join(save_dir, f'GTF/ground_truth_filter_{idx}.npy')
        np.save(gt_filter_path, sp)

        # Generate EL-like (buzzed) speech
        simulated_path = os.path.join(save_dir, f'ELS/el_simulated_{idx}.wav')
        params = simulator.process_array(audio_array, simulated_path, return_params=True)

        # Load simulated waveform
        y_simulated, sr_simulated = sf.read(simulated_path)

        # Save simulated waveform as .npy for faster training later
        simulated_npy_path = os.path.join(save_dir, f'SWF/simulated_waveform_{idx}.npy')
        np.save(simulated_npy_path, y_simulated)

        # Extract mel-spectrogram from simulated
        mel = extract_mel_spectrogram(y_simulated, sr_simulated)

        # Save mel-spectrogram
        mel_path = os.path.join(save_dir, f'MelInput/mel_input_{idx}.npy')
        np.save(mel_path, mel)

        # Save hyperparameters for this sample
        log_data[f'sample_{idx}'] = params

    # Save the hyperparameter log to a JSON file
    with open(os.path.join(save_dir, 'generation_log.json'), 'w') as f:
        json.dump(log_data, f, indent=4)

    print(f"\n✅ Done! Saved {len(samples)} samples and full parameter log to {save_dir}")

def pair_display(dataloader, sample_idx=0):
    """
    Displays a pair of audio files: the original and the simulated waveform.
    """
    # Pick a random sample from batch (e.g., index 0)

    batch = next(iter(dataloader))
    # Get the target waveform and simulated waveform
    target_wave = batch['target_waveform'][sample_idx].cpu().numpy()
    simulated_wave = batch['sim_waveform'][sample_idx].cpu().numpy()



    # Play the clean target
    print("🎵 Clean target speech:")
    sf.write('target.wav', target_wave, 16000)
    ipd.display(ipd.Audio(target_wave, rate=16000))

    # Play the simulated electrolarynx-like speech
    print("🎵 Simulated electrolarynx speech:")
    sf.write('simulated.wav', simulated_wave, 16000)
    ipd.display(ipd.Audio(simulated_wave, rate=16000))
