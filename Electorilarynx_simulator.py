import numpy as np
import pyworld as pw
import librosa
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm


class ElectrolarynxChunkedDataset:
    def __init__(self, target_f0=120.0, fs=16000, frame_period=5.0, add_buzz=True, robust_mode=True, debug=False,
                 data_dir=None, segment_seconds=4.0, hop_length=256, use_npy_simulated=True, prepare_segments=False,
                 dataloader_backup_path='./datasets/train_dataloader_dataset.pth'):
        # Simulation-related attributes
        self.target_f0 = target_f0
        self.fs = fs
        self.frame_period = frame_period
        self.add_buzz = add_buzz
        self.robust_mode = robust_mode
        self.debug = debug

        # Dataset-related attributes
        self.data_dir = data_dir
        self.dataloader_backup_path = dataloader_backup_path
        self.sample_rate = fs
        self.hop_length = hop_length
        self.segment_samples = int(segment_seconds * self.sample_rate)
        self.segment_frames = 251  # int((self.sample_rate * segment_seconds) // hop_length)
        self.use_npy_simulated = use_npy_simulated
        self.segments = []

        if prepare_segments and data_dir:
            self.segments = self._prepare_segments()
            self.save()
        else:
            self.load()

        if data_dir:
            print(f"Total segments: {len(self.segments)}")
        self.frame_period = int((1000 * hop_length) / self.sample_rate)

    # Simulation methods
    def _add_buzz(self, y, threshold=0.4, buzz_amplitude=0.1, max_extension_sec=0.1, shift_back_sec=0.01):
        length = len(y)
        mask = np.abs(y) > threshold

        mask_diff = np.diff(mask.astype(int))
        start_indices = np.where(mask_diff == 1)[0] + 1
        end_indices = np.where(mask_diff == -1)[0] + 1

        if len(end_indices) == 0 or (len(start_indices) > 0 and start_indices[0] > end_indices[0]):
            end_indices = np.concatenate([[0], end_indices])
        if len(start_indices) > 0 and (len(end_indices) < len(start_indices)):
            end_indices = np.concatenate([end_indices, [length]])

        new_mask = np.zeros_like(y, dtype=bool)
        for start, end in zip(start_indices, end_indices):
            extension = np.random.randint(0, int(max_extension_sec * self.fs))
            new_end = min(end + extension, length)
            new_mask[start:new_end] = True

        shift_samples = int(shift_back_sec * self.fs)
        shifted_mask = np.zeros_like(new_mask)
        shifted_mask[max(0, 0 - shift_samples):length - shift_samples] = new_mask[:length - max(0, shift_samples)]

        t = np.linspace(0, len(y) / self.fs, len(y), endpoint=False)
        excitation = buzz_amplitude * np.sin(2 * np.pi * self.target_f0 * t)

        y_with_excitation = y.copy()
        y_with_excitation[shifted_mask] += excitation[shifted_mask]

        return y_with_excitation

    def process_array(self, audio_array, output_path, return_params=False):
        x = audio_array.astype(np.float64)

        if self.robust_mode:
            current_target_f0 = np.random.uniform(90, 140)
            jitter_strength = np.random.uniform(0, 3)
            distortion_strength = np.random.uniform(0.65, 0.85)
            buzz_amplitude = np.random.uniform(0.015, 0.03)
            threshold = np.random.uniform(0.08, 0.15)
            shift_back_sec = np.random.uniform(0.08, 0.15)
        else:
            current_target_f0 = self.target_f0
            jitter_strength = 2.0
            distortion_strength = 0.7
            buzz_amplitude = 0.1
            threshold = 0.4
            shift_back_sec = 0.01

        f0, timeaxis = pw.harvest(x, self.fs, frame_period=self.frame_period)
        sp = pw.cheaptrick(x, f0, timeaxis, self.fs)
        ap = pw.d4c(x, f0, timeaxis, self.fs)

        f0 = np.full_like(f0, current_target_f0) + np.random.uniform(-jitter_strength, jitter_strength, size=f0.shape)
        y = pw.synthesize(f0, sp, ap, self.fs, frame_period=self.frame_period)

        if self.add_buzz:
            y = self._add_buzz(y, threshold=threshold, buzz_amplitude=buzz_amplitude, shift_back_sec=shift_back_sec)

        speech_mask = np.abs(y) > threshold
        y_distorted = y.copy()
        y_distorted[speech_mask] = np.sign(y[speech_mask]) * (np.abs(y[speech_mask]) ** distortion_strength)
        y = y_distorted

        y = np.sign(y) * (np.abs(y) ** distortion_strength)
        sf.write(output_path, y.astype(np.float32), self.fs)

        if return_params:
            return {
                "target_f0": current_target_f0,
                "jitter_strength": jitter_strength,
                "distortion_strength": distortion_strength,
                "buzz_amplitude": buzz_amplitude,
                "buzz_threshold": threshold,
                "shift_back_sec": shift_back_sec
            }

    # Dataset methods
    def _prepare_segments(self):
        segments = []
        files = [f for f in os.listdir(os.path.join(self.data_dir, 'MelInput')) if f.startswith('mel_input_') and f.endswith('.npy')]

        indices = [int(f.split('_')[-1].replace('.npy', '')) for f in files]
        indices = sorted(indices)

        for idx in tqdm(indices):
            F0_path = os.path.join(self.data_dir, f'GTf0/ground_truth_f0_{idx}.npy')
            filter_path = os.path.join(self.data_dir, f'GTF/ground_truth_filter_{idx}.npy')
            simulated_waveform_path = os.path.join(self.data_dir, f'SWF/simulated_waveform_{idx}.npy')
            target_waveform_path = os.path.join(self.data_dir, f'target/target_wav_{idx}.wav')

            if not os.path.exists(F0_path) or not os.path.exists(filter_path) or not os.path.exists(simulated_waveform_path) or not os.path.exists(target_waveform_path):
                continue

            GTf0 = np.load(F0_path)
            sp = np.load(filter_path)
            sim = np.load(simulated_waveform_path)
            target, _ = sf.read(target_waveform_path)

            if np.max(np.abs(sim)) > 0:
                sim = sim / np.max(np.abs(sim))
            if np.max(np.abs(target)) > 0:
                target = target / np.max(np.abs(target))

            total_samples = target.shape[0]
            num_segments = total_samples // self.segment_samples

            for seg_idx in num_segments:
                frame_start = seg_idx * self.segment_frames
                sample_start = seg_idx * self.segment_samples

                GTf0_seg = GTf0[frame_start:frame_start + self.segment_frames]
                sim_seg = sim[sample_start:sample_start + self.segment_samples]
                target_seg = target[sample_start:sample_start + self.segment_samples]
                f0, timeaxis = pw.harvest(sim_seg, self.sample_rate, frame_period=16.0)
                sp_seg = pw.cheaptrick(sim_seg, f0, timeaxis, self.sample_rate)

                segments.append({
                    'sim_waveform': sim_seg,
                    'target_waveform': target_seg,
                    'sample_idx': idx
                })

        return segments

    def __len__(self):
        return len(self.segments)

    def save(self):
        torch.save(self.segments, self.dataloader_backup_path)

    def load(self):
        self.segments = torch.load(self.dataloader_backup_path, weights_only=False)

    def __getitem__(self, idx):
        seg = self.segments[idx]
        sim_segment = torch.as_tensor(seg['sim_waveform']).float()
        target_segment = torch.as_tensor(seg['target_waveform']).float()
        filter_segment = torch.as_tensor(seg['sample_idx']).float()

        return {
            'sim_waveform': sim_segment,
            'target_waveform': target_segment,
            'sample_idx': filter_segment,
        }

    def create_dataloader(self, data_dir, output_dataloader_path, batch_size=8, num_workers=2):
        """
        Creates a dataloader from the dataset and saves it to the specified path.

        Args:
            data_dir (str): Path to the dataset directory.
            output_dataloader_path (str): Path to save the dataloader.
            batch_size (int): Batch size for the dataloader.
            num_workers (int): Number of workers for the dataloader.
        """
        dataset = ElectrolarynxProcessor(
            data_dir=data_dir,
            segment_seconds=4.0,
            prepare_segments=False,
            dataloader_backup_path=output_dataloader_path
        )

        print(f"Dataset size: {len(dataset)}")
        sample = dataset[0]

        for key, value in sample.items():
            try:
                print(f"{key}: {value.shape}")
            except:
                break

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True
        )

        # Save the dataloader to the specified path
        torch.save(dataloader, output_dataloader_path)
        print(f"Dataloader saved to {output_dataloader_path}")

        # Return the dataloader for immediate use if needed
        return dataloader

    def load_dataset(self, dataset_name, config_name):
        """
        Load a dataset using the Hugging Face `datasets` library.

        Args:
            dataset_name (str): The name of the dataset to load.
            config_name (str): The configuration name for the dataset.

        Returns:
            DatasetDict: The loaded dataset.
        """
        from datasets import load_dataset

        # Load the dataset
        dataset = load_dataset(dataset_name, config_name)

        # Print sample info
        print(dataset)

        # Access a sample (example: first training sample)
        sample = dataset['train'][0]

        # Get audio array and sampling rate
        audio_array = sample['audio']['array']
        sampling_rate = sample['audio']['sampling_rate']

        print(f"Audio array shape: {audio_array.shape}")
        print(f"Sampling rate: {sampling_rate}")

        return dataset


if __name__ == "__main__":
    data_dir = '/content/drive/MyDrive/electrolarynx_dataset/train/'
    output_dataloader_path = '/content/drive/MyDrive/dataloader_dataset.pth'

    processor = ElectrolarynxProcessor()
    dataset = processor.load_dataset("google/fleurs", "he_il")  # 'he_il' = Hebrew from FLEURS multilingual dataset
    processor.create_dataloader(data_dir, output_dataloader_path)


