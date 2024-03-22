import torch
import torchaudio
import torchaudio.transforms as T
import bittensor as bt
from lib.reward import score
import math

class CloneScore:
    def __init__(self, n_mels=64):  # Reduced number of Mel frequency bins
        self.n_mels = n_mels

    def load_and_normalize_waveform(self, file_path):
        waveform, sample_rate = torchaudio.load(file_path)
        # Normalize waveform amplitude
        waveform = waveform / torch.max(torch.abs(waveform))
        return waveform, sample_rate

    def extract_mel_spectrogram(self, file_path):
        waveform, sample_rate = self.load_and_normalize_waveform(file_path)
        # Assuming you have a function to trim silence - replace with actual function
        waveform = self.trim_silence(waveform)  # Implement this method based on your needs
        mel_spectrogram_transform = T.MelSpectrogram(sample_rate=sample_rate, n_mels=self.n_mels)
        mel_spectrogram = mel_spectrogram_transform(waveform)
        db_transform = T.AmplitudeToDB()
        mel_spectrogram_db = db_transform(mel_spectrogram)
        return mel_spectrogram_db
    
    
    def pad_or_trim_to_same_length(self, spec1, spec2):
        if spec1.size(2) > spec2.size(2):
            padding_size = spec1.size(2) - spec2.size(2)
            spec2 = torch.nn.functional.pad(spec2, (0, padding_size))
        elif spec2.size(2) > spec1.size(2):
            padding_size = spec2.size(2) - spec1.size(2)
            spec1 = torch.nn.functional.pad(spec1, (0, padding_size))
        return spec1, spec2

    def calculate_mse(self, spec1, spec2):
        return torch.mean((spec1 - spec2) ** 2)

    def calculate_decay_score(self, mse_score, decay_rate):
        decay_score = math.exp(-decay_rate * mse_score)
        return decay_score

    def compare_audio(self, file_path1, file_path2, input_text, decay_rate):
        # Extract Mel Spectrograms
        try:
            print("Extracting Mel spectrograms...")
            print("File 1:", file_path1)
            print("File 2:", file_path2)
            print("Input Text:", input_text)
            spec1 = self.extract_mel_spectrogram(file_path1)
            spec2 = self.extract_mel_spectrogram(file_path2)
        except Exception as e:
            print(f"Error extracting Mel spectrograms: {e}")
            spec1 = spec2 = None

        # Pad or Trim
        if spec1 is not None and spec2 is not None:
            spec1, spec2 = self.pad_or_trim_to_same_length(spec1, spec2)

            # Calculate MSE
            mse_score = self.calculate_mse(spec1, spec2).item()
            bt.logging.info(f"MSE Score for Voice Cloning: {mse_score}")

            # Calculate Decay Score based on MSE
            decay_score = self.calculate_decay_score(mse_score, decay_rate)
            bt.logging.info(f"Decay Score for Voice Cloning: {decay_score}")
        else:
            mse_score = float('inf')  # Assigning a default high value if spectrograms extraction failed
            decay_score = 0

        try:
            nisqa_wer_score = score(file_path2, input_text)
        except Exception as e:
            print(f"Error calculating NISQA score inside compare_audio function: {e}")
            nisqa_wer_score = 0

        # Calculate Final Score considering Decay Score and NISQA score
        if nisqa_wer_score == 0 or decay_score == 0:
            final_score = 0
        else:
            final_score = (decay_score + nisqa_wer_score) / 2
        bt.logging.info(f"Final Score for Voice Cloning: {final_score}")

        return final_score