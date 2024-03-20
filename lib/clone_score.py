import torch
import torchaudio
import torchaudio.transforms as T
import bittensor as bt
from lib.reward import score
import math


class CloneScore:
    def __init__(self, n_mels=128):
        self.n_mels = n_mels

    def extract_mel_spectrogram(self, file_path):
        waveform, sample_rate = torchaudio.load(file_path)
        mel_spectrogram_transform = T.MelSpectrogram(sample_rate=sample_rate, n_mels=self.n_mels)
        mel_spectrogram = mel_spectrogram_transform(waveform)
        return mel_spectrogram

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
        """
        Calculate decay score based on mse_score and a decay rate.

        Parameters:
        mse_score (float): The Mean Squared Error score.
        decay_rate (float): The rate of decay, controls how fast the score decreases.

        Returns:
        float: The decay score.
        """
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
        spec1, spec2 = self.pad_or_trim_to_same_length(spec1, spec2)

        # Calculate MSE
        mse_score = self.calculate_mse(spec1, spec2).item()
        bt.logging.info(f"MSE Score for Voice Cloning: {mse_score}")

        # Calculate Decay Score based on MSE
        decay_score = self.calculate_decay_score(mse_score, decay_rate)
        bt.logging.info(f"Decay Score for Voice Cloning: {decay_score}")

        try:
            nisqa_wer_score = score(file_path2, input_text)
        except Exception as e:
            print(f"Error calculating NISQA score inside compare_audio function : {e}")
            nisqa_wer_score = 0

        # Calculate Final Score considering Decay Score and NISQA score
        if nisqa_wer_score == 0:
            final_score = 0
        else:
            final_score = (decay_score + nisqa_wer_score) / 2
        bt.logging.info(f"Final Score for Voice Cloning: {final_score}")
        
        return final_score