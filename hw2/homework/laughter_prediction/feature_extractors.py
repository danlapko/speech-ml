import os
import tempfile

import pandas as pd
import librosa
# import laughter_classification.psf_features as psf_features
import numpy as np
import scipy.io.wavfile as wav


class FeatureExtractor:
    def extract_features(self, wav_path):
        """
        Extracts features for classification ny frames for .wav file

        :param wav_path: string, path to .wav file
        :return: pandas.DataFrame with features of shape (n_chunks, n_features)
        """
        raise NotImplementedError("Should have implemented this")


class PyAAExtractor(FeatureExtractor):
    """Python Audio Analysis features extractor"""

    def __init__(self):
        self.extract_script = "laughter_prediction/extract_pyAA_features.py"
        self.py_env_name = "ipykernel_py2"

    def extract_features(self, wav_path):
        with tempfile.NamedTemporaryFile() as tmp_file:
            feature_save_path = tmp_file.name
            cmd = f"python \"{self.extract_script}\" --wav_path=\"{wav_path}\" --feature_save_path=\"{feature_save_path}\""
            # os.system(f"source activate {self.py_env_name}; {cmd}")
            os.system(f"{cmd}")

            feature_df = pd.read_csv(feature_save_path)

        print("PyAAExtractor:", feature_df.shape)
        return feature_df


class FBankExtractor(FeatureExtractor):
    """Python Audio Analysis features extractor"""

    def __init__(self, n_mels):
        self.n_mels = n_mels

    def extract_features(self, wav_path):
        y, sr = librosa.load(wav_path, sr=16000, dtype=float)
        mels = librosa.feature.melspectrogram(y, sr=sr, n_mels=self.n_mels,hop_length=512)
        mels = librosa.power_to_db(mels, np.max).T
        return pd.DataFrame(mels)


class MFCCExtractor(FeatureExtractor):
    """Python Audio Analysis features extractor"""

    def __init__(self, n_mfcc):
        self.n_mfcc = n_mfcc

    def extract_features(self, wav_path):
        y, sr = librosa.load(wav_path, sr=16000, dtype=float)
        mfcc = librosa.feature.mfcc(y, sr, n_mfcc=self.n_mfcc, hop_length=512).T
        return pd.DataFrame(mfcc)
