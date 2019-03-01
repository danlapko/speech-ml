import glob

import numpy as np
from kaldi.feat.mfcc import Mfcc, MfccOptions
from kaldi.feat.fbank import Fbank, FbankOptions
from kaldi.feat.mel import MelBanksOptions
from kaldi.matrix import SubVector, SubMatrix
from kaldi.util.options import ParseOptions
from kaldi.util.table import SequentialWaveReader, MatrixWriter, SequentialMatrixReader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale


def create_scp_file(dir, scp_file):
    filenames = glob.iglob(dir + "/**/*.wav", recursive=True)

    print("start writing scp_file...")
    n_files = 0
    with open(scp_file, "w") as f:
        for idx, name in enumerate(filenames):
            f.write(f"{idx} {name}\n")
            n_files += 1
    print("finished")
    return n_files


def extract_features_and_store_npa(scp_file, n_files, npa_file):
    max_frames = 1926
    n_feats = 13
    X = np.zeros((n_files, max_frames, n_feats), dtype=np.float32)

    usage = """Extract MFCC features.
               Usage:  example.py [opts...] <rspec> <wspec>
            """

    po = ParseOptions(usage)
    po.register_float("min-duration", 0.0,
                      "minimum segment duration")
    mfcc_opts = MfccOptions()
    mfcc_opts.frame_opts.samp_freq = 8000  # real 48000
    # mfcc_opts.mel_opts = MelBanksOptions(n_feats)
    mfcc_opts.register(po)

    opts = po.parse_args()
    rspec = "scp:" + scp_file

    # Create MFCC object and obtain sample frequency
    mfcc = Mfcc(mfcc_opts)
    sf = mfcc_opts.frame_opts.samp_freq

    print("starting extracting features...")
    i = 0
    with SequentialWaveReader(rspec) as reader:
        for key, wav in reader:
            if wav.duration < opts.min_duration:
                continue

            assert (wav.samp_freq >= sf)
            assert (wav.samp_freq % sf == 0)

            s = wav.data()

            # downsample to sf [default=8kHz]
            s = s[:, ::int(wav.samp_freq / sf)]

            # mix-down stereo to mono
            m = SubVector(np.mean(s, axis=0))

            # compute MFCC features
            f = mfcc.compute_features(m, sf, 1.0)

            # standardize features
            f = SubMatrix(scale(f))

            # write features to archive
            X[i, 0:f.shape[0]] = f.numpy()

            if i % 1000 == 0:
                print(key, f.shape)
            i += 1
            if i >= n_files:
                break

    np.save(npa_file, X)
    print("finished")


def load(npa_file):
    X = np.load(npa_file + ".npy")
    print(npa_file + ".npy file loaded", X.shape, X.dtype)
    return X


def split(X, k_test=0.2):
    X_train, X_test = train_test_split(X, test_size=k_test, random_state=1)
    print("data splited ")
    return X_train, X_test

# y, sr = librosa.load("f_m.wav", sr=16000, dtype=float)
