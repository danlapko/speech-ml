import numpy as np
import glob

from hw2.homework.laughter_classification.sspnet_data_sampler import SSPNetDataSampler
from hw2.homework.laughter_prediction.feature_extractors import FBankExtractor, MFCCExtractor
from hw2.homework.laughter_prediction.predictors import RnnPredictor


def prepare_fbank():
    filenames = glob.glob("./vocalizationcorpus/data/*")

    n_features = 96
    n_frames = 344

    X = np.zeros((2763, n_frames, n_features), dtype=np.float64)
    skips = []
    extractor = FBankExtractor(n_features)
    for idx, name in enumerate(filenames):
        fs = extractor.extract_features(name)
        print(idx, fs.shape)
        if fs.shape[0] != X.shape[1]:
            skips.append(idx)
            print("skipping", idx)
            continue
        X[idx] = fs
    np.save("X_fbank", X)
    print(skips)

    labels = SSPNetDataSampler.read_labels("./vocalizationcorpus/labels.txt")
    print(labels.columns)
    y = np.zeros((2763, n_frames), dtype=np.float64)
    for idx, row in labels.iterrows():
        if idx in skips:
            continue
        y[idx] = 0
        print(idx, end=" ")
        for i in ('0', '1', '2', '3', '4', '5'):
            if row["type_voc_" + i] == 'laughter':
                a, b = row["start_voc_" + i], row["end_voc_" + i]
                a = a / 11 * n_frames
                b = b / 11 * n_frames
                a, b = int(a), int(b)
                print((a, b), end=" ")
                y[idx, a:b] = 1
        print()
    np.save("y_fbank", y)


def prepare_mfcc():
    filenames = glob.glob("./vocalizationcorpus/data/*")

    n_features = 13
    n_frames = 344

    X = np.zeros((2763, n_frames, n_features), dtype=np.float)
    skips = []
    extractor = MFCCExtractor(n_features)
    for idx, name in enumerate(filenames):
        fs = extractor.extract_features(name)
        print(idx, fs.shape)
        if fs.shape[0] != X.shape[1]:
            skips.append(idx)
            print("skipping", idx)
            continue
        X[idx] = fs
    np.save("X_mfcc", X)
    print(skips)

    labels = SSPNetDataSampler.read_labels("./vocalizationcorpus/labels.txt")
    print(labels.columns)
    y = np.zeros((2763, n_frames), dtype=np.float)
    for idx, row in labels.iterrows():
        if idx in skips:
            continue
        y[idx] = 0
        print(idx, end=" ")
        for i in ('0', '1', '2', '3', '4', '5'):
            if row["type_voc_" + i] == 'laughter':
                a, b = row["start_voc_" + i], row["end_voc_" + i]
                a = a / 11 * n_frames
                b = b / 11 * n_frames
                a, b = int(a), int(b)
                print((a, b), end=" ")
                y[idx, a:b] = 1
        print()
    np.save("y_mfcc", y)


def load_X_y_fbank():
    X = np.load("X_fbank.npy")
    y = np.load("y_fbank.npy")
    return X.astype(np.float32), y.astype(np.float32)


def load_X_y_mfcc():
    X = np.load("X_mfcc.npy")
    y = np.load("y_mfcc.npy")
    return X.astype(np.float32), y.astype(np.float32)


def frames_to_interval(tags):
    changing = np.where(tags[:-1] != tags[1:])[0]
    changing = changing / 344 * 11
    return changing


def train():
    n_for_test = 400
    X_mfcc, y = load_X_y_mfcc()
    X_fbank, y = load_X_y_fbank()

    good_idxs = np.where(y.sum(axis=1) >= 10)[0]
    print(good_idxs.shape)

    X_mfcc, X_mfcc_test = X_mfcc[n_for_test:], X_mfcc[:n_for_test]
    X_fbank, X_fbank_test = X_fbank[n_for_test:], X_fbank[:n_for_test]
    y, y_test = y[n_for_test:], y[:n_for_test]

    X_mfcc = np.vstack(
        (X_mfcc, X_mfcc[good_idxs], X_mfcc[good_idxs], X_mfcc[good_idxs], X_mfcc[good_idxs], X_mfcc[good_idxs]))
    X_fbank = np.vstack(
        (X_fbank, X_fbank[good_idxs], X_fbank[good_idxs], X_fbank[good_idxs], X_fbank[good_idxs], X_fbank[good_idxs]))
    y = np.vstack((y, y[good_idxs], y[good_idxs], y[good_idxs], y[good_idxs], y[good_idxs]))

    predictor = RnnPredictor("rnn_model")
    predictor.train(X_mfcc, X_fbank, y,
                    X_mfcc_test, X_fbank_test, y_test,
                    batch_size=8, epoches=30, hidden_size=10)


def evaluate(file_idx):
    file_idx -= 1
    X_mfcc, y = load_X_y_mfcc()
    X_fbank, y = load_X_y_fbank()
    y_oh = np.eye(2, dtype=np.float32)[y.astype(int)]  # one hot

    audio_mfcc, tags = X_mfcc[file_idx], y_oh[file_idx]
    audio_fbank, tags = X_fbank[file_idx], y_oh[file_idx]

    predictor = RnnPredictor("rnn_model")

    tags_mfcc, tags_global = predictor.predict(audio_mfcc, audio_fbank)

    print(file_idx, frames_to_interval(tags[:, 1]))

    return frames_to_interval(tags_mfcc), frames_to_interval(tags_global)


def evaluate_proba(file_idx):
    file_idx -= 1
    X_mfcc, y = load_X_y_mfcc()
    X_fbank, y = load_X_y_fbank()
    y_oh = np.eye(2, dtype=np.float32)[y.astype(int)]  # one hot

    audio_mfcc, tags = X_mfcc[file_idx], y_oh[file_idx]
    audio_fbank, tags = X_fbank[file_idx], y_oh[file_idx]

    predictor = RnnPredictor("rnn_model")

    tags_mfcc, tags_global = predictor.predict_proba(audio_mfcc, audio_fbank)

    return tags_mfcc, tags_global


if __name__ == "__main__":
    # prepare_fbank()
    # prepare_mfcc()

    train()
