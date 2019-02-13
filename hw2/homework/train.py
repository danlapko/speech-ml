import numpy as np
import glob

from hw2.homework.laughter_classification.sspnet_data_sampler import SSPNetDataSampler
from hw2.homework.laughter_prediction.feature_extractors import FBankExtractor, MFCCExtractor
from hw2.homework.laughter_prediction.predictors import RnnPredictor


def prepare_dataset():
    filenames = glob.glob("./vocalizationcorpus/data/*")

    X = np.zeros((2763, 344, 13), dtype=np.float64)
    skips = []
    extractor = MFCCExtractor()
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
    y = np.zeros((2763, 344), dtype=np.float64)
    for idx, row in labels.iterrows():
        if idx in skips:
            continue
        y[idx] = 0
        print(idx, end=" ")
        for i in ('0', '1', '2', '3', '4', '5'):
            if row["type_voc_" + i] == 'laughter':
                a, b = row["start_voc_" + i], row["end_voc_" + i]
                a = a / 11 * 344
                b = b / 11 * 344
                a, b = int(a), int(b)
                print((a, b), end=" ")
                y[idx, a:b] = 1
        print()
    np.save("y", y)


def load_X_y():
    X = np.load("X_mfcc.npy")
    y = np.load("y.npy")
    return X, y


def train():
    # prepare_dataset()
    X, y = load_X_y()
    y_oh = np.eye(2)[y.astype(int)]  # one hot
    predictor = RnnPredictor()
    predictor.train(X, y_oh, epoches=1)


if __name__ == "__main__":
    train()
