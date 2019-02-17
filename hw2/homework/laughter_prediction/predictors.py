import glob
import os

import numpy as np

from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch import optim
from sklearn.metrics import roc_auc_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Predictor:
    """
    Wrapper class used for loading serialized model and
    using it in classification task.
    Defines unified interface for all inherited predictors.
    """

    def predict(self, X):
        """
        Predict target values of X given a model

        :param X: numpy.ndarray, dtype=float, shape=[n_samples, n_features]
        :return: numpy.array predicted classes
        """
        raise NotImplementedError("Should have implemented this")

    def predict_proba(self, X):
        """
        Predict probabilities of target class

        :param X: numpy.ndarray, dtype=float, shape=[n_samples, n_features]
        :return: numpy.array target class probabilities
        """
        raise NotImplementedError("Should have implemented this")


class XgboostPredictor(Predictor):
    """Parametrized wrapper for xgboost-based predictors"""

    def __init__(self, model_path, threshold, scaler=None):
        self.threshold = threshold
        self.clf = joblib.load(model_path)
        self.scaler = scaler

    def _simple_smooth(self, data, n=50):
        dlen = len(data)

        def low_pass(data, i, n):
            if i < n // 2:
                return data[:i]
            if i >= dlen - n // 2 - 1:
                return data[i:]
            return data[i - n // 2: i + n - n // 2]

        sliced = np.array([low_pass(data, i, n) for i in range(dlen)])
        sumz = np.array([np.sum(x) for x in sliced])
        return sumz / n

    def predict(self, X):
        y_pred = self.clf.predict_proba(X)
        ypreds_bin = np.where(y_pred[:, 1] >= self.threshold, np.ones(len(y_pred)), np.zeros(len(y_pred)))
        return ypreds_bin

    def predict_proba(self, X):
        X_scaled = self.scaler.fit_transform(X) if self.scaler is not None else X
        not_smooth = self.clf.predict_proba(X_scaled)[:, 1]
        return self._simple_smooth(not_smooth)


class StrictLargeXgboostPredictor(XgboostPredictor):
    """
    Predictor trained on 3kk training examples, using PyAAExtractor
    for input features
    """

    def __init__(self, threshold=0.045985743):
        XgboostPredictor.__init__(self, model_path="models/XGBClassifier_3kk_pyAA10.pkl",
                                  threshold=threshold, scaler=StandardScaler())


class RnnPredictor(Predictor):
    def __init__(self, model_path="rnn_model"):
        self.model_path = model_path
        self.model = None

    def save(self):
        torch.save(self.model.state_dict(), self.model_path)

    def train(self, audios_mfcc, audios_fbank, all_tags,
              audios_mfcc_test, audios_fbank_test, all_tags_test,
              batch_size=2, epoches=30, hidden_size=10):
        """
        :param audios: np.array shape=[n_audios, m_frames, k_features]
                        m_frames - individual for each audio, k_features - same for all audios and frames
        :param all_tags: np.array shape=[n_audios, m_frames, k_classes]
                        m_frames - individual for each audio, k_classes - same for all audios and frames
        :param epoches: int
        """
        _, _, mfcc_k_features = audios_mfcc.shape
        _, _, fbank_k_features = audios_fbank.shape
        n_audios, audio_length = all_tags.shape

        audios_mfcc = torch.tensor(audios_mfcc).to(device)
        audios_fbank = torch.tensor(audios_fbank).to(device)
        all_tags = torch.tensor(all_tags).to(device)

        audios_mfcc_test = torch.tensor(audios_mfcc_test).to(device)
        audios_fbank_test = torch.tensor(audios_fbank_test).to(device)
        all_tags_test = torch.tensor(all_tags_test).to(device)

        if self.model is None:
            self.model = GRUTagger(n_features_mfcc=mfcc_k_features, n_features_fbank=fbank_k_features,
                                   hidden_size=hidden_size).to(device)
        if self.model_path and os.path.isfile(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path))
            print("loading..")

        loss_function = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters())

        for epoch in range(epoches):
            # TEST
            with torch.no_grad():
                y_mfcc, y_global = self.model(audios_mfcc_test, audios_fbank_test)
                loss_mfcc = loss_function(y_mfcc, all_tags_test)
                loss_global = loss_function(y_global, all_tags_test)

            y_mfcc, y_global = y_mfcc.cpu().numpy(), y_global.cpu().numpy()
            y_true = all_tags_test.cpu().numpy()

            auc_mfcc = roc_auc_score(y_true, y_mfcc)
            auc_global = roc_auc_score(y_true, y_global)

            print("##########################################################################3")
            print(f"##### TEST: epoch {epoch} loss_mfcc {loss_mfcc.item():.5} loss {loss_global.item():.5} "
                  f"auc_mfcc {auc_mfcc:.3} auc_global {auc_global:.3} #####")
            print("##########################################################################3")

            # prepare data
            permutation = np.random.permutation(n_audios)

            audios_mfcc = audios_mfcc[permutation]
            audios_fbank = audios_fbank[permutation]
            all_tags = all_tags[permutation]

            # train by batches
            for idx in range(0, n_audios, batch_size):
                batch_X_mfcc = audios_mfcc[idx: idx + batch_size]
                batch_X_fbank = audios_fbank[idx: idx + batch_size]
                batch_y = all_tags[idx: idx + batch_size]

                self.model.zero_grad()
                y_mfcc, y_global = self.model(batch_X_mfcc, batch_X_fbank)

                loss_mfcc = loss_function(y_mfcc, batch_y)
                loss_global = loss_function(y_global, batch_y)

                loss_mfcc.backward(retain_graph=True)
                loss_global.backward()
                optimizer.step()
                if idx // batch_size % 20 == 0:
                    print(f"epoch {epoch} batch {idx} loss_mfcc {loss_mfcc.item():.3} loss {loss_global.item():.3}")

        self.save()

    def predict(self, audios_mfcc, audios_fbank):
        return NotImplementedError()

    def predict_proba(self, audios_mfcc, audios_fbank):
        audio_mfcc = torch.tensor(audios_mfcc).to(device)
        audio_fbank = torch.tensor(audios_fbank).to(device)

        if self.model is None:
            raise RuntimeError("model is not trained")

        with torch.no_grad():
            y_mfcc, y_global = self.model(audio_mfcc, audio_fbank)

        return y_mfcc.cpu().numpy(), y_global.cpu().numpy()


# seq of frames features -> seq of tags
class GRUTagger(nn.Module):
    def __init__(self, n_features_mfcc, n_features_fbank, hidden_size):
        super(GRUTagger, self).__init__()
        self.hidden_size = hidden_size

        self.mfcc_gru = nn.GRU(n_features_mfcc, hidden_size, num_layers=1, bidirectional=True, batch_first=True)

        self.mfcc_linear = nn.Linear(2 * hidden_size, 1)

        self.fbank_gru = nn.GRU(n_features_fbank, hidden_size, num_layers=1, bidirectional=True, batch_first=True)

        self.global_linear = nn.Linear(2 * hidden_size * 2, 1)

    def forward(self, audios_mfcc, audios_fbank):
        """
        :param audios_mfcc: (batch, audio_length, n_features)
        :param audios_fbank: (batch, audio_length, n_features)
        :return:
        """
        out_mfcc, hidden_mfcc = self.mfcc_gru(audios_mfcc.float())
        out_mfcc_n_classes = self.mfcc_linear(out_mfcc)

        out_fbank, hidden_fbank = self.fbank_gru(audios_fbank.float())
        out_n_classes = self.global_linear(torch.cat((out_mfcc, out_fbank), dim=-1))

        out_mfcc_probs = torch.sigmoid(out_mfcc_n_classes)
        out_probs = torch.sigmoid(out_n_classes)

        return torch.squeeze(out_mfcc_probs), torch.squeeze(out_probs)
