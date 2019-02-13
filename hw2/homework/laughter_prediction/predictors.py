import numpy as np

from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

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
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.model = None
        if model_path is not None:
            self.model = torch.load(self.model_path)

    def save(self):
        torch.save(self.model.state_dict(), self.model_path)

    def train(self, audios, all_tags, epoches=100):
        """
        :param audios: shape=[n_audios, m_frames, k_features]
                        m_frames - individual for each audio, k_features - same for all audios and frames
        :param all_tags: shape=[n_audios, m_frames, k_classes]
                        m_frames - individual for each audio, k_classes - same for all audios and frames
        :param epoches: int
        """
        if self.model is None:
            self.model = GRUTagger(frame_features_size=audios.shape[0, 0], hidden_size=10,
                                   tagset_size=all_tags.shape[0, 0])

        loss_function = nn.NLLLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=0.1)

        for epoch in range(epoches):
            for frames, tags in zip(audios, all_tags):
                self.model.zero_grad()
                self.model.hidden = self.model.init_hidden()

                tags_ = self.model(frames)

                loss = loss_function(tags_, tags)
                loss.backward()
                optimizer.step()

    def predict(self, seq_of_features):
        with torch.no_grad():
            tags_probs = self.model(seq_of_features)
        return torch.argmax(tags_probs)

    def predict_proba(self, seq_of_features):
        with torch.no_grad():
            tags_probs = self.model(seq_of_features)
        return tags_probs


# seq of frames features -> seq of tags
class GRUTagger(nn.Module):
    def __init__(self, frame_features_size, hidden_size, tagset_size):
        super(GRUTagger, self).__init__()
        self.frame_features_size = frame_features_size
        self.hidden_size = hidden_size
        self.tagset_size = tagset_size

        self.gru = nn.GRU(frame_features_size, hidden_size)
        self.linear = nn.Linear(hidden_size, tagset_size)
        self.hidden = self.init_hidden()

    def forward(self, input_seq):
        gru_out, hidden = self.gru(input_seq, self.hidden)
        out = self.linear(gru_out)
        tags_probs = F.log_softmax(out)
        return tags_probs

    def init_hidden(self):
        return torch.randn(1, 1, self.hidden_size, device=device)
