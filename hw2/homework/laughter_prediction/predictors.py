import glob

import numpy as np

from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


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
            self.model = GRUTagger(n_features=audios.shape[-1], hidden_size=10,
                                   n_classes=2).to(device)

        loss_function = nn.NLLLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=0.1)

        for epoch in range(epoches):
            for frames, tags in zip(audios, all_tags):
                self.model.zero_grad()
                hidden = self.model.init_hidden(batch_size=1)

                tags_ = self.model(frames, hidden)

                loss = loss_function(tags_, tags)
                loss.backward()
                optimizer.step()

    def predict(self, audio):
        batch_size = 1
        audio = torch.unsqueeze(torch.tensor(audio).to(device), batch_size)
        print("RnnPredictor.predict", audio.shape)

        if self.model is None:
            self.model = GRUTagger(n_features=audio.shape[-1], hidden_size=10,
                                   n_classes=2).to(device)

        with torch.no_grad():
            tags_probs = self.model(audio, self.model.init_hidden(batch_size))

        tags_probs = torch.squeeze(tags_probs)
        result = tags_probs >= 0.5
        return result[:, 1]

    def predict_proba(self, audio):
        batch_size = 1
        audio = torch.unsqueeze(torch.tensor(audio).to(device), batch_size)

        if self.model is None:
            self.model = GRUTagger(n_features=audio.shape[-1], hidden_size=10,
                                   n_classes=2).to(device)
        with torch.no_grad():
            tags_probs = self.model(audio, self.model.init_hidden(batch_size))

        return tags_probs


# seq of frames features -> seq of tags
class GRUTagger(nn.Module):
    def __init__(self, n_features, hidden_size, n_classes):
        super(GRUTagger, self).__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.k_classes = n_classes

        self.gru = nn.GRU(n_features, hidden_size)
        self.linear = nn.Linear(hidden_size, n_classes)

    def forward(self, audios, hidden):
        """

        :param audios: (audio_length, batch, n_features)
        :return:
        """
        print("GRUTagger.forward", audios.shape, hidden.shape, self.gru.input_size, self.gru.hidden_size)
        gru_out, hidden = self.gru(audios.float(), hidden)
        out = self.linear(gru_out)
        # tags_probs = F.log_softmax(out, dim=2)
        tags_probs = F.softmax(out, dim=2)
        return tags_probs

    def init_hidden(self, batch_size):
        """

        :return:(num_layers * num_directions, batch, hidden_size)
        """
        return torch.randn(1, batch_size, self.hidden_size, device=device).float()


