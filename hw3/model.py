import os

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import r2_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")


class RNNDenoiser(nn.Module):
    def __init__(self, n_features, hidden_size):
        super(RNNDenoiser, self).__init__()
        self.hidden_size = hidden_size

        self.encoder_gru = nn.GRU(n_features, hidden_size, num_layers=2, bidirectional=True, batch_first=True)
        self.encoder_linear = nn.Linear(2 * hidden_size, 2 * hidden_size)

        self.decoder_gru = nn.GRU(2 * hidden_size, n_features, num_layers=2, bidirectional=True, batch_first=True)
        self.decoder_linear = nn.Linear(2 * n_features, n_features)

    def forward(self, X):
        """
        :param X: (batch, frames, n_features)
        :return:
        """
        enc_out, enc_hidden = self.encoder_gru(X)
        enc_out = self.encoder_linear(enc_out)
        # enc_out = F.leaky_relu(enc_out)

        dec_out, dec_hidden = self.decoder_gru(enc_out)
        dec_out = self.decoder_linear(dec_out)

        return torch.squeeze(dec_out)


class Trainer:
    def __init__(self, n_features, model_path="model", hidden_size=10):
        self.model_path = model_path

        self.model = RNNDenoiser(n_features, hidden_size).to(device)

        if os.path.isfile(model_path):
            self.model.load_state_dict(torch.load(model_path))
            print("model loaded")

        self.loss_function = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def test(self, X, X_noised):

        X = torch.tensor(X)
        X_noised = torch.tensor(X_noised)

        with torch.no_grad():
            Y_denoised = self.model(X_noised.to(device)).to(cpu)

            X = X.view(-1)
            X_noised = X_noised.view(-1)
            Y_denoised = Y_denoised.view(-1)

            loss_origin = self.loss_function(X_noised, X)
            loss_denoised = self.loss_function(Y_denoised, X)

            r2_origin = r2_score(X, X_noised)
            r2_denoised = r2_score(X, Y_denoised)

        print(f"\n###### TEST MSE loss: origin {loss_origin.item():.5} denoised {loss_denoised.item():.5} #######")
        print(f"######      R2 score: origin {r2_origin.item():.5} denoised {r2_denoised.item():.5} #######\n")
        return Y_denoised

    def train(self, X, X_noised, X_test, X_test_noised, batch_size=2, epoches=30):
        """
        :param X: np.array shape=[n_audios, n_frames, n_features]
        :param X_test: np.array shape=[n_audios, n_frames, n_features]
        :param epoches: int
        """
        print("\nSTARTING TRAINING\n")

        n_audios, n_frames, n_feats = X.shape

        X = torch.tensor(X)
        X_noised = torch.tensor(X_noised)

        for epoch in range(epoches):
            self.test(X_test, X_test_noised)

            # prepare data
            permutation = np.random.permutation(n_audios)
            X = X[permutation]
            X_noised = X_noised[permutation]

            # train by batches
            for idx in range(0, n_audios, batch_size):
                btch_X = X[idx: idx + batch_size].to(device)
                btch_X_noised = X_noised[idx: idx + batch_size].to(device)

                self.model.zero_grad()
                btch_Y = self.model(btch_X_noised)

                loss = self.loss_function(btch_Y, btch_X)

                loss.backward()
                self.optimizer.step()
                if idx // batch_size % 2 == 0:
                    print(f"epoch {epoch} batch {idx} loss_ {loss.item():.3} ")

            torch.save(self.model.state_dict(), self.model_path)
