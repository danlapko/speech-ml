from model import Trainer
from util import extract_features_and_store_npa, create_scp_file, split, load


def create_and_store_npa(n_files=None):
    origin_dir = "/home/danlapko/speech-ml/hw3/data/VCTK-Corpus/wav48"  # VCTK-Corpus
    scp_file = "data/X.scp"
    npa_file = "data/X"

    if n_files is None:
        n_files = create_scp_file(origin_dir, scp_file)
    else:
        create_scp_file(origin_dir, scp_file)

    extract_features_and_store_npa(scp_file, n_files, npa_file)


def main_(n_audios_to_use, epoches):
    npa_file = "data/X"
    npa_file_noised = "data/X_noised"

    X = load(npa_file)[:n_audios_to_use]
    X, X_test = split(X)
    X_noised = load(npa_file_noised)[:n_audios_to_use]
    X_noised, X_test_noised = split(X_noised)

    trainer = Trainer(n_features=13, model_path="model_mfcc", hidden_size=10)
    trainer.train(X, X_noised, X_test, X_test_noised, epoches=epoches, batch_size=128)
    trainer.test(X_test, X_test_noised)


if __name__ == "__main__":
    n_audios_to_use = 10000
    epoches = 100
    # create_and_store_npa(n_files=n_audios_to_use)
    main_(n_audios_to_use, epoches)
