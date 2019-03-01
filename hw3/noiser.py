import glob
import os
import random

import librosa
import soundfile as sf
import numpy as np
from itertools import chain

origin_dir = "/home/danlapko/speech-ml/hw3/data/VCTK-Corpus/wav48"
beeps_dir = "/home/danlapko/speech-ml/hw3/data/bg_noise/FRESOUND_BEEPS_gsm/train"
bgmusic_dir = "/home/danlapko/speech-ml/hw3/data/bg_noise/AUDIONAUTIX_MUSIC_gsm/train"
attenuation = 0.3


def load_audios(dir):
    full_names = glob.iglob(dir + "/**/*", recursive=True)
    result = dict()

    for full_name in full_names:
        if not full_name.endswith(('.wav', '.flac')):
            continue

        data, rate = sf.read(full_name)
        result[full_name] = (data, rate)

    return result


def write_audios(audios_dict):
    for full_name, (data, rate) in audios_dict.items():
        try:
            full_name = full_name.replace("VCTK-Corpus", "noised")
            if not os.path.exists(os.path.dirname(full_name)):
                os.makedirs(os.path.dirname(full_name))
            sf.write(full_name, data, rate)
        except Exception as e:
            print(full_name, data.shape, rate, data.dtype)
            raise e


def add_noise(audio, noise, rate, noise_rate, attenuation):
    noise = noise * attenuation

    if rate != noise_rate:
        noise = librosa.resample(noise, noise_rate, rate)
        noise_rate = rate

    result = audio.copy()
    if noise.shape[0] < result.shape[0]:
        times = (result.shape[0] + noise.shape[0] - 1) // noise.shape[0]
        noise = np.tile(noise, times)

    noise = noise[:result.shape[0]]
    result = result + noise
    return result


def noise_all(origins, noises):
    result = dict()
    for name in origins.keys():
        origin, rate = origins[name]
        noise_name = random.choice(list(noises.keys()))
        noise, noise_rate = noises[noise_name]

        result[name] = (add_noise(origin, noise, rate, noise_rate, attenuation), rate)
    return result


def main_():
    beeps = load_audios(beeps_dir)
    musics = load_audios(bgmusic_dir)

    noises = dict(beeps)
    noises.update(musics)

    noised_dir = origin_dir.replace("VCTK-Corpus", "noised")
    already_noised = set(os.listdir(noised_dir))

    for i, sub_dir in enumerate(os.listdir(origin_dir)):
        if sub_dir in already_noised:
            print("skipping", sub_dir)
            continue

        origins = load_audios(origin_dir + "/" + sub_dir)
        print(i, "noising", sub_dir, "...")
        noised = noise_all(origins, noises)
        print(i, "writing", sub_dir, "...")
        write_audios(noised)


if __name__ == "__main__":
    main_()
