import os
import random

import librosa
import soundfile as sf
import numpy as np

origin_dir = "origin"
noised_dir = "noised"  # out
beeps_dir = "bg_noise/FRESOUND_BEEPS_gsm/train"
bgmusic_dir = "bg_noise/AUDIONAUTIX_MUSIC_gsm/train"
attenuation = 0.5


def load_audios(dir):
    names = os.listdir(dir)

    result = dict()

    for name in names:

        if not name.endswith(('.wav', '.flac')):
            continue

        full_name = dir + "/" + name
        data, rate = sf.read(full_name)
        result[name] = (data, rate)

    return result


def write_audios(dir, audios_dict):
    for name, (data, rate) in audios_dict.items():
        full_name = dir + "/" + name
        sf.write(full_name, data, rate)


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
    origins = load_audios(origin_dir)

    noises = dict(beeps)
    noises.update(musics)

    noised = noise_all(origins, noises)
    write_audios(noised_dir, noised)


if __name__ == "__main__":
    main_()
