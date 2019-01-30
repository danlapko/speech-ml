# Speech augmentation

Now only .wav and .flac formats supported.

Example origin and noised file placed in "origin" and  "noised" folder.

HOWTO:

* install requirements: `pip install -r requirements.txt`
* put origin audio files into "origin" folder
* point the path to folder with noises (beeps_dir, bgmusic_dir variables in main.py). You can use the noise available at https://yadi.sk/d/ZR5JdkhO3SPoLN (bg_noise.tar.gz)
* run `main.py` and get augumented audio files in "noised" folder