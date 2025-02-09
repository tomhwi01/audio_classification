from scipy.io import wavfile
import os
from glob import glob
import numpy as np
import pandas as pd
from librosa.core import resample, to_mono
from tqdm import tqdm
import wavio

def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/20),
                       min_periods=1,
                       center=True).max()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask, y_mean

def downsample_mono(path, sr):
    try:
        obj = wavio.read(path)
    except Exception:
        return -1, None
    wav = obj.data.astype(np.float32, order='F')
    rate = obj.rate
    try:
        channel = wav.shape[1]
        if channel == 2:
            wav = to_mono(wav.T)
        elif channel == 1:
            wav = to_mono(wav.reshape(-1))
    except IndexError:
        wav = to_mono(wav.reshape(-1))
        pass
    except Exception as exc:
        raise exc
    wav = resample(wav, rate, sr)
    wav = wav.astype(np.int16)
    return sr, wav

def save_sample(sample, rate, target_dir, fn, ix):
    fn = fn.split('.wav')[0]
    dst_path = os.path.join(target_dir.split('.')[0], fn+'_{}.wav'.format(str(ix)))
    if os.path.exists(dst_path):
        return
    wavfile.write(dst_path, rate, sample)

def check_dir(path):
    if os.path.exists(path) is False:
        os.mkdir(path)

if __name__ == '__main__':
    source_dir = 'UrbanSound8K/audio'
    destination_dir = 'UrbanSound8K/sampled'
    sr = 16000 # rate to downsample audio
    dt = 1.0 # delta time in seconds to sample audio
    threshold = 20 # threshold magnitude for np.int16 dtype

    wav_paths = glob('{}/**'.format(source_dir), recursive=True)
    wav_paths = [x for x in wav_paths if '.wav' in x]
    dirs = os.listdir(source_dir)
    check_dir(destination_dir)
    classes = os.listdir(source_dir)
    for _cls in classes:
        target_dir = os.path.join(destination_dir, _cls)
        check_dir(target_dir)
        src_dir = os.path.join(source_dir, _cls)
        for fn in tqdm(os.listdir(src_dir)):
            src_fn = os.path.join(src_dir, fn)
            rate, wav = downsample_mono(src_fn, sr)
            if rate == -1:
                continue
            mask, y_mean = envelope(wav, rate, threshold=threshold)
            wav = wav[mask]
            delta_sample = int(dt*rate)

            # cleaned audio is less than a single sample
            # pad with zeros to delta_sample size
            if wav.shape[0] < delta_sample:
                sample = np.zeros(shape=(delta_sample,), dtype=np.int16)
                sample[:wav.shape[0]] = wav
                save_sample(sample, rate, target_dir, fn, 0)
            # step through audio and save every delta_sample
            # discard the ending audio if it is too short
            else:
                trunc = wav.shape[0] % delta_sample
                for cnt, i in enumerate(np.arange(0, wav.shape[0]-trunc, delta_sample)):
                    start = int(i)
                    stop = int(i + delta_sample)
                    sample = wav[start:stop]
                    save_sample(sample, rate, target_dir, fn, cnt)
