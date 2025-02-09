import tensorflow as tf
from kapre.time_frequency import STFT, Magnitude, ApplyFilterbank, MagnitudeToDecibel
import numpy as np
import os
from sampling import downsample_mono, envelope
from pydub import AudioSegment

sr = 16000
dt = 1.0
input_shape = ((int)(sr*dt), 1)
batch_size = 16
data_dir = 'UrbanSound8K/audio'
threshold = 20
model = tf.keras.models.load_model('models/lstm.h5',
    custom_objects={'STFT':STFT,
                    'Magnitude':Magnitude,
                    'ApplyFilterbank':ApplyFilterbank,
                    'MagnitudeToDecibel':MagnitudeToDecibel})
classes = sorted(os.listdir(data_dir))
test_dir = 'UrbanSound8K/test/'
directory = os.fsencode(test_dir)
outputstr = ''
for file in os.listdir(directory):
    filepath = test_dir+os.fsdecode(file)
    if '.mp3' in filepath:
        if not os.path.exists(filepath.replace('.mp3','.wav')):
            sound = AudioSegment.from_mp3(filepath)
            filepath = filepath.replace('.mp3','.wav')
            sound.export(filepath, format="wav")
        else:
            filepath = filepath.replace('.mp3','.wav')
    rate, wav = downsample_mono(filepath, sr)
    mask, env = envelope(wav, rate, threshold=threshold)
    clean_wav = wav[mask]
    step = int(sr*dt)
    batch = []

    for i in range(0, clean_wav.shape[0], step):
        sample = clean_wav[i:i+step]
        sample = sample.reshape(-1, 1)
        if sample.shape[0] < step:
            tmp = np.zeros(shape=(step, 1), dtype=np.float32)
            tmp[:sample.shape[0],:] = sample.flatten().reshape(-1, 1)
            sample = tmp
        batch.append(sample)
    X_batch = np.array(batch, dtype=np.float32)
    y_pred = model.predict(X_batch)
    y_mean = np.mean(y_pred, axis=0)
    y_pred = np.argmax(y_mean)
    outputstr += 'In the path \"{}\", Predicted class: {}\n'.format(filepath, classes[y_pred])
print(outputstr)
