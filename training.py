import tensorflow as tf
from keras import layers
from keras.layers import TimeDistributed, LayerNormalization
from keras.models import Model
from keras.regularizers import l2
from keras.utils import to_categorical
from kapre import STFT, Magnitude, MagnitudeToDecibel
from kapre.composed import get_melspectrogram_layer, get_log_frequency_spectrogram_layer
import os
import numpy as np
from glob import glob
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.io import wavfile

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, wav_paths, labels, sr, dt, n_classes,
                 batch_size=32, shuffle=True):
        self.wav_paths = wav_paths
        self.labels = labels
        self.sr = sr
        self.dt = dt
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.shuffle = True
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.wav_paths) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        wav_paths = [self.wav_paths[k] for k in indexes]
        labels = [self.labels[k] for k in indexes]

        # generate a batch of time data
        X = np.empty((self.batch_size, int(self.sr*self.dt), 1), dtype=np.float32)
        Y = np.empty((self.batch_size, self.n_classes), dtype=np.float32)

        for i, (path, label) in enumerate(zip(wav_paths, labels)):
            rate, wav = wavfile.read(path)
            X[i,] = wav.reshape(-1, 1)
            Y[i,] = to_categorical(label, num_classes=self.n_classes)

        return X, Y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.wav_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

sr = 16000 # sample rate of clean audio
dt = 1.0 # delta time in seconds to sample audio
input_shape = ((int)(sr*dt), 1) # 1 channels (!), maybe 1-sec audio signal, for an example.
batch_size = 16
data_dir = 'UrbanSound8K/sampled'
n_classes = len(os.listdir(data_dir))
i = get_melspectrogram_layer(input_shape=input_shape,
                                    n_mels=128,
                                    pad_end=True,
                                    n_fft=512,
                                    win_length=400,
                                    hop_length=160,
                                    sample_rate=sr,
                                    return_decibel=True,
                                    input_data_format='channels_last',
                                    output_data_format='channels_last',
                                    name='2d_convolution')
x = LayerNormalization(axis=2, name='batch_norm')(i.output)
x = TimeDistributed(layers.Reshape((-1,)), name='reshape')(x)
s = TimeDistributed(layers.Dense(64, activation='tanh'),
                    name='td_dense_tanh')(x)
x = layers.Bidirectional(layers.LSTM(32, return_sequences=True),
                            name='bidirectional_lstm')(s)
x = layers.concatenate([s, x], axis=2, name='skip_connection')
x = layers.Dense(64, activation='relu', name='dense_1_relu')(x)
x = layers.MaxPooling1D(name='max_pool_1d')(x)
x = layers.Dense(32, activation='relu', name='dense_2_relu')(x)
x = layers.Flatten(name='flatten')(x)
x = layers.Dropout(rate=0.2, name='dropout')(x)
x = layers.Dense(32, activation='relu',
                        activity_regularizer=l2(0.001),
                        name='dense_3_relu')(x)
o = layers.Dense(n_classes, activation='softmax', name='softmax')(x)
model = Model(inputs=i.input, outputs=o, name='long_short_term_memory')

# Compile the model
model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']) # if single-label classification

wav_paths = glob('{}/**'.format(data_dir), recursive=True)
wav_paths = [x.replace(os.sep, '/') for x in wav_paths if '.wav' in x]
classes = sorted(os.listdir(data_dir))
le = LabelEncoder()
le.fit(classes)
labels = [os.path.split(x)[0].split('/')[-1] for x in wav_paths]
labels = le.transform(labels)
wav_train, wav_val, label_train, label_val = train_test_split(wav_paths,
                                                                labels,
                                                                test_size=0.1,
                                                                random_state=0)
tg = DataGenerator(wav_train, label_train, sr, dt,
                    n_classes, batch_size=batch_size)
vg = DataGenerator(wav_val, label_val, sr, dt,
                    n_classes, batch_size=batch_size)
model.fit(tg, validation_data=vg,
            epochs=30, verbose=1)
model.save('models/lstm.h5')
