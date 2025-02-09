import tensorflow as tf
from kapre.time_frequency import STFT, Magnitude, ApplyFilterbank, MagnitudeToDecibel

model = tf.keras.models.load_model('models/lstm.h5',
    custom_objects={'STFT':STFT,
                    'Magnitude':Magnitude,
                    'ApplyFilterbank':ApplyFilterbank,
                    'MagnitudeToDecibel':MagnitudeToDecibel})

print(model.summary())
