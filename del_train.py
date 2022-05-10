import tensorflow as tf

import data_generator
import globals
from model import naml
from data_generator import AdressaIterator


def train(model:tf.keras.model.Model, training_data, val_data):
    model.fit(x=training_data['x'],
              y=training_data['y'],
              batch_size=globals.learning_params['batch_size'],
              epochs=globals.learning_params['epochs'],
              validation_data=(val_data['x'], val_data['y']),
              # callbacks=[WandbCallback()],
              )

    return model