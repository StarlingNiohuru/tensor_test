import sys
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense
import numpy as np


class Seq2SeqAttention(object):

    def __init__(self, batch_size=64, epochs=100, latent_dim=256, num_samples=10000, validation_split=0.2,
                 data_path=None, model_path=None):
        self.batch_size = batch_size  # Batch size for training.
        self.epochs = epochs  # Number of epochs to train for.
        self.latent_dim = latent_dim  # Latent dimensionality of the encoding space.
        self.num_samples = num_samples  # Number of samples to train on.
        self.validation_split = validation_split

    def build_training_model(self):
        pass
