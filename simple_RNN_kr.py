import os

from keras import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Embedding, LSTM, Dense
import numpy as np
from keras.models import load_model


class SimpleRNNDataLoader(object):
    def __init__(self, text_path, vocab_path=None, embedding_path=None, batch_size=64, sequence_length=20,
                 token_step=3):
        self.text_path = text_path
        self.vocab_path = vocab_path
        self.embedding_path = embedding_path
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.token_step = token_step
        self.pointer = 0
        self.vocab_to_int = {}
        self.int_to_vocab = {}
        self.num_samples = None
        self.generator = None
        self.embedding_dim = None
        self.embedding_matrix = None

        if self.vocab_path:
            self.load_vocab()
        else:
            self.load_character_vocab()
        int_seq = self.text_preprocessing()
        self.load_embedding_matrix()
        self.generator = self.build_generator(int_seq)

    def load_character_vocab(self):
        with open(self.text_path, 'r', encoding='utf-8') as f:
            lines = sorted(list(set(f.read())))
        self.vocab_to_int = {token: num for num, token in enumerate(lines)}
        self.int_to_vocab = {num: token for num, token in enumerate(lines)}
        print('No vocab path. Loaded {} of character vocabs:'.format(len(self.vocab_to_int)))

    def load_vocab(self):
        with open(self.vocab_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        self.vocab_to_int = {token.strip().lower(): num for num, token in enumerate(lines)}
        self.int_to_vocab = {num: token.strip().lower() for num, token in enumerate(lines)}
        print('Loaded {} of vocabs:'.format(len(self.vocab_to_int)))

    def text_preprocessing(self, delimiter=None):
        with open(self.text_path, 'r', encoding='utf-8') as f:
            text_sequences = f.read().split(delimiter) if delimiter else f.read()
            int_sequences = [self.vocab_to_int[token] for token in text_sequences]
            self.num_samples = (len(int_sequences) - self.sequence_length) // self.token_step
        print('Number of samples:', self.num_samples)
        return int_sequences

    def load_embedding_matrix(self):
        embeddings_index = dict()
        if self.embedding_path:
            with open(self.embedding_path, 'r', encoding='UTF-8') as f:
                lines = f.readlines()
            for line in lines:
                values = line.split()
                token = values[0]
                vector = np.array(values[1:], dtype='float32')
                embeddings_index[token] = vector
            self.embedding_dim = len(values) - 1
            print('Loaded {} word vectors. Dimension is {}'.format(len(embeddings_index), self.embedding_dim))
            self.embedding_matrix = np.zeros((len(self.vocab_to_int), self.embedding_dim))
            for token, num in self.vocab_to_int.items():
                embedding_vector = embeddings_index.get(token)
                if embedding_vector is not None:
                    self.embedding_matrix[num] = embedding_vector
            print('embedding_matrix shape: %s' % str(self.embedding_matrix.shape))
        else:
            self.embedding_dim = len(self.vocab_to_int)
            print('No embedding path. Use one-hot encoding. Dimension is {}'.format(self.embedding_dim))
            self.embedding_matrix = np.zeros((len(self.vocab_to_int), self.embedding_dim))
            for _, num in self.vocab_to_int.items():
                self.embedding_matrix[num, num] = 1
        return self.embedding_matrix

    def build_generator(self, int_sequences):
        while True:
            input_data = []  # input shape:(batch_size, sequence_length, embedding_dim)
            target_data = []  # target use one-hot,shape:(batch_size, vocab_size)
            for sample in range(self.batch_size):
                try:
                    target = int_sequences[self.pointer + self.sequence_length]
                    inputs = int_sequences[self.pointer:self.pointer + self.sequence_length]
                    target_data.append([0] * len(self.vocab_to_int))
                    target_data[-1][target] = 1
                    input_data.append([[]] * self.sequence_length)
                    for time_step, num in enumerate(inputs):
                        input_data[-1][time_step] = self.embedding_matrix[num]
                except IndexError:
                    self.pointer = 0
                    break
                else:
                    self.pointer += self.token_step
            yield (np.array(input_data), np.array(target_data))


class SimpleRNNModel(object):
    def __init__(self, model_path, batch_size=64, epochs=100,
                 dropout=0.1, data_loader: SimpleRNNDataLoader = None):
        self.batch_size = batch_size
        self.epochs = epochs
        self.output_dim = None
        self.dropout = dropout
        self.data_loader = data_loader
        self.model = None
        self.model_path = model_path
        self.embedding_path = None
        self.embedding_matrix = None

    def build_model(self):
        vocab_length = len(self.data_loader.vocab_to_int)
        self.output_dim = self.data_loader.embedding_dim
        self.model = Sequential()
        self.model.add(
            LSTM(units=self.output_dim, input_shape=(self.data_loader.sequence_length, self.data_loader.embedding_dim),
                 dropout=self.dropout, recurrent_dropout=self.dropout))
        self.model.add(Dense(units=vocab_length, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        print(self.model.summary())

    def restore_model(self, model_path):
        self.model_path = model_path
        self.model = load_model(self.model_path)
        print('Restore model in {}'.format(model_path))

    def train_model(self):
        checkpointer = ModelCheckpoint(filepath=self.model_path, verbose=1, monitor='loss', save_best_only=True)
        earlystopping = EarlyStopping(monitor='loss', patience=3, verbose=1)
        self.model.fit_generator(generator=self.data_loader.generator,
                                 steps_per_epoch=self.data_loader.num_samples // self.batch_size,
                                 epochs=self.epochs, callbacks=[checkpointer, earlystopping])

    def generate_text(self, pre_text, text_length=1000):
        token_num_seq = []
        # tokenize
        input_token_seq = list(pre_text)
        input_data = np.zeros((1, self.data_loader.sequence_length, self.data_loader.embedding_dim), dtype='float32')
        for time_step, token in enumerate(input_token_seq[-self.data_loader.sequence_length:]):
            num = self.data_loader.vocab_to_int[token]
            token_num_seq.append(num)
            input_data[0, time_step] = self.data_loader.embedding_matrix[num]
        for _ in range(text_length):
            predict_label = self.model.predict(input_data)
            # topN_y = predict_label.argsort()[0, -top_n:][::-1]
            # word_choice=random.choice(topN_y)
            token_choice = np.random.choice(range(len(predict_label[0])), p=predict_label[0])
            new_data = np.array([[self.data_loader.embedding_matrix[token_choice]]])
            input_data = np.concatenate((input_data[:, 1:], new_data), axis=1)
            token_num_seq.append(token_choice)
        result_text = ''.join([self.data_loader.int_to_vocab[num] for num in token_num_seq])[-text_length:]
        print(result_text)


if __name__ == "__main__":
    # dl = SimpleRNNDataLoader(text_path='D:\deep_learning\datasets/1984.txt',
    #                          embedding_path='D:\deep_learning\glove.6B\glove.6B.50d.txt')
    dl = SimpleRNNDataLoader(text_path='D:\deep_learning\datasets/1984.txt')
    # dl = SimpleRNNDataLoader(text_path='D:\deep_learning\datasets\cmn-eng\source.txt', batch_size=6000)
    # for x, y in dl.generator:
    #     print(dl.pointer)
    #     print(y.shape)
    #     break

    #     if dl.pointer == 0:
    #         print(x[-1])
    #         break
    mp = 'D:\deep_learning\models/1984.hdf5'
    rm = SimpleRNNModel(model_path=mp, data_loader=dl)
    # rm.build_model()
    # if os.path.exists(mp):
    #     rm.restore_model(mp)
    # rm.train_model()
    rm.restore_model(mp)
    rm.generate_text(
        pre_text='It was chiefly in order to allow time for the preliminary work of translation that the final adoption of Newspeak had been fixed for so late a date as 2050.')
