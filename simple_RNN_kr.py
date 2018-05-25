import os
import sys

import jieba
import nltk
from keras import Input, Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import LSTM, Dense
import numpy as np
from keras.models import load_model
from keras.utils import Sequence, multi_gpu_model
from tensorflow.python.client import device_lib


class SimpleRNNDataLoader(Sequence):
    def __init__(self, text_path, vocab_path=None, embedding_path=None, int_seq_path=None, batch_size=64,
                 sequence_length=20, token_step=3, language='English'):
        self.text_path = text_path
        self.vocab_path = vocab_path
        self.embedding_path = embedding_path
        self.int_seq_path = int_seq_path
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
        self.language = language

        if self.vocab_path and self.embedding_path:
            try:
                self.load_vocab()
            except FileNotFoundError:
                self.create_vocab_file()
            self.load_embedding_matrix()
        else:
            self.load_character_vocab()
            self.load_one_hot_matrix()

    def load_character_vocab(self):
        with open(self.text_path, 'r', encoding='utf-8') as f:
            lines = sorted(list(set(f.read())))
        self.vocab_to_int = {token: num for num, token in enumerate(lines)}
        self.int_to_vocab = {num: token for num, token in enumerate(lines)}
        print('No vocab path. Loaded {} of character vocabs:'.format(len(self.int_to_vocab)))

    def load_vocab(self):
        with open(self.vocab_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        self.vocab_to_int = {token.strip().lower(): num for num, token in enumerate(lines)}
        self.int_to_vocab = {num: token.strip().lower() for num, token in enumerate(lines)}
        print('Loaded {} of vocabs:'.format(len(self.int_to_vocab)))

    def create_vocab_file(self):
        with open(self.text_path, 'r', encoding='utf-8') as f:
            text = f.read()
        if self.language == 'English':
            token_seq = nltk.word_tokenize(text.lower())
        elif self.language == 'Chinese':
            text = text.lower().replace('\t', '').replace('\n', '').replace(' ', '').replace('\u3000', '')
            token_seq = jieba.cut(text)
        vocab_list = sorted(list(set(token_seq)))
        with open(self.vocab_path, 'w', encoding='utf-8') as f:
            f.writelines([word + '\n' for word in vocab_list])
        print('Created the vocab file in {}.'.format(self.vocab_path))
        self.load_vocab()

    def text_preprocessing(self):
        with open(self.text_path, 'r', encoding='utf-8') as f:
            text = f.read()
        if self.vocab_path and self.embedding_path:
            if self.language == 'English':
                text_sequences = nltk.word_tokenize(text.lower())
            elif self.language == 'Chinese':
                text = text.lower().replace('\t', '').replace('\n', '').replace(' ', '').replace('\u3000', '')
                text_sequences = jieba.cut(text)
        else:
            text_sequences = text
        # int_sequences = [self.vocab_to_int[token] for token in text_sequences]
        int_sequences = []
        for token in text_sequences:
            num = self.vocab_to_int.get(token)
            if num:
                int_sequences.append(num)
        self.num_samples = (len(int_sequences) - self.sequence_length) // self.token_step
        print('Number of samples:', self.num_samples)
        return int_sequences

    def save_int_seq_file(self, int_sequences):
        with open(self.int_seq_path, 'w', encoding='utf-8') as f:
            f.write(','.join(list(map(lambda x: str(x), int_sequences))))
        print('Created the int seq file in {}.'.format(self.int_seq_path))

    def load_int_seq_file(self):
        with open(self.int_seq_path, 'r', encoding='utf-8') as f:
            raw = f.read()
        int_sequences = list(map(lambda x: int(x), raw.split(',')))
        self.num_samples = (len(int_sequences) - self.sequence_length) // self.token_step
        print('Load int seq data from {}.Number of samples {}:'.format(self.int_seq_path, self.num_samples))
        return int_sequences

    def load_embedding_matrix(self):
        embeddings_index = dict()
        with open(self.embedding_path, 'r', encoding='UTF-8') as f:
            lines = f.readlines()
        for line in lines:
            values = line.split()
            if len(values) < 3:
                continue
            token = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings_index[token] = vector
        self.embedding_dim = len(values) - 1
        print('Loaded {} word vectors. Dimension is {}'.format(len(embeddings_index), self.embedding_dim))
        self.embedding_matrix = np.zeros((len(self.int_to_vocab), self.embedding_dim))
        for num, token in self.int_to_vocab.items():
            embedding_vector = embeddings_index.get(token)
            if embedding_vector is not None:
                self.embedding_matrix[num] = embedding_vector
        print('embedding_matrix shape: %s' % str(self.embedding_matrix.shape))
        return self.embedding_matrix

    def load_one_hot_matrix(self):
        self.embedding_dim = len(self.int_to_vocab)
        print('No embedding path. Use one-hot encoding. Dimension is {}'.format(self.embedding_dim))
        self.embedding_matrix = np.zeros((len(self.int_to_vocab), self.embedding_dim))
        for num, _ in self.int_to_vocab.items():
            self.embedding_matrix[num, num] = 1
        return self.embedding_matrix

    def build_generator(self, int_sequences):
        while True:
            input_data = []  # input shape:(batch_size, sequence_length, embedding_dim)
            target_data = []  # target shape:(batch_size, 1)
            for sample in range(self.batch_size):
                try:
                    target = int_sequences[self.pointer + self.sequence_length]
                    inputs = int_sequences[self.pointer:self.pointer + self.sequence_length]
                    # target_data.append([0] * len(self.vocab_to_int))
                    # target_data[-1][target] = 1
                    target_data.append([target])
                    input_data.append([[]] * self.sequence_length)
                    for time_step, num in enumerate(inputs):
                        input_data[-1][time_step] = self.embedding_matrix[num]
                except IndexError:
                    self.pointer = 0
                    break
                else:
                    self.pointer += self.token_step
            yield (np.array(input_data), np.array(target_data))

    def __len__(self):
        return self.batch_size

    def __getitem__(self, int_sequences):
        self.build_generator(int_sequences)

    def load_data_and_build_generator(self):
        if self.int_seq_path and os.path.exists(self.int_seq_path):
            int_seq = self.load_int_seq_file()
        else:
            int_seq = self.text_preprocessing()
            if self.int_seq_path:
                self.save_int_seq_file(int_seq)
        self.generator = self.build_generator(int_seq)


class SimpleRNNModel(object):
    def __init__(self, model_path, batch_size=64, epochs=100, output_dim=256,
                 dropout=0.1, data_loader: SimpleRNNDataLoader = None):
        self.batch_size = batch_size
        self.epochs = epochs
        self.output_dim = output_dim
        self.dropout = dropout
        self.data_loader = data_loader
        self.model = None
        self.model_path = model_path
        self.embedding_path = None
        self.embedding_matrix = None
        self.gpu_num = None

        self.get_gpu_num()

    def get_gpu_num(self):
        self.gpu_num = len([info for info in device_lib.list_local_devices() if info.device_type == 'GPU'])
        return self.gpu_num

    def build_model(self):
        vocab_length = len(self.data_loader.vocab_to_int)
        input_layer = Input(shape=(self.data_loader.sequence_length, self.data_loader.embedding_dim))
        lstm = LSTM(units=self.output_dim,
                    input_shape=(self.data_loader.sequence_length, self.data_loader.embedding_dim),
                    dropout=self.dropout, recurrent_dropout=self.dropout)(input_layer)
        dense = Dense(units=vocab_length, activation='softmax')(lstm)
        if self.gpu_num == 0:
            self.model = Model(inputs=input_layer, outputs=dense)
        else:
            self.model = multi_gpu_model(model=Model(inputs=input_layer, outputs=dense), gpus=self.gpu_num)
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop')
        print(self.model.summary())

    def restore_model(self, model_path):
        self.model_path = model_path
        self.model = load_model(self.model_path)
        print('Restore model in {}'.format(model_path))

    def train_model(self):
        checkpointer = ModelCheckpoint(filepath=self.model_path, verbose=1, monitor='loss', save_best_only=True)
        earlystopping = EarlyStopping(monitor='loss', patience=5, verbose=1)
        self.model.fit_generator(generator=self.data_loader.generator, shuffle=True,
                                 steps_per_epoch=self.data_loader.num_samples // self.batch_size,
                                 epochs=self.epochs, callbacks=[checkpointer, earlystopping])

    def generate_text(self, pre_text, text_length=1000, sample_type='random_choice'):
        token_num_seq = []
        # tokenize
        if self.data_loader.language == 'English':
            input_token_seq = nltk.word_tokenize(pre_text.lower())
            delimiter = ' '
        elif self.data_loader.language == 'Chinese':
            pre_text = pre_text.lower().replace('\t', '').replace('\n', '').replace(' ', '').replace('\u3000', '')
            input_token_seq = [word for word in jieba.cut(pre_text)]

            delimiter = ''
        else:
            input_token_seq = list(pre_text)
            delimiter = ''
        input_data = np.zeros((1, self.data_loader.sequence_length, self.data_loader.embedding_dim), dtype='float32')
        for time_step, token in enumerate(input_token_seq[-self.data_loader.sequence_length:]):
            num = self.data_loader.vocab_to_int[token]
            token_num_seq.append(num)
            input_data[0, time_step] = self.data_loader.embedding_matrix[num]
        for _ in range(text_length):
            predict_label = self.model.predict(input_data)
            # Sample a token
            if sample_type == 'max':
                token_choice = predict_label.argmax()
            elif sample_type == 'top3':
                token_choice = np.random.choice(predict_label.argsort()[0, -3:])
            elif sample_type == 'random_choice':
                token_choice = np.random.choice(range(len(predict_label[0])), p=predict_label[0])
            elif sample_type == 'threshold':
                idx = np.argwhere(predict_label >= 0.3)
                if idx.shape[0] > 0:
                    token_choice = np.random.choice(idx[-1])
                else:
                    token_choice = np.random.choice(range(len(predict_label[0])), p=predict_label[0])
            new_data = np.array([[self.data_loader.embedding_matrix[token_choice]]])
            input_data = np.concatenate((input_data[:, 1:], new_data), axis=1)
            token_num_seq.append(token_choice)
        result_text = delimiter.join([self.data_loader.int_to_vocab[num] for num in token_num_seq])[-text_length:]
        print(result_text)


if __name__ == "__main__":
    dl = SimpleRNNDataLoader(text_path='D:\deep_learning\datasets/骆驼祥子.txt',
                             embedding_path='D:\deep_learning\word2vec\word2vec_c',
                             vocab_path='D:\deep_learning\datasets/骆驼祥子_vocab.txt',
                             int_seq_path='D:\deep_learning\datasets/骆驼祥子_intseq.txt',
                             language='Chinese')
    mp = 'D:\deep_learning\models/camel.hdf5'
    rm = SimpleRNNModel(model_path=mp, data_loader=dl)
    if sys.argv[1] == 'train':
        dl.load_data_and_build_generator()
        if os.path.exists(mp):
            rm.restore_model(mp)
        rm.build_model()
        rm.train_model()
    elif sys.argv[1] == 'test':
        rm.restore_model(mp)
        rm.generate_text(
            pre_text='外面的黑暗渐渐习惯了，心中似乎停止了活动，他的眼不由的闭上了。', sample_type='random_choice')
    else:
        pass
