import sys
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense
import numpy as np


class Seq2SeqHandler(object):

    def __init__(self, batch_size=64, epochs=100, latent_dim=256, num_samples=10000, validation_split=0.2,
                 data_path=None, model_path=None):
        self.batch_size = batch_size  # Batch size for training.
        self.epochs = epochs  # Number of epochs to train for.
        self.latent_dim = latent_dim  # Latent dimensionality of the encoding space.
        self.num_samples = num_samples  # Number of samples to train on.
        self.validation_split = validation_split
        # Path to the data txt file on disk.
        self.data_path = data_path
        self.model_path = model_path
        self.input_characters = None
        self.target_characters = None
        self.num_encoder_tokens = None
        self.num_decoder_tokens = None
        self.max_encoder_seq_length = None
        self.max_decoder_seq_length = None
        self.input_token_index = None
        self.target_token_index = None
        self.reverse_input_char_index = None
        self.reverse_target_char_index = None
        self.model = None
        self.encoder_model = None
        self.decoder_model = None

    def load_data(self):
        # Vectorize the data.
        input_texts = []
        target_texts = []
        input_characters = set()
        target_characters = set()
        with open(self.data_path, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')
        for line in lines[: min(self.num_samples, len(lines) - 1)]:
            input_text, target_text = line.split('\t')
            # We use "tab" as the "start sequence" character
            # for the targets, and "\n" as "end sequence" character.
            target_text = '\t' + target_text + '\n'
            input_texts.append(input_text)
            target_texts.append(target_text)
            for char in input_text:
                if char not in input_characters:
                    input_characters.add(char)
            for char in target_text:
                if char not in target_characters:
                    target_characters.add(char)

        self.num_samples = len(input_texts)
        self.input_characters = sorted(list(input_characters))
        self.target_characters = sorted(list(target_characters))
        self.num_encoder_tokens = len(input_characters)
        self.num_decoder_tokens = len(target_characters)
        self.max_encoder_seq_length = max([len(txt) for txt in input_texts])
        self.max_decoder_seq_length = max([len(txt) for txt in target_texts])

        print('Number of samples:', self.num_samples)
        print('Number of unique input tokens:', self.num_encoder_tokens)
        print('Number of unique output tokens:', self.num_decoder_tokens)
        print('Max sequence length for inputs:', self.max_encoder_seq_length)
        print('Max sequence length for outputs:', self.max_decoder_seq_length)

        self.input_token_index = dict(
            [(char, i) for i, char in enumerate(self.input_characters)])
        self.target_token_index = dict(
            [(char, i) for i, char in enumerate(self.target_characters)])

        # Reverse-lookup token index to decode sequences back to
        # something readable.
        self.reverse_input_char_index = dict(
            (i, char) for char, i in self.input_token_index.items())
        self.reverse_target_char_index = dict(
            (i, char) for char, i in self.target_token_index.items())

        return input_texts, target_texts

    def vecotorize_data(self, input_texts, target_texts):
        encoder_input_data = np.zeros(
            (len(input_texts), self.max_encoder_seq_length, self.num_encoder_tokens),
            dtype='float32')
        decoder_input_data = np.zeros(
            (len(input_texts), self.max_decoder_seq_length, self.num_decoder_tokens),
            dtype='float32')
        decoder_target_data = np.zeros(
            (len(input_texts), self.max_decoder_seq_length, self.num_decoder_tokens),
            dtype='float32')

        for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
            for t, char in enumerate(input_text):
                encoder_input_data[i, t, self.input_token_index[char]] = 1.
            for t, char in enumerate(target_text):
                # decoder_target_data is ahead of decoder_input_data by one timestep
                decoder_input_data[i, t, self.target_token_index[char]] = 1.
                if t > 0:
                    # decoder_target_data will be ahead by one timestep
                    # and will not include the start character.
                    decoder_target_data[i, t - 1, self.target_token_index[char]] = 1.
        return encoder_input_data, decoder_input_data, decoder_target_data

    def vecotorize_data_generator(self, input_texts, target_texts):
        count = 0
        # while count < len(input_texts) - self.batch_size:
        while True:
            batch_input = input_texts[count:count + self.batch_size]
            batch_target = target_texts[count:count + self.batch_size]
            encoder_input_data, decoder_input_data, decoder_target_data = self.vecotorize_data(batch_input,
                                                                                               batch_target)
            yield ([encoder_input_data, decoder_input_data], decoder_target_data)
            count = count + self.batch_size if count < len(input_texts) + 1 - self.batch_size else 0

    def build_training_model(self):
        # Define an input sequence and process it.
        encoder_inputs = Input(shape=(None, self.num_encoder_tokens))
        encoder = LSTM(self.latent_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None, self.num_decoder_tokens))
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        decoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                             initial_state=encoder_states)
        decoder_dense = Dense(self.num_decoder_tokens, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        print(self.model.summary())
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    def train_model(self, encoder_input_data=None, decoder_input_data=None, decoder_target_data=None, generator=None):
        # Run training
        checkpointer = ModelCheckpoint(filepath=self.model_path, monitor='loss', verbose=1, save_best_only=True)
        earlystopping = EarlyStopping(monitor='loss', patience=3, verbose=1)
        if generator:
            self.model.fit_generator(generator=generator, steps_per_epoch=self.num_samples // self.batch_size,
                                     epochs=self.epochs,
                                     callbacks=[checkpointer, earlystopping])
        else:
            self.model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                           batch_size=self.batch_size,
                           epochs=self.epochs,
                           validation_split=self.validation_split,
                           callbacks=[checkpointer, earlystopping])

    def restore_model(self):
        self.model = load_model(self.model_path)
        encoder_inputs = self.model.input[0]  # input_1
        encoder_outputs, state_h_enc, state_c_enc = self.model.layers[2].output  # lstm_1
        encoder_states = [state_h_enc, state_c_enc]
        self.encoder_model = Model(encoder_inputs, encoder_states)

        decoder_inputs = self.model.input[1]  # input_2
        decoder_state_input_h = Input(shape=(self.latent_dim,), name='input_3')
        decoder_state_input_c = Input(shape=(self.latent_dim,), name='input_4')
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_lstm = self.model.layers[3]
        decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
            decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h_dec, state_c_dec]
        decoder_dense = self.model.layers[4]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)

    def decode_sequence(self, input_seq):
        # Encode the input as state vectors.
        states_value = self.encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, self.num_decoder_tokens))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, self.target_token_index['\t']] = 1.

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict(
                [target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            # sampled_token_index = np.random.choice(output_tokens[0, -1, :].argsort()[-3:][::-1])
            # sampled_token_index = np.random.choice(range(self.num_decoder_tokens), p=output_tokens[0, -1, :])
            sampled_char = self.reverse_target_char_index[sampled_token_index]
            decoded_sentence += sampled_char

            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_char == '\n' or
                    len(decoded_sentence) > self.max_decoder_seq_length):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, self.num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.

            # Update states
            states_value = [h, c]

        return decoded_sentence

    def generate_text(self, input_char, num_of_sentences=10):
        total_text = ''
        for _ in range(num_of_sentences):
            input_seq = np.zeros((1, self.max_encoder_seq_length, self.num_encoder_tokens), dtype='float32')
            for t, char in enumerate(input_char):
                input_seq[0, t, self.input_token_index[char]] = 1.
            input_char = self.decode_sequence(input_seq).replace('\t', '').replace('\n', '')
            total_text += input_char
        return total_text


if __name__ == "__main__":
    sh = Seq2SeqHandler(data_path='D:/datasets/Water Margin pretrain.txt', model_path='D:\models\water_margin.hdf5',
                        num_samples=10000000)
    if sys.argv[1] in ['build', 'continue']:
        input_texts, target_texts = sh.load_data()
        # encoder_input_data, decoder_input_data, decoder_target_data = sh.vecotorize_data(input_texts, target_texts)
        g = sh.vecotorize_data_generator(input_texts, target_texts)
        if sys.argv[1] == 'build':
            sh.build_training_model()
            # sh.train_model(encoder_input_data, decoder_input_data, decoder_target_data)
            sh.train_model(generator=g)
        elif sys.argv[1] == 'continue':
            sh.restore_model()
            # sh.train_model(encoder_input_data, decoder_input_data, decoder_target_data)
            sh.train_model(generator=g)
    elif sys.argv[1] == 'test':
        sh.load_data()
        sh.restore_model()
        # for seq_index in range(30):
        #     Take one sequence (part of the training set)
        #     for trying out decoding.
            # encoder_input_data, decoder_input_data, decoder_target_data = sh.vecotorize_data(input_texts, target_texts)
        #     input_seq = encoder_input_data[seq_index: seq_index + 1]
        #     decoded_sentence = sh.decode_sequence(input_seq)
        #     print('-')
        #     print('Input sentence:', input_texts[seq_index])
        #     print('Decoded sentence:', decoded_sentence)
        # sample_text = sh.generate_text(input_char='鲁提辖坐了主位，李忠对席，史进下首坐了。')
        sample_text = sh.generate_text(input_char='你好。')
        print(sample_text)
    elif sys.argv[1] == 'debug':
        input_texts, target_texts = sh.load_data()
        g = sh.vecotorize_data_generator(input_texts, target_texts)
        for i in g:
            print(i[0].shape)
            print(i[1].shape)
            break
