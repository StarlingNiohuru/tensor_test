import re
import sys
from numpy import asarray, zeros, random, append
from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense
from keras.layers.embeddings import Embedding
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint, Callback

TEXT_DATA_DIR = 'D:/人类简史.txt'
WORD2VEC_DIR = 'D://word2vec_human.txt'
MODEL_CHECKPOINT_FILE = 'D://homo_history.hdf5'
MAX_SEQUENCE_LENGTH = 20
CHAR_STEP = 1
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 60
LATENT_DIM = 100
DROPOUT = 0.1
EPOCHS = 10


def load_text():
    # Chinese text load and preprocessing 中文用字，英文用词
    with open(TEXT_DATA_DIR, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    # text= ''.join([i if ord(i) < 128 else ' ' for i in raw_text])
    text = raw_text.replace('\r', '').replace('\n', '').replace('\t', '').lower()
    partern = '[\u4e00-\u9fa5]|[，、。！？；：“”《》——【】『』]|[-.%/]|[A-Za-z]+'
    sequences = re.findall(partern, text)
    vocab_to_int = {}
    num = 0
    for word in sequences:
        if word not in vocab_to_int:
            vocab_to_int[word] = num
            num += 1
    vocab_length = len(vocab_to_int)
    print('Found %s unique tokens.' % vocab_length)
    # int_to_vocab = {i: c for c, i in vocab_to_int.items()}
    print('len of total sequences: %s' % len(sequences))
    return vocab_to_int, sequences


def train_data_generator(vocab_to_int, sequences, batch_size=BATCH_SIZE):
    encoder_input = []
    decoder_input = []
    decoder_target = []
    count = 0
    for i in range(0, len(sequences) - 2 * MAX_SEQUENCE_LENGTH, CHAR_STEP):
        x_seq = sequences[i:i + MAX_SEQUENCE_LENGTH]
        y_seq = sequences[(i + MAX_SEQUENCE_LENGTH - 1):(i + 2 * MAX_SEQUENCE_LENGTH)]
        x = [vocab_to_int[c] for c in x_seq]
        y = [vocab_to_int[c] for c in y_seq]
        encoder_input.append(x)
        decoder_input.append(y[:-1])
        decoder_target.append(y[1:])
        count += 1
        if count % batch_size == 0:
            encoder_input_data = asarray(encoder_input[-batch_size:])
            decoder_input_data = asarray(decoder_input[-batch_size:])
            decoder_target_data = to_categorical(asarray(decoder_target[-batch_size:]),
                                                 num_classes=len(vocab_to_int))  # labels use one-hot format
            # decoder_target_data = asarray(decoder_target)
            # print('encoder_input_data shape: %s' % str(encoder_input_data.shape))
            # print('decoder_input_data shape: %s' % str(decoder_input_data.shape))
            # print('decoder_target_data shape (one-hot): %s' % str(decoder_target_data.shape))
            yield ([encoder_input_data, decoder_input_data], decoder_target_data)


def load_embedding_matrix(vocab_to_int):
    # load words_embedding_vector
    embeddings_index = dict()
    f = open(WORD2VEC_DIR, 'r', encoding='UTF-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))

    # get embedding matrix
    embedding_matrix = zeros((len(vocab_to_int), LATENT_DIM))
    for word, i in vocab_to_int.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    print('embedding_matrix shape: %s' % str(embedding_matrix.shape))
    return embedding_matrix


def build_model(embedding_matrix, vocab_length):
    num_encoder_tokens = vocab_length
    num_decoder_tokens = vocab_length
    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(None,))
    x = Embedding(num_encoder_tokens, LATENT_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH,
                  trainable=False)(encoder_inputs)
    x, state_h, state_c = LSTM(LATENT_DIM, return_state=True)(x)
    # encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None,))
    x = Embedding(num_decoder_tokens, LATENT_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH,
                  trainable=False)(decoder_inputs)
    x = LSTM(LATENT_DIM, return_sequences=True)(x, initial_state=encoder_states)
    decoder_outputs = Dense(num_decoder_tokens, activation='softmax')(x)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    print(model.summary())
    # Compile & run training
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
    return model


def train_model(model, data_generator, sequences):
    checkpointer = ModelCheckpoint(filepath=MODEL_CHECKPOINT_FILE, verbose=1, monitor='acc', save_best_only=True)
    model.fit_generator(data_generator,
                        steps_per_epoch=(len(sequences) - 2 * MAX_SEQUENCE_LENGTH) / CHAR_STEP / BATCH_SIZE,
                        epochs=EPOCHS, callbacks=[checkpointer])


def generate_text(pre_text, vocab_to_int, model, text_length=1):
    def char_seq_to_int(char_seq):
        char_num_list = []
        for char in char_seq:
            char_num = vocab_to_int[char]
            # print(char)
            char_num_list.append(char_num)
        # print(char_num_list)
        return asarray([char_num_list])[:, -MAX_SEQUENCE_LENGTH:]

    input_text = pre_text
    int_list = char_seq_to_int(input_text)

    for i in range(text_length):
        current_list = int_list[:, -MAX_SEQUENCE_LENGTH:]
        zero_list = asarray([[0 for i in range(MAX_SEQUENCE_LENGTH)]])
        predict_label = model.predict([current_list, zero_list])
        # topN_y = predict_label.argsort()[0, -top_n:][::-1]
        # word_choice=random.choice(topN_y)
        for k in range(MAX_SEQUENCE_LENGTH):
            word_choice = random.choice(range(len(predict_label[0][k])), p=predict_label[0][k])
            int_list = append(int_list, [[word_choice]], axis=1)
        int_to_vocab = {i: c for c, i in vocab_to_int.items()}
        result_text = ''.join([int_to_vocab[d] for d in int_list[0]])
        print(result_text)


if __name__ == "__main__":
    if sys.argv[1] == 'build':
        vocab_to_int, sequences = load_text()
        data_generator = train_data_generator(vocab_to_int, sequences)
        embedding_matrix = load_embedding_matrix(vocab_to_int)
        model = build_model(embedding_matrix, len(vocab_to_int))
        train_model(model, data_generator, sequences)
    elif sys.argv[1] == 'continue':
        vocab_to_int, sequences = load_text()
        data_generator = train_data_generator(vocab_to_int, sequences)
        model = load_model(MODEL_CHECKPOINT_FILE)
        train_model(model, data_generator, sequences)
    elif sys.argv[1] == 'test':
        vocab_to_int, sequences = load_text()
        model = load_model(MODEL_CHECKPOINT_FILE)
        pre_text = u'拥有神的能力，但是不负责任、贪得无厌，而且连想要什么都不知道。天下危险，恐怕莫此为甚。'
        generate_text(pre_text, vocab_to_int, model)
