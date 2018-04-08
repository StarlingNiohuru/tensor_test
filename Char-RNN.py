import sys
import re
from numpy import asarray, zeros, random, append
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint, Callback
from keras.utils.np_utils import to_categorical

# TEXT_DATA_DIR = 'D://A Brief History of Time.txt'
TEXT_DATA_DIR = 'D:/人类简史.txt'
# WORD2VEC_DIR = 'D://word2vec/word2vec_c_test'
WORD2VEC_DIR = 'D://word2vec_human.txt'
MODEL_CHECKPOINT_FILE = 'D://human_history.hdf5'
MAX_SEQUENCE_LENGTH = 20
CHAR_STEP = 1
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 60
EMBEDDING_DIM = 100
DROPOUT = 0.1
OUTPUT_DIM = 100
EPOCHS = 30


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
    return vocab_to_int, sequences


def load_train_data_and_labels(vocab_to_int, sequences):
    # load data without generator
    data = []
    labels = []
    count = 0
    for i in range(0, len(sequences) - MAX_SEQUENCE_LENGTH, CHAR_STEP):
        x_seq = sequences[i:i + MAX_SEQUENCE_LENGTH]
        y_seq = sequences[i + MAX_SEQUENCE_LENGTH]
        x = [vocab_to_int[c] for c in x_seq]
        y = [vocab_to_int[y_seq]]
        data.append(x)
        labels.append(y)
    data = asarray(data)
    labels = to_categorical(asarray(labels), num_classes=len(vocab_to_int))  # labels use one-hot format
    print('data X shape: %s' % str(data.shape))
    print('labels y shape (one-hot): %s' % str(labels.shape))
    return data, labels


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
    embedding_matrix = zeros((len(vocab_to_int), EMBEDDING_DIM))
    for word, i in vocab_to_int.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    print('embedding_matrix shape: %s' % str(embedding_matrix.shape))
    return embedding_matrix


# build a model
def build_model(embedding_matrix, vocab_length):
    model = Sequential()
    model.add(Embedding(vocab_length, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH,
                        trainable=False))
    model.add(LSTM(OUTPUT_DIM, input_shape=(MAX_SEQUENCE_LENGTH, vocab_length), return_sequences=True, dropout=DROPOUT,
                   recurrent_dropout=DROPOUT))
    model.add(LSTM(OUTPUT_DIM, return_sequences=True, dropout=DROPOUT, recurrent_dropout=DROPOUT))
    model.add(LSTM(OUTPUT_DIM, dropout=DROPOUT, recurrent_dropout=DROPOUT))
    model.add(Dense(vocab_length, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
    print(model.summary())
    return model


def train_model(model, data, labels):
    checkpointer = ModelCheckpoint(filepath=MODEL_CHECKPOINT_FILE, verbose=1, monitor='acc', save_best_only=True)
    model.fit(data, labels, batch_size=BATCH_SIZE, epochs=EPOCHS,  # validation_split=VALIDATION_SPLIT,
              callbacks=[checkpointer])


class GenerateText(Callback):

    def __init__(self, vocab_to_int):
        super(GenerateText, self).__init__()
        self.vocab_to_int = vocab_to_int

    def on_batch_end(self, pre_text, top_n=10, text_length=500):
        generate_text(pre_text, top_n, text_length)


def generate_text(pre_text, vocab_to_int, model, text_length=500):
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
        predict_label = model.predict(current_list)
        # topN_y = predict_label.argsort()[0, -top_n:][::-1]
        # word_choice=random.choice(topN_y)
        word_choice = random.choice(range(len(predict_label[0])), p=predict_label[0])
        int_list = append(int_list, [[word_choice]], axis=1)
    int_to_vocab = {i: c for c, i in vocab_to_int.items()}
    result_text = ''.join([int_to_vocab[d] for d in int_list[0]])[-text_length:]
    print(result_text)


if __name__ == "__main__":
    if sys.argv[1] == 'build':
        vocab_to_int, sequences = load_text()
        data, labels = load_train_data_and_labels(vocab_to_int, sequences)
        embedding_matrix = load_embedding_matrix(vocab_to_int)
        model = build_model(embedding_matrix, len(vocab_to_int))
        train_model(model, data, labels)
    elif sys.argv[1] == 'continue':
        vocab_to_int, sequences = load_text()
        data, labels = load_train_data_and_labels(vocab_to_int, sequences)
        model = load_model(MODEL_CHECKPOINT_FILE)
        train_model(model, data, labels)
    elif sys.argv[1] == 'test':
        vocab_to_int, sequences = load_text()
        model = load_model(MODEL_CHECKPOINT_FILE)
        # gt = GenerateText(vocab_to_int, model=model)
        pre_text = u'拥有神的能力，但是不负责任、贪得无厌，而且连想要什么都不知道。天下危险，恐怕莫此为甚。'
        # gt.on_batch_end(pre_text=pre_text)
        generate_text(pre_text, vocab_to_int, model)
