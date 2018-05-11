import jieba
import nltk
import tensorflow as tf
import numpy as np
from tensorflow.python.framework.errors_impl import NotFoundError


class Seq2SeqDataLoader(object):
    def __init__(self, file_path_list, num_samples=1000000, batch_size=64):
        self.file_path_list = file_path_list
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.input_seqs = []
        self.target_seqs = []
        self.input_vocabs = None
        self.target_vocabs = None
        self.num_encoder_tokens = None
        self.num_decoder_tokens = None
        self.max_encoder_seq_length = None
        self.max_decoder_seq_length = None
        self.input_token_index = None
        self.target_token_index = None
        self.pointer = 0

    def load_total_text_data(self):
        input_vocabs = set()
        target_vocabs = set()
        for file_path in self.file_path_list:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.read().split('\n')
            for line in lines[: min(self.num_samples, len(lines) - 1)]:
                input_text, target_text = line.split('\t')
                # We use "tab" as the "start sequence" character
                # for the targets, and "\n" as "end sequence" character.
                target_text = '\t' + target_text + '\n'
                # tokenize
                input_seq = nltk.word_tokenize(input_text.lower())
                target_seq = [word for word in jieba.cut(target_text)]
                self.input_seqs.append(input_seq)
                self.target_seqs.append(target_seq)
                for token in input_seq:
                    if token not in input_vocabs:
                        input_vocabs.add(token)
                for token in target_seq:
                    if token not in target_vocabs:
                        target_vocabs.add(token)

        self.num_samples = len(self.input_seqs)
        self.input_vocabs = sorted(list(input_vocabs))
        self.target_vocabs = sorted(list(target_vocabs))
        self.num_encoder_tokens = len(input_vocabs)
        self.num_decoder_tokens = len(target_vocabs)
        self.max_encoder_seq_length = max([len(txt) for txt in self.input_seqs])
        self.max_decoder_seq_length = max([len(txt) for txt in self.target_seqs])

        print('Number of samples:', self.num_samples)
        print('Number of unique input tokens:', self.num_encoder_tokens)
        print('Number of unique output tokens:', self.num_decoder_tokens)
        print('Max sequence length for inputs:', self.max_encoder_seq_length)
        print('Max sequence length for outputs:', self.max_decoder_seq_length)

        self.input_token_index = dict(
            [(token, i) for i, token in enumerate(self.input_vocabs, 1)])
        self.target_token_index = dict(
            [(token, i) for i, token in enumerate(self.target_vocabs, 1)])

    def next_batch(self):
        x = self.input_seqs[self.pointer:min(self.pointer + self.batch_size, self.num_samples - 1)]
        y = self.target_seqs[self.pointer:min(self.pointer + self.batch_size, self.num_samples - 1)]
        self.pointer += self.batch_size
        seq_len_x = list(map(lambda k: len(k), x))
        seq_len_y = list(map(lambda k: len(k), y))

        encoder_input_data = np.zeros((self.batch_size, max(seq_len_x)), dtype='int32')
        decoder_input_data = np.zeros((self.batch_size, max(seq_len_y)), dtype='int32')
        decoder_target_data = np.zeros((self.batch_size, max(seq_len_y)), dtype='int32')

        for i, (input_text, target_text) in enumerate(zip(x, y)):
            for t, token in enumerate(input_text):
                encoder_input_data[i, t] = self.input_token_index[token]
            for t, token in enumerate(target_text):
                # decoder_target_data is ahead of decoder_input_data by one timestep
                decoder_input_data[i, t] = self.target_token_index[token]
                if t > 0:
                    # decoder_target_data will be ahead by one timestep
                    # and will not include the start character.
                    decoder_target_data[i, t - 1] = self.target_token_index[token]

        return encoder_input_data, decoder_input_data, decoder_target_data, np.array(seq_len_x), np.array(seq_len_y)


class Seq2SeqAttention(object):
    def __init__(self, batch_size=64, epochs=100, num_samples=10000, embedding_dim=256, num_units=256,
                 max_gradient_norm=5.0, learning_rate=1.0, input_max_length=20, target_max_length=20,
                 model_path="D:/deep_learning/test.ckpt"):
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_samples = num_samples
        self.num_units = num_units
        self.model_path = model_path

        self.embedding_dim = embedding_dim
        self.input_max_length = input_max_length
        self.input_vocab_size = 30000
        self.target_max_length = target_max_length
        self.target_vocab_size = 30000
        self.max_gradient_norm = max_gradient_norm
        self.learning_rate = learning_rate

    def build_model(self):
        self.encoder_inputs = tf.placeholder(dtype=tf.int32,
                                             shape=[self.batch_size, None],
                                             name='encoder_inputs')
        self.encoder_seq_len = tf.placeholder(dtype=tf.int32, shape=[self.batch_size], name='encoder_seq_len')
        self.decoder_inputs = tf.placeholder(dtype=tf.int32,
                                             shape=[self.batch_size, None],
                                             name='decoder_inputs')
        self.decoder_seq_len = tf.placeholder(dtype=tf.int32, shape=[self.batch_size], name='decoder_seq_len')
        self.decoder_targets = tf.placeholder(dtype=tf.int32,
                                              shape=[self.batch_size, None],
                                              name='decoder_targets')
        # embedding
        embedding_encoder = tf.get_variable(name="embedding_encoder",
                                            shape=[self.input_vocab_size, self.embedding_dim])
        encoder_emb_inp = tf.nn.embedding_lookup(params=embedding_encoder, ids=self.encoder_inputs)
        embedding_decoder = tf.get_variable(name="embedding_decoder",
                                            shape=[self.target_vocab_size, self.embedding_dim])
        decoder_emb_inp = tf.nn.embedding_lookup(params=embedding_decoder, ids=self.decoder_inputs)
        decoder_emb_tar = tf.nn.embedding_lookup(params=embedding_decoder, ids=self.decoder_targets)
        # encoder
        encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.num_units)
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(cell=encoder_cell, inputs=encoder_emb_inp, dtype=tf.float32,
                                                           sequence_length=self.encoder_seq_len)
        # attention mechanism
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=self.num_units, memory=encoder_outputs,
                                                                memory_sequence_length=self.encoder_seq_len)
        # decoder
        decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.num_units)
        attention_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,
                                                             attention_layer_size=self.num_units)
        helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_emb_inp, sequence_length=self.decoder_seq_len)
        decoder = tf.contrib.seq2seq.BasicDecoder(cell=attention_cell, helper=helper,
                                                  initial_state=attention_cell.zero_state(self.batch_size,
                                                                                          tf.float32).clone(
                                                      cell_state=encoder_state))
        decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder)
        logits = decoder_outputs.rnn_output
        # loss
        crossent = tf.nn.softmax_cross_entropy_with_logits_v2(labels=decoder_emb_tar, logits=logits)
        target_weights = tf.sequence_mask(lengths=self.decoder_seq_len, maxlen=tf.reduce_max(self.decoder_seq_len),
                                          dtype=logits.dtype)
        self.train_loss = (tf.reduce_sum(input_tensor=crossent * target_weights) / self.batch_size)
        params = tf.trainable_variables()
        gradients = tf.gradients(ys=self.train_loss, xs=params)
        clipped_gradients, _ = tf.clip_by_global_norm(t_list=gradients, clip_norm=self.max_gradient_norm)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        update_step = self.optimizer.apply_gradients(grads_and_vars=zip(clipped_gradients, params))

        with tf.Session() as sess:
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter('D:/deep_learning/graph/test', sess.graph)

    def train(self, train_steps=200, is_restore=1):
        sdl = Seq2SeqDataLoader(file_path_list=['D:\deep_learning\datasets\cmn-eng\cmn.txt'],
                                batch_size=self.batch_size)
        sdl.load_total_text_data()
        self.saver = tf.train.Saver()
        with tf.Session() as sess:
            if is_restore:
                try:
                    self.saver.restore(sess, self.model_path)
                    print("Model restored in path: %s" % self.model_path)
                except NotFoundError:
                    print("Model path not found %s" % self.model_path)
            sess.run(tf.global_variables_initializer())
            for step in range(1, train_steps + 1):
                x, y, yt, input_seq_len, target_seq_len = sdl.next_batch()
                loss = sess.run(self.train_loss,
                                feed_dict={self.encoder_inputs: x, self.decoder_inputs: y, self.decoder_targets: yt,
                                           self.encoder_seq_len: input_seq_len, self.decoder_seq_len: target_seq_len})
                print("Step {}: Loss:{}".format(step, loss))
                save_path = self.saver.save(sess, self.model_path)
                print("Model saved in path: %s" % save_path)

    def infer(self):
        # Helper
        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding_decoder, tf.fill([self.batch_size], '\t'), '\n')
        # Decoder
        decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, encoder_state)
        # Dynamic decoding
        maximum_iterations = tf.round(tf.reduce_max(source_sequence_length) * 2)
        outputs, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=maximum_iterations)
        translations = outputs.sample_id


if __name__ == "__main__":
    sa = Seq2SeqAttention()
    sa.build_model()
    sa.train()
    # sdl = Seq2SeqDataLoader(file_path_list=['D:\deep_learning\datasets\cmn-eng\cmn.txt'])
    # sdl.load_total_text_data()
