import math
import os
import sys

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.framework.errors_impl import NotFoundError
from tensorflow.examples.tutorials.mnist import input_data


class GANDataSet(object):
    def __init__(self, data_path=None, height=None, width=None, channels=3, batch_size=128):
        self.data_path = data_path
        self.height = height
        self.width = width
        self.channels = channels
        self.image_shape = (self.height, self.width, self.channels)
        self.batch_size = batch_size
        self.num_samples = None
        self.dataset = None
        self.iterator = None

        if self.data_path:
            self.build_dataset()
        else:
            self.data_path = 'D://deep_learning/datasets/mnist_data'
            self.build_mnist_dataset()
            self.height = 32
            self.width = 32
            self.channels = 1

    @staticmethod
    def read_function(filename):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_image(image_string)
        return image_decoded

    def parse_function(self, filename):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=self.channels)
        image_resized = tf.image.resize_images(image_decoded, [self.height, self.width])
        image_noisy = image_resized + np.random.normal(0, 1.0, image_resized.shape)
        image_noisy = tf.clip_by_value(image_noisy, 0, 255)
        image_normal = (tf.cast(image_noisy, tf.float32) - 127.5) / 127.5
        return image_normal

    def build_dataset(self):
        fname_list = [self.data_path + '/' + fname for fname in os.listdir(self.data_path)]
        self.num_samples = len(fname_list)
        print('Total number of samples: %s' % self.num_samples)
        filenames = tf.constant(fname_list)
        dataset = tf.data.Dataset.from_tensor_slices(filenames)
        dataset = dataset.map(self.parse_function)
        self.dataset = dataset.shuffle(buffer_size=self.num_samples).batch(self.batch_size).repeat()
        self.iterator = self.dataset.make_one_shot_iterator()

    def build_mnist_dataset(self):
        mnist = input_data.read_data_sets(self.data_path, one_hot=False)
        self.num_samples = mnist.train.num_examples
        print('Total number of samples: %s' % self.num_samples)
        x, _ = mnist.train.next_batch(self.num_samples)
        x = x.reshape([-1, 28, 28, 1])
        x = np.pad(x, [(0, 0), (2, 2), (2, 2), (0, 0)], 'constant', constant_values=(0, 0))
        mnist_ds = tf.data.Dataset.from_tensor_slices(x)
        self.dataset = mnist_ds.shuffle(buffer_size=self.num_samples).batch(self.batch_size).repeat()
        self.iterator = self.dataset.make_one_shot_iterator()


class DCGANModel(object):
    def __init__(self, session=tf.Session(), noise_dim=100, disc_latent_dim=128, gen_latent_dim=128, kernel_size=5,
                 beta1=0.5, gen_learning_rate=0.0002, disc_learning_rate=0.0002, momentum=0.8, dropout=0,
                 model_path=None, sample_images_path=None, dataset: GANDataSet = None):
        self.session = session
        self.noise_dim = noise_dim
        self.disc_latent_dim = disc_latent_dim
        self.gen_latent_dim = gen_latent_dim
        self.kernel_size = kernel_size
        self.beta1 = beta1
        self.gen_learning_rate = gen_learning_rate
        self.disc_learning_rate = disc_learning_rate
        self.momentum = momentum
        self.dropout = dropout
        self.model_path = model_path
        self.sample_images_path = sample_images_path
        self.dataset = dataset

        self.saver = None
        self.noise_input = None
        self.real_image_input = None
        self.gen_samples = None
        self.disc_target = None
        self.gen_target = None
        self.disc_loss = None
        self.gen_loss = None
        self.train_gen_op = None
        self.train_disc_op = None

    def generator(self, x, reuse=False):
        with tf.variable_scope('Generator', reuse=reuse):
            strides_num = 2
            padding_type = 'same'
            num_layers = math.ceil(math.log2(int(min(self.dataset.height, self.dataset.width)) / self.kernel_size))
            num_layers = min(int(num_layers), 4)
            filters_num = 2 ** (num_layers - 1) * self.gen_latent_dim

            print("Gen dense layer input shape: {}".format(x.shape))
            x = tf.layers.dense(x,
                                self.dataset.height * self.dataset.width * filters_num // (2 ** (2 * num_layers)),
                                kernel_initializer=tf.random_normal_initializer(stddev=0.02))
            print("Gen reshape layer input shape: {}".format(x.shape))
            x = tf.reshape(x, [-1, self.dataset.height // 2 ** num_layers, self.dataset.width // 2 ** num_layers,
                               filters_num])
            x = tf.layers.batch_normalization(x, epsilon=1e-5, momentum=self.momentum)
            x = tf.nn.relu(x)

            for i in range(1, num_layers + 1):
                print("Gen conv_trans layer-{} input shape: {}".format(i, x.shape))
                filters_num /= 2
                if i == num_layers:
                    filters_num = self.dataset.channels
                x = tf.layers.conv2d_transpose(x, filters=int(filters_num), kernel_size=self.kernel_size,
                                               strides=strides_num, padding=padding_type,
                                               kernel_initializer=tf.random_normal_initializer(stddev=0.02))
                x = tf.layers.batch_normalization(x, epsilon=1e-5, momentum=self.momentum)
                if i < num_layers:
                    x = tf.nn.relu(x)
                else:
                    x = tf.nn.tanh(x)
            print("Gen output shape: {}".format(x.shape))
            return x

    def discriminator(self, x, reuse=False):
        with tf.variable_scope('Discriminator', reuse=reuse):
            filters_num = self.disc_latent_dim
            strides_num = 2
            padding_type = 'same'
            num_layers = math.ceil(math.log2(int(min(x.shape[1], x.shape[2])) / self.kernel_size))
            num_layers = min(int(num_layers), 4)
            for i in range(1, num_layers + 1):
                # if i == num_layers:
                #     strides_num = 1
                #     padding_type = 'valid'
                print("Disc conv layer-{} input shape: {}".format(i, x.shape))
                x = tf.layers.conv2d(x, filters=filters_num, kernel_size=self.kernel_size, strides=strides_num,
                                     padding=padding_type, kernel_initializer=tf.random_normal_initializer(stddev=0.02))
                if i > 1:
                    x = tf.layers.batch_normalization(x, epsilon=1e-5, momentum=self.momentum)
                x = tf.nn.leaky_relu(x)
                x = tf.layers.dropout(x, rate=self.dropout)
                filters_num *= 2

            print("Disc reshape layer input shape: {}".format(x.shape))
            x = tf.reshape(x, [-1, x.shape[1] * x.shape[2] * x.shape[3]])
            print("Disc dense layer input shape: {}".format(x.shape))
            x = tf.layers.dense(x, 2, activation=tf.nn.softmax,
                                kernel_initializer=tf.random_normal_initializer(stddev=0.02))
            print("Disc output shape: {}".format(x.shape))
            return x

    def build_model(self):
        self.noise_input = tf.placeholder(tf.float32, shape=[None, self.noise_dim])
        self.real_image_input = tf.placeholder(tf.float32, shape=[None, self.dataset.height, self.dataset.width,
                                                                  self.dataset.channels])

        # Build Generator Network
        self.gen_samples = self.generator(self.noise_input)

        # Build 2 Discriminator Networks (one from noise input, one from generated samples)
        disc_real = self.discriminator(self.real_image_input)
        disc_fake = self.discriminator(self.gen_samples, reuse=True)
        disc_concat = tf.concat([disc_real, disc_fake], axis=0)

        # Build the stacked generator/discriminator
        stacked_gan = self.discriminator(self.gen_samples, reuse=True)

        # Build Targets (real or fake images)
        self.disc_target = tf.placeholder(tf.int32, shape=[None])
        self.gen_target = tf.placeholder(tf.int32, shape=[None])

        # Build Loss
        self.disc_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=disc_concat, labels=self.disc_target))
        self.gen_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=stacked_gan, labels=self.gen_target))

        # Build Optimizers
        optimizer_gen = tf.train.AdamOptimizer(learning_rate=self.gen_learning_rate, beta1=self.beta1)
        optimizer_disc = tf.train.AdamOptimizer(learning_rate=self.disc_learning_rate, beta1=self.beta1)

        # Training Variables for each optimizer
        gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
        disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')

        # Create training operations
        self.train_gen_op = optimizer_gen.minimize(self.gen_loss, var_list=gen_vars)
        self.train_disc_op = optimizer_disc.minimize(self.disc_loss, var_list=disc_vars)

        self.saver = tf.train.Saver()

    def restore_model(self):
        try:
            self.saver.restore(self.session, self.model_path)
            print("Model restored from path: %s" % self.model_path)
        except NotFoundError:
            print("Model path not found %s" % self.model_path)

    def train_model(self, num_epochs=1000, d_times=1, g_times=1):
        steps_one_epoch = int(math.ceil(self.dataset.num_samples / self.dataset.batch_size))
        num_steps = num_epochs * steps_one_epoch
        self.session.run(tf.global_variables_initializer())
        next_element = self.dataset.iterator.get_next()
        for step in range(1, num_steps + 1):
            batch_x = self.session.run(next_element)
            batch_size = batch_x.shape[0]
            # training discriminator
            for _ in range(d_times):
                z = np.random.uniform(-1, 1, [batch_size, self.noise_dim])
                batch_disc_y = np.concatenate([np.ones([batch_size]), np.zeros([batch_size])], axis=0)
                feed_dict = {self.real_image_input: batch_x, self.noise_input: z, self.disc_target: batch_disc_y}
                _, disc_loss = self.session.run([self.train_disc_op, self.disc_loss], feed_dict=feed_dict)
            # training generator
            for _ in range(g_times):
                z = np.random.uniform(-1, 1, [batch_size * 2, self.noise_dim])
                batch_gen_y = np.ones([batch_size * 2])
                feed_dict = {self.noise_input: z, self.gen_target: batch_gen_y}
                _, gen_loss = self.session.run([self.train_gen_op, self.gen_loss], feed_dict=feed_dict)

            epoch = int(math.ceil(step / steps_one_epoch))
            print(
                'Epoch: %i, Step: %i, Generator Loss: %f, Discriminator Loss: %f' % (epoch, step, gen_loss, disc_loss))
            if step % steps_one_epoch == 0:
                self.sample_images(epoch)
                save_path = self.saver.save(self.session, self.model_path)
                print('Model saved in path: %s' % save_path)

    def sample_images(self, epoch, images_path=None):
        rows = 5
        cols = 5
        fig, axes = plt.subplots(rows, cols, figsize=(8, 8), dpi=100)
        z = np.random.uniform(-1, 1, size=[rows * cols, self.noise_dim])
        samples = self.session.run(self.gen_samples, feed_dict={self.noise_input: z})
        count = 0
        for row in range(rows):
            for col in range(cols):
                # print(0.5*samples[count] + 0.5)
                axes[row, col].imshow(np.clip(0.5 * samples[count] + 0.5, 0, 1), interpolation='nearest')
                axes[row, col].axis('off')
                count += 1
        images_path = images_path if images_path else self.sample_images_path + "\sample_%d.png" % epoch
        fig.savefig(images_path)
        plt.close()

    def test(self):
        self.session.run(tf.global_variables_initializer())
        # self.sample_images(1)
        next_element = self.dataset.iterator.get_next()
        batch_x = self.session.run(next_element)
        # z = np.random.normal(0, 1., size=[25, self.noise_dim])
        # samples = self.session.run(self.gen_samples, feed_dict={self.noise_input: z})
        # print(samples.shape)
        # print(samples[0,30,:,0]*127.5+127.5)
        rows = 5
        cols = 5
        fig, axes = plt.subplots(rows, cols, figsize=(8, 8), dpi=100)
        count = 0
        for row in range(rows):
            for col in range(cols):
                ia = batch_x[count]
                print(ia)
                axes[row, col].imshow(0.5 * ia + 0.5, interpolation='nearest')
                axes[row, col].axis('off')
                count += 1
        images_path = self.sample_images_path + "\sample_test.png"
        fig.savefig(images_path)
        plt.close()

    def test2(self):
        self.session.run(tf.global_variables_initializer())
        next_element = self.dataset.iterator.get_next()
        batch_x = self.session.run(next_element)
        print(batch_x.shape)
        print(batch_x[0])


if __name__ == "__main__":
    # gds = GANDataSet(data_path='D://deep_learning/datasets/actress', height=64, width=64)
    # dgm = DCGANModel(model_path='D://deep_learning/models/actress/actress.ckpt',
    #                  sample_images_path='D://deep_learning/samples',
    #                  gen_learning_rate=0.0002,
    #                  disc_learning_rate=0.00005,
    #                  dataset=gds)
    if sys.argv[1] == 'train':
        gds = GANDataSet(data_path='D://deep_learning/datasets/sheephead', height=64, width=64)
        dgm = DCGANModel(model_path='D://deep_learning/models/sheephead/sheephead.ckpt',
                         sample_images_path='D://deep_learning/samples',
                         gen_learning_rate=0.0002,
                         disc_learning_rate=0.00005,
                         dataset=gds)
        dgm.build_model()
        # dgm.restore_model()
        # dgm.train_model(num_epochs=1000)
    elif sys.argv[1] == 'test':
        gds = GANDataSet()
        dgm = DCGANModel(model_path='D://deep_learning/models/mnist/mnist.ckpt',
                         sample_images_path='D://deep_learning/samples',
                         dataset=gds)
        dgm.build_model()
        dgm.restore_model()
        dgm.train_model(num_epochs=1000)
