import math
import os
import sys

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework.errors_impl import NotFoundError


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


class WGANGPModel(object):
    def __init__(self, session=tf.Session(), noise_dim=100, latent_dim=64, kernel_size=3, disc_iters=5, gp_lambda=10,
                 model_path=None, sample_images_path=None, summaries_path=None, dataset: GANDataSet = None):
        self.session = session
        self.noise_dim = noise_dim
        self.latent_dim = latent_dim
        self.kernel_size = kernel_size
        self.disc_iters = disc_iters
        self.gp_lambda = gp_lambda
        self.model_path = model_path
        self.sample_images_path = sample_images_path
        self.summaries_path = summaries_path
        self.dataset = dataset
        self.train_writer = tf.summary.FileWriter(self.summaries_path + '/train', self.session.graph)

        self.saver = None
        self.is_restore = False
        self.noise_input = None
        self.real_image_input = None
        self.gen_samples = None
        self.disc_target = None
        self.gen_target = None
        self.disc_loss = None
        self.gen_loss = None
        self.train_gen_op = None
        self.train_disc_op = None
        self.disc_sum_merged = None
        self.gen_sum_merged = None

    def generator(self, x, reuse=False):
        with tf.variable_scope('Generator', reuse=reuse):
            strides_num = 1
            padding_type = 'valid'
            num_layers = math.ceil(math.log2(int(min(self.dataset.height, self.dataset.width)) / self.kernel_size))
            num_layers = min(int(num_layers), 4)
            filters_num = 2 ** (num_layers - 1) * self.gen_latent_dim

            print("Gen dense layer input shape: {}".format(x.shape))
            x = tf.layers.dense(x,
                                self.dataset.height * self.dataset.width * filters_num // (2 ** (2 * num_layers)),
                                kernel_initializer=tf.random_normal_initializer())
            print("Gen reshape layer input shape: {}".format(x.shape))
            x = tf.reshape(x, [-1, self.dataset.height // 2 ** num_layers, self.dataset.width // 2 ** num_layers,
                               filters_num])
            x = tf.layers.batch_normalization(x, epsilon=1e-5, momentum=self.momentum, training=True)
            x = tf.nn.relu(x)

            for i in range(1, num_layers + 1):
                print("Gen conv_trans layer-{} input shape: {}".format(i, x.shape))
                filters_num /= 2
                if i > 1:
                    strides_num = 2
                    padding_type = 'same'
                if i == num_layers:
                    filters_num = self.dataset.channels
                x = tf.layers.conv2d_transpose(x, filters=int(filters_num), kernel_size=self.kernel_size,
                                               strides=strides_num, padding=padding_type,
                                               kernel_initializer=tf.random_normal_initializer(stddev=0.02))
                x = tf.layers.batch_normalization(x, epsilon=1e-5, momentum=self.momentum, training=True)
                if i < num_layers:
                    x = tf.nn.relu(x)
                else:
                    x = tf.nn.tanh(x)
            print("Gen output shape: {}".format(x.shape))
            return x

    def discriminator(self, x, reuse=False):
        with tf.variable_scope('Discriminator', reuse=reuse):
            print("Disc input shape: {}".format(x.shape))
            x = tf.layers.conv2d(x, filters=self.latent_dim, kernel_size=self.kernel_size,
                                 padding='same', kernel_initializer=tf.random_normal_initializer(stddev=0.02))
            num_layers = 4
            for i in range(1, num_layers + 1):
                print("Disc residual block-{} input shape: {}".format(i, x.shape))
                with tf.variable_scope("DiscResidualBlock-{}".format(i)):
                    shortcut = x
                    shortcut = tf.layers.average_pooling2d(shortcut, pool_size=2, strides=2)
                    shortcut = tf.layers.conv2d(shortcut, filters=2 ** i * self.latent_dim,
                                                kernel_size=self.kernel_size, padding='same',
                                                kernel_initializer=tf.random_normal_initializer(stddev=0.02))

                    x = tf.layers.batch_normalization(x, epsilon=1e-5)
                    x = tf.nn.relu(x)
                    x = tf.layers.conv2d(x, filters=2 ** i * self.latent_dim, kernel_size=self.kernel_size,
                                         padding='same', kernel_initializer=tf.random_normal_initializer(stddev=0.02))
                    x = tf.layers.batch_normalization(x, epsilon=1e-5)
                    x = tf.nn.relu(x)
                    x = tf.layers.conv2d(x, filters=2 ** i * self.latent_dim, kernel_size=self.kernel_size,
                                         padding='same', kernel_initializer=tf.random_normal_initializer(stddev=0.02))
                    outputs = tf.layers.average_pooling2d(x, pool_size=2, strides=2)
                    x = shortcut + outputs

            print("Disc reshape layer input shape: {}".format(x.shape))
            x = tf.reshape(x, [-1, 4 * 4 * 8 * self.latent_dim])
            print("Disc dense layer input shape: {}".format(x.shape))
            x = tf.layers.dense(x, 1, kernel_initializer=tf.random_normal_initializer(stddev=0.02))
            print("Disc output shape: {}".format(x.shape))
            return x

    def build_model(self):
        self.noise_input = tf.placeholder(tf.float32, shape=[None, self.noise_dim])
        self.real_image_input = tf.placeholder(tf.float32, shape=[None, self.dataset.height, self.dataset.width,
                                                                  self.dataset.channels])

        self.gen_samples = self.generator(self.noise_input)

        real_logits = self.discriminator(self.real_image_input)
        fake_logits = self.discriminator(self.gen_samples, reuse=True)

        self.disc_loss = tf.reduce_mean(fake_logits - real_logits)
        disc_loss_sum = tf.summary.scalar('C_loss', self.disc_loss)
        self.gen_loss = tf.reduce_mean(-fake_logits)
        gen_loss_sum = tf.summary.scalar('G_loss', self.gen_loss)

        alpha = tf.random_uniform(shape=[self.dataset.batch_size, 1], minval=0., maxval=1.)
        differences = self.gen_samples - self.real_image_input
        interpolates = self.real_image_input + (alpha * differences)
        gradients = tf.gradients(self.discriminator(interpolates), [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        self.disc_loss += self.gp_lambda * gradient_penalty

        optimizer_gen = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0., beta2=0.9)
        optimizer_disc = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0., beta2=0.9)

        gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
        disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')

        self.train_gen_op = optimizer_gen.minimize(self.gen_loss, var_list=gen_vars)
        self.train_disc_op = optimizer_disc.minimize(self.disc_loss, var_list=disc_vars)

        self.disc_sum_merged = tf.summary.merge([disc_loss_sum])
        self.gen_sum_merged = tf.summary.merge([gen_loss_sum])
        self.saver = tf.train.Saver()

    def restore_model(self):
        try:
            self.saver.restore(self.session, self.model_path)
            print("Model restored from path: %s" % self.model_path)
            self.is_restore = True
        except NotFoundError:
            print("Model path not found %s" % self.model_path)

    def train_model(self, num_epochs=1000):
        steps_one_epoch = int(math.ceil(self.dataset.num_samples / self.dataset.batch_size))
        num_steps = num_epochs * steps_one_epoch
        if not self.is_restore:
            self.session.run(tf.global_variables_initializer())
        next_element = self.dataset.iterator.get_next()
        for step in range(1, num_steps + 1):
            batch_x = self.session.run(next_element)
            batch_size = batch_x.shape[0]
            # training discriminator
            for _ in range(self.disc_iters):
                z = np.random.uniform(-1, 1, [batch_size, self.noise_dim])
                feed_dict = {self.real_image_input: batch_x, self.noise_input: z}
                # _, _, disc_loss = self.session.run([self.train_disc_op, self.clip_op, self.disc_loss], feed_dict=feed_dict)
                _, disc_loss, disc_sum = self.session.run(
                    [self.train_disc_op, self.disc_loss, self.disc_sum_merged], feed_dict=feed_dict)
            self.train_writer.add_summary(disc_sum, step)

            # training generator
            z = np.random.uniform(-1, 1, [batch_size * 2, self.noise_dim])
            feed_dict = {self.noise_input: z}
            _, gen_loss, gen_sum = self.session.run(
                [self.train_gen_op, self.gen_loss, self.gen_sum_merged], feed_dict=feed_dict)
            self.train_writer.add_summary(gen_sum, step)

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
        fig, axes = plt.subplots(rows, cols, figsize=(4, 4), dpi=160)
        z = np.random.uniform(-1, 1, size=[rows * cols, self.noise_dim])
        samples = self.session.run(self.gen_samples, feed_dict={self.noise_input: z})
        count = 0
        for row in range(rows):
            for col in range(cols):
                # print(0.5*samples[count] + 0.5)
                axes[row, col].imshow(np.clip(0.5 * samples[count] + 0.5, 0, 1).squeeze(), interpolation='nearest')
                axes[row, col].axis('off')
                count += 1
        images_path = images_path if images_path else self.sample_images_path + "/sample_%d.png" % epoch
        fig.savefig(images_path)
        plt.close()

    def test(self):
        self.real_image_input = tf.placeholder(tf.float32, shape=[None, self.dataset.height, self.dataset.width,
                                                                  self.dataset.channels])
        self.discriminator(self.real_image_input)


if __name__ == "__main__":
    gds = GANDataSet(data_path='D://deep_learning/datasets/sheephead', height=64, width=64)
    wgm = WGANGPModel(model_path='D://deep_learning/models/sheephead/sheephead.ckpt',
                      sample_images_path='D://deep_learning/samples',
                      summaries_path='D://deep_learning/summaries/sheephead',
                      dataset=gds)
    if sys.argv[1] == 'train':
        wgm.build_model()
        wgm.restore_model()
        wgm.train_model(num_epochs=100)
    elif sys.argv[1] == 'sample':
        wgm.build_model()
        wgm.restore_model()
        wgm.sample_images(epoch='test')
    elif sys.argv[1] == 'test':
        wgm.test()
        # dgm.restore_model()
        # dgm.train_model(num_epochs=100)
