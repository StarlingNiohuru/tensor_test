import math
import os

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data


# mnist = input_data.read_data_sets("D:\deep_learning\datasets\mnist_data")


class GANDataSet(object):
    def __init__(self, data_path, height, width, channels=3, batch_size=128):
        self.data_path = data_path
        self.height = height
        self.width = width
        self.channels = channels
        self.image_shape = (self.height, self.width, self.channels)
        self.batch_size = batch_size
        self.num_samples = None
        self.dataset = None
        self.iterator = None

        self.build_dataset()

    @staticmethod
    def read_function(filename):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_image(image_string)
        return image_decoded

    def parse_function(self, filename):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string)
        image_resized = tf.image.resize_images(image_decoded, [self.height, self.width])
        return image_resized

    def build_dataset(self):
        fname_list = [self.data_path + '/' + fname for fname in os.listdir(self.data_path)]
        self.num_samples = len(fname_list)
        print('Total number of samples: %s' % self.num_samples)
        filenames = tf.constant(fname_list)
        dataset = tf.data.Dataset.from_tensor_slices(filenames)
        dataset = dataset.map(self.parse_function)
        self.dataset = dataset.shuffle(buffer_size=self.num_samples).batch(self.batch_size).repeat()
        self.iterator = self.dataset.make_one_shot_iterator()


class DCGANModel(object):
    def __init__(self, session=tf.Session(), noise_dim=100, disc_first_dim=64, gen_first_dim=64, model_path=None,
                 sample_images_path=None, dataset: GANDataSet = None):
        self.session = session
        self.noise_dim = noise_dim
        self.disc_first_dim = disc_first_dim
        self.gen_first_dim = gen_first_dim
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
            x = tf.layers.dense(x, self.dataset.height * self.dataset.width * self.gen_first_dim // 16,
                                kernel_initializer=tf.random_normal_initializer(stddev=0.02))
            x = tf.layers.batch_normalization(x, epsilon=1e-5, momentum=0.5)
            x = tf.nn.relu(x)
            x = tf.reshape(x, [-1, self.dataset.height // 16, self.dataset.width // 16, self.gen_first_dim * 16])

            x = tf.layers.conv2d_transpose(x, filters=self.gen_first_dim * 8, kernel_size=5, strides=2, padding='same',
                                           kernel_initializer=tf.random_normal_initializer(stddev=0.02))
            x = tf.layers.batch_normalization(x, epsilon=1e-5, momentum=0.5)
            x = tf.nn.relu(x)

            x = tf.layers.conv2d_transpose(x, filters=self.gen_first_dim * 4, kernel_size=5, strides=2, padding='same',
                                           kernel_initializer=tf.random_normal_initializer(stddev=0.02))
            x = tf.layers.batch_normalization(x, epsilon=1e-5, momentum=0.5)
            x = tf.nn.relu(x)

            x = tf.layers.conv2d_transpose(x, filters=self.gen_first_dim * 2, kernel_size=5, strides=2, padding='same',
                                           kernel_initializer=tf.random_normal_initializer(stddev=0.02))
            x = tf.layers.batch_normalization(x, epsilon=1e-5, momentum=0.5)
            x = tf.nn.relu(x)

            x = tf.layers.conv2d_transpose(x, filters=self.dataset.channels, kernel_size=5, strides=2, padding='same',
                                           activation=tf.nn.tanh,
                                           kernel_initializer=tf.random_normal_initializer(stddev=0.02))
            return x

    def discriminator(self, x, reuse=False):
        with tf.variable_scope('Discriminator', reuse=reuse):
            x = tf.layers.conv2d(x, filters=self.disc_first_dim, kernel_size=5, strides=2, padding='same',
                                 activation=tf.nn.leaky_relu,
                                 kernel_initializer=tf.random_normal_initializer(stddev=0.02))

            x = tf.layers.conv2d(x, filters=self.disc_first_dim * 2, kernel_size=5, strides=2, padding='same',
                                 kernel_initializer=tf.random_normal_initializer(stddev=0.02))
            x = tf.layers.batch_normalization(x, epsilon=1e-5, momentum=0.5)
            x = tf.nn.leaky_relu(x)

            x = tf.layers.conv2d(x, filters=self.disc_first_dim * 4, kernel_size=5, strides=2, padding='same',
                                 kernel_initializer=tf.random_normal_initializer(stddev=0.02))
            x = tf.layers.batch_normalization(x, epsilon=1e-5, momentum=0.5)
            x = tf.nn.leaky_relu(x)

            x = tf.layers.conv2d(x, filters=self.disc_first_dim * 8, kernel_size=5, strides=2, padding='same',
                                 kernel_initializer=tf.random_normal_initializer(stddev=0.02))
            x = tf.layers.batch_normalization(x, epsilon=1e-5, momentum=0.5)
            x = tf.nn.leaky_relu(x)

            x = tf.layers.flatten(x)
            x = tf.layers.dense(x, 2, activation=tf.nn.softmax,
                                kernel_initializer=tf.random_normal_initializer(stddev=0.02))
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
        optimizer_gen = tf.train.AdamOptimizer(learning_rate=0.0002)
        optimizer_disc = tf.train.AdamOptimizer(learning_rate=0.0002)

        # Training Variables for each optimizer
        gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
        disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')

        # Create training operations
        self.train_gen_op = optimizer_gen.minimize(self.gen_loss, var_list=gen_vars)
        self.train_disc_op = optimizer_disc.minimize(self.disc_loss, var_list=disc_vars)

    def train_model(self, num_epochs=50, sample_interval=50, save_interval=10):
        self.saver = tf.train.Saver()
        steps_one_epoch = int(math.ceil(self.dataset.num_samples / self.dataset.batch_size))
        num_steps = num_epochs * steps_one_epoch
        self.session.run(tf.global_variables_initializer())
        next_element = self.dataset.iterator.get_next()
        for step in range(1, num_steps + 1):
            batch_x = self.session.run(next_element)
            batch_size = batch_x.shape[0]
            # Generate noise to feed to the generator
            z = np.random.uniform(-1., 1., size=[batch_size, self.noise_dim])
            # Prepare Targets (Real image: 1, Fake image: 0)
            # The first half of data fed to the generator are real images,
            # the other half are fake images (coming from the generator).
            batch_disc_y = np.concatenate(
                [np.ones([batch_size]), np.zeros([batch_size])], axis=0)
            # Generator tries to fool the discriminator, thus targets are 1.
            batch_gen_y = np.ones([batch_size])
            # Training
            feed_dict = {self.real_image_input: batch_x, self.noise_input: z,
                         self.disc_target: batch_disc_y, self.gen_target: batch_gen_y}
            _, _, gen_loss, disc_loss = self.session.run(
                [self.train_gen_op, self.train_disc_op, self.gen_loss, self.disc_loss], feed_dict=feed_dict)
            epoch = int(math.ceil(step / steps_one_epoch))
            print(
                'Epoch: %i, Step: %i, Generator Loss: %f, Discriminator Loss: %f' % (epoch, step, gen_loss, disc_loss))
            if step % sample_interval == 0:
                self.sample_images(step)
            if step % save_interval == 0:
                save_path = self.saver.save(self.session, self.model_path)
                print('Model saved in path: %s' % save_path)

    def sample_images(self, step, images_path=None):
        rows = 5
        cols = 5
        fig, axes = plt.subplots(rows, cols)
        z = np.random.uniform(-1., 1., size=[rows * cols, self.noise_dim])
        samples = self.session.run(self.gen_samples, feed_dict={self.noise_input: z})
        count = 0
        for row in range(rows):
            for col in range(cols):
                axes[row, col].imshow(samples[count, :, :, 0])
                axes[row, col].axis('off')
                count += 1
        images_path = images_path if images_path else self.sample_images_path + "\sample_%d.png" % step
        fig.savefig(images_path)
        plt.close()

    def test(self):
        self.noise_input = tf.placeholder(tf.float32, shape=[None, self.noise_dim])
        self.real_image_input = tf.placeholder(tf.float32, shape=[None, self.dataset.height, self.dataset.width,
                                                                  self.dataset.channels])
        self.gen_samples = self.generator(self.noise_input)
        self.session.run(tf.global_variables_initializer())
        z = np.random.uniform(-1., 1., size=[self.dataset.batch_size, self.noise_dim])
        feed_dict = {self.noise_input: z}
        gs = self.session.run(self.gen_samples, feed_dict=feed_dict)
        print(gs.shape)


if __name__ == "__main__":
    gds = GANDataSet(data_path='D://deep_learning/datasets/17flowers', height=64, width=64)
    dgm = DCGANModel(model_path='D:/deep_learning/models/dcgan_test.ckpt',
                     sample_images_path='D://deep_learning/samples', dataset=gds)
    dgm.build_model()
    dgm.train_model()
