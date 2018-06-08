import os

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("D:\deep_learning\datasets\mnist_data")


class ImagesDataSet(object):
    def __init__(self, data_path, height, width, channels):
        self.data_path = data_path
        self.height = height
        self.width = width
        self.channels = channels
        self.image_shape = (self.height, self.width, self.channels)

    def parse_function(self, filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_image(image_string)
        image_resized = tf.image.resize_images(image_decoded, [self.height, self.width])
        return image_resized, label

    def load_dataset(self):
        filenames = tf.constant([self.data_path + fname for fname in os.listdir(self.data_path)])
        dataset = tf.data.Dataset.from_tensor_slices(filenames)
        # dataset = dataset.map(_parse_function)
        return dataset


class DCGAN(object):
    def __init__(self, session=tf.Session(), image_dim=784, noise_dim=200,
                 sample_images_path='D:\deep_learning\samples'):
        self.session = session
        self.image_dim = image_dim
        self.noise_dim = noise_dim
        self.sample_images_path = sample_images_path

        self.noise_input = None
        self.real_image_input = None
        self.gen_samples = None
        self.disc_target = None
        self.gen_target = None
        self.disc_loss = None
        self.gen_loss = None
        self.train_gen_op = None
        self.train_disc_op = None

    @staticmethod
    def generator(x, reuse=False):
        with tf.variable_scope('Generator', reuse=reuse):
            x = tf.layers.dense(x, units=6 * 6 * 128)
            x = tf.nn.tanh(x)
            # Reshape to a 4-D array of images: (batch, height, width, channels)
            # New shape: (batch, 6, 6, 128)
            x = tf.reshape(x, shape=[-1, 6, 6, 128])
            # Deconvolution, image shape: (batch, 14, 14, 64)
            x = tf.layers.conv2d_transpose(x, 64, 4, strides=2)
            # Deconvolution, image shape: (batch, 28, 28, 1)
            x = tf.layers.conv2d_transpose(x, 1, 2, strides=2)
            # Apply sigmoid to clip values between 0 and 1
            x = tf.nn.sigmoid(x)
            return x

    @staticmethod
    def discriminator(x, reuse=False):
        with tf.variable_scope('Discriminator', reuse=reuse):
            x = tf.layers.conv2d(x, 64, 5)
            x = tf.nn.tanh(x)
            x = tf.layers.average_pooling2d(x, 2, 2)
            x = tf.layers.conv2d(x, 128, 5)
            x = tf.nn.tanh(x)
            x = tf.layers.average_pooling2d(x, 2, 2)
            x = tf.contrib.layers.flatten(x)
            x = tf.layers.dense(x, 1024)
            x = tf.nn.tanh(x)
            x = tf.layers.dense(x, 2)
            return x

    def build_model(self):
        self.noise_input = tf.placeholder(tf.float32, shape=[None, self.noise_dim])
        self.real_image_input = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])

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
        optimizer_gen = tf.train.AdamOptimizer(learning_rate=0.001)
        optimizer_disc = tf.train.AdamOptimizer(learning_rate=0.001)

        # Training Variables for each optimizer
        gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
        disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')

        # Create training operations
        self.train_gen_op = optimizer_gen.minimize(self.gen_loss, var_list=gen_vars)
        self.train_disc_op = optimizer_disc.minimize(self.disc_loss, var_list=disc_vars)

    def train_model(self, batch_size=64, num_steps=10000, sample_interval=50):
        self.batch_size = batch_size
        self.session.run(tf.global_variables_initializer())
        for step in range(1, num_steps + 1):
            batch_x, _ = mnist.train.next_batch(self.batch_size)
            batch_x = np.reshape(batch_x, newshape=[-1, 28, 28, 1])
            # Generate noise to feed to the generator
            z = np.random.uniform(-1., 1., size=[self.batch_size, self.noise_dim])
            # Prepare Targets (Real image: 1, Fake image: 0)
            # The first half of data fed to the generator are real images,
            # the other half are fake images (coming from the generator).
            batch_disc_y = np.concatenate(
                [np.ones([self.batch_size]), np.zeros([self.batch_size])], axis=0)
            # Generator tries to fool the discriminator, thus targets are 1.
            batch_gen_y = np.ones([self.batch_size])
            # Training
            feed_dict = {self.real_image_input: batch_x, self.noise_input: z,
                         self.disc_target: batch_disc_y, self.gen_target: batch_gen_y}
            _, _, gen_loss, disc_loss = self.session.run(
                [self.train_gen_op, self.train_disc_op, self.gen_loss, self.disc_loss], feed_dict=feed_dict)
            print('Step %i: Generator Loss: %f, Discriminator Loss: %f' % (step, gen_loss, disc_loss))
            if step % sample_interval == 0:
                self.sample_images(step)

    def sample_images(self, step, images_path=None):
        rows = 5
        cols = 5
        fig, axes = plt.subplots(rows, cols)
        z = np.random.uniform(-1., 1., size=[rows * cols, self.noise_dim])
        samples = self.session.run(self.gen_samples, feed_dict={self.noise_input: z})
        count = 0
        for row in range(rows):
            for col in range(cols):
                axes[row, col].imshow(samples[count, :, :, 0], cmap='gray')
                axes[row, col].axis('off')
                count += 1
        images_path = images_path if images_path else self.sample_images_path + "\mnist_%d.png" % step
        fig.savefig(images_path)
        plt.close()


if __name__ == "__main__":
    dg = DCGAN()
    dg.build_model()
    dg.train_model()
