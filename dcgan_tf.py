import tensorflow as tf
import numpy as np
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


class DCGAN(object):
    def __init__(self, image_dim=784, noise_dim=200):
        self.image_dim = image_dim
        self.noise_dim = noise_dim

    def generator(self, x, reuse=False):
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

    def discriminator(self, x, reuse=False):
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
            # Output 2 classes: Real and Fake images
            x = tf.layers.dense(x, 2)
        return x

    def build_model(self):
        self.noise_input = tf.placeholder(tf.float32, shape=[None, self.noise_dim])
        self.real_image_input = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])

        # Build Generator Network
        gen_sample = self.generator(self.noise_input)

        # Build 2 Discriminator Networks (one from noise input, one from generated samples)
        disc_real = self.discriminator(self.real_image_input)
        disc_fake = self.discriminator(gen_sample, reuse=True)
        disc_concat = tf.concat([disc_real, disc_fake], axis=0)

        # Build the stacked generator/discriminator
        stacked_gan = self.discriminator(gen_sample, reuse=True)

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
        self.train_gen = optimizer_gen.minimize(self.gen_loss, var_list=gen_vars)
        self.train_disc = optimizer_disc.minimize(self.disc_loss, var_list=disc_vars)

    def train_model(self, batch_size=64, num_steps=1000):
        self.batch_size = batch_size
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(1, num_steps + 1):
                batch_x, _ = mnist.train.next_batch(self.batch_size)
                batch_x = tf.reshape(batch_x, shape=[-1, 28, 28, 1])
                # Generate noise to feed to the generator
                z = np.random.uniform(-1., 1., size=[self.batch_size, self.noise_dim])

                # Prepare Targets (Real image: 1, Fake image: 0)
                # The first half of data fed to the generator are real images,
                # the other half are fake images (coming from the generator).
                batch_disc_y = np.concatenate(
                    [tf.ones([self.batch_size]), tf.zeros([self.batch_size])], axis=0)
                # Generator tries to fool the discriminator, thus targets are 1.
                batch_gen_y = tf.ones([self.batch_size])

                # Training
                feed_dict = {self.real_image_input: batch_x, self.noise_input: z,
                             self.disc_target: batch_disc_y, self.gen_target: batch_gen_y}
                _, _, gl, dl = sess.run([self.train_gen, self.train_disc, self.gen_loss, self.disc_loss],
                                        feed_dict=feed_dict)
