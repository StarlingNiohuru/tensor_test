import math
import sys

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.framework.errors_impl import NotFoundError

from dcgan_tf import GANDataSet


class WGANModel(object):
    def __init__(self, session=tf.Session(), noise_dim=100, cri_latent_dim=128, gen_latent_dim=128, kernel_size=5,
                 gen_learning_rate=0.00005, cri_learning_rate=0.00005, momentum=0.8, dropout=0, num_critic=5,
                 clip_value=0.01, model_path=None, sample_images_path=None, summaries_path=None,
                 dataset: GANDataSet = None):
        self.session = session
        self.noise_dim = noise_dim
        self.cri_latent_dim = cri_latent_dim
        self.gen_latent_dim = gen_latent_dim
        self.kernel_size = kernel_size
        self.gen_learning_rate = gen_learning_rate
        self.cri_learning_rate = cri_learning_rate
        self.momentum = momentum
        self.dropout = dropout
        self.num_critic = num_critic
        self.clamp_upper = clip_value
        self.clamp_lower = -clip_value
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
        self.cri_target = None
        self.gen_target = None
        self.cri_loss = None
        self.gen_loss = None
        self.train_gen_op = None
        self.train_cri_op = None
        self.clip_op = None
        self.cri_sum_merged = None
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

    def critic(self, x, reuse=False):
        with tf.variable_scope('Critic', reuse=reuse):
            filters_num = self.cri_latent_dim
            strides_num = 2
            padding_type = 'same'
            num_layers = math.ceil(math.log2(int(min(x.shape[1], x.shape[2])) / self.kernel_size))
            num_layers = min(int(num_layers), 4)
            for i in range(1, num_layers + 1):
                if i == num_layers:
                    strides_num = 1
                    padding_type = 'valid'
                print("Cri conv layer-{} input shape: {}".format(i, x.shape))
                x = tf.layers.conv2d(x, filters=filters_num, kernel_size=self.kernel_size, strides=strides_num,
                                     padding=padding_type, kernel_initializer=tf.random_normal_initializer(stddev=0.02))
                if i > 1:
                    x = tf.layers.batch_normalization(x, epsilon=1e-5, momentum=self.momentum, training=True)
                x = tf.nn.leaky_relu(x)
                x = tf.layers.dropout(x, rate=self.dropout)
                filters_num *= 2

            print("Cri reshape layer input shape: {}".format(x.shape))
            x = tf.reshape(x, [-1, x.shape[1] * x.shape[2] * x.shape[3]])
            print("Cri dense layer input shape: {}".format(x.shape))
            x = tf.layers.dense(x, 1, kernel_initializer=tf.random_normal_initializer(stddev=0.02))
            print("Cri output shape: {}".format(x.shape))
            return x

    def build_model(self):
        self.noise_input = tf.placeholder(tf.float32, shape=[None, self.noise_dim])
        self.real_image_input = tf.placeholder(tf.float32, shape=[None, self.dataset.height, self.dataset.width,
                                                                  self.dataset.channels])

        self.gen_samples = self.generator(self.noise_input)

        real_logits = self.critic(self.real_image_input)
        fake_logits = self.critic(self.gen_samples, reuse=True)

        stacked_gan = self.critic(self.gen_samples, reuse=True)

        self.cri_loss = tf.reduce_mean(fake_logits - real_logits)
        cri_loss_sum = tf.summary.scalar('C_loss', self.cri_loss)
        self.gen_loss = tf.reduce_mean(-stacked_gan)
        gen_loss_sum = tf.summary.scalar('G_loss', self.gen_loss)

        optimizer_gen = tf.train.RMSPropOptimizer(learning_rate=self.gen_learning_rate)
        optimizer_cri = tf.train.RMSPropOptimizer(learning_rate=self.cri_learning_rate)

        gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
        cri_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')

        self.train_gen_op = optimizer_gen.minimize(self.gen_loss, var_list=gen_vars)
        self.train_cri_op = optimizer_cri.minimize(self.cri_loss, var_list=cri_vars)

        clip_ops_list = [tf.assign(var, tf.clip_by_value(var, self.clamp_lower, self.clamp_upper)) for var in cri_vars]
        self.clip_op = tf.group(*clip_ops_list)

        self.cri_sum_merged = tf.summary.merge([cri_loss_sum])
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
            # training critic
            for _ in range(self.num_critic):
                z = np.random.uniform(-1, 1, [batch_size, self.noise_dim])
                feed_dict = {self.real_image_input: batch_x, self.noise_input: z}
                # _, _, cri_loss = self.session.run([self.train_cri_op, self.clip_op, self.cri_loss], feed_dict=feed_dict)
                _, cri_loss, cri_sum = self.session.run(
                    [self.train_cri_op, self.cri_loss, self.cri_sum_merged], feed_dict=feed_dict)
                self.session.run(self.clip_op)
            self.train_writer.add_summary(cri_sum, step)

            # training generator
            z = np.random.uniform(-1, 1, [batch_size * 2, self.noise_dim])
            feed_dict = {self.noise_input: z}
            _, gen_loss, gen_sum = self.session.run(
                [self.train_gen_op, self.gen_loss, self.gen_sum_merged], feed_dict=feed_dict)
            self.train_writer.add_summary(gen_sum, step)

            epoch = int(math.ceil(step / steps_one_epoch))
            print(
                'Epoch: %i, Step: %i, Generator Loss: %f, Critic Loss: %f' % (epoch, step, gen_loss, cri_loss))
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
                axes[row, col].imshow(np.clip(0.5 * samples[count] + 0.5, 0, 1).squeeze(), interpolation='nearest')
                axes[row, col].axis('off')
                count += 1
        images_path = images_path if images_path else self.sample_images_path + "/sample_%d.png" % epoch
        fig.savefig(images_path)
        plt.close()


if __name__ == "__main__":
    gds = GANDataSet(data_path='D://deep_learning/datasets/sheephead', height=64, width=64)
    wgm = WGANModel(model_path='D://deep_learning/models/sheephead/sheephead.ckpt',
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
        gds = GANDataSet()
        dgm = WGANModel(model_path='D://deep_learning/models/mnist/mnist.ckpt',
                        sample_images_path='D://deep_learning/samples',
                        dataset=gds)
        dgm.build_model()
        # dgm.restore_model()
        # dgm.train_model(num_epochs=100)
