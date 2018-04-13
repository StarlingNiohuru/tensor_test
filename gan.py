import sys
from keras import Sequential, Input, Model
from keras.datasets import mnist
from keras.layers import Dense, BatchNormalization, LeakyReLU, Reshape, Flatten
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import Adam


class GANHandler(object):

    def __init__(self, image_rows=28, image_cols=28, channels=1, batch_size=128, noise_dim=100,
                 optimizer=Adam(0.0002, 0.5)):
        self.image_rows = image_rows
        self.image_cols = image_cols
        self.channels = channels
        self.image_shape = (self.image_rows, self.image_cols, self.channels)
        self.batch_size = batch_size
        self.noise_dim = noise_dim
        self.generator = None
        self.discriminator = None
        self.combined = None
        self.optimizer = optimizer
        self.data_path = 'D:\deep_learning\datasets\mnist.npz'
        self.sample_images_path = 'D:\deep_learning\samples'
        self.generator_model_path = 'D:\deep_learning\models\generator.hdf5'
        self.discriminator_model_path = 'D:\deep_learning\models\discriminator.hdf5'

    def build_generator(self):
        noise_shape = (self.noise_dim,)

        model = Sequential()
        model.add(Dense(256, input_shape=noise_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.image_shape), activation='tanh'))
        model.add(Reshape(self.image_shape))
        model.summary()

        noise = Input(shape=noise_shape)
        image = model(noise)
        self.generator = Model(noise, image)
        self.generator.compile(loss='binary_crossentropy', optimizer=self.optimizer)

    def build_discriminator(self):
        model = Sequential()
        model.add(Flatten(input_shape=self.image_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        image = Input(shape=self.image_shape)
        validity = model(image)
        self.discriminator = Model(image, validity)
        self.discriminator.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['acc'])

    def build_combined(self):
        # The generator takes noise as input and generates images
        z = Input(shape=(self.noise_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=self.optimizer)

    def load_generator(self):
        self.build_generator()
        self.generator.load_weights(filepath=self.generator_model_path)

    def load_discriminator(self):
        self.build_discriminator()
        self.discriminator.load_weights(filepath=self.discriminator_model_path)

    def save_models_weights(self):
        self.discriminator.save_weights(self.discriminator_model_path, True)
        self.generator.save_weights(self.generator_model_path, True)

    def train(self, epochs, k=1, sample_interval=50):
        # Load the dataset
        (X_train, _), (_, _) = mnist.load_data(self.data_path)

        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)

        half_batch = int(self.batch_size / 2)

        for epoch in range(epochs):
            for _ in range(k):
                # Select a random half batch of real images
                idx = np.random.randint(0, X_train.shape[0], half_batch)
                real_images = X_train[idx]

                # Generate a half batch of new fake images
                noise = np.random.normal(0, 1, (half_batch, self.noise_dim))
                fake_images = self.generator.predict(noise)

                # Train the discriminator
                self.discriminator.trainable = True
                d_loss_real = self.discriminator.train_on_batch(real_images, np.ones((half_batch, 1)))
                d_loss_fake = self.discriminator.train_on_batch(fake_images, np.zeros((half_batch, 1)))
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            noise = np.random.normal(0, 1, (self.batch_size, self.noise_dim))
            # The generator wants the discriminator to label the generated samples as valid (ones)
            valid_y = np.array([1] * self.batch_size)

            # Train the generator in the combined model
            g_loss = self.combined.train_on_batch(noise, valid_y)
            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)
                self.save_models_weights()

    def sample_images(self, epoch, images_path=None):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.noise_dim))
        gen_images = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_images = 0.5 * gen_images + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_images[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        images_path = images_path if images_path else self.sample_images_path + "\mnist_%d.png" % epoch
        fig.savefig(images_path)
        plt.close()


if __name__ == '__main__':
    gan = GANHandler()
    if sys.argv[1] in ['build', 'continue']:
        if sys.argv[1] == 'build':
            gan.build_generator()
            gan.build_discriminator()
            gan.build_combined()
            gan.train(epochs=30000)
        elif sys.argv[1] == 'continue':
            gan.load_generator()
            gan.load_discriminator()
            gan.build_combined()
            gan.train(epochs=30000)
    elif sys.argv[1] == 'generate':
        gan.load_generator()
        gan.sample_images('test', 'D:/deep_learning/test.png')
