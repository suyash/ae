"""
Convolutional Variational Autoencoder

Implemented based on https://www.tensorflow.org/alpha/tutorials/generative/cvae for 2.0
"""

from absl import app, flags
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Dense, Input, Conv2D, Conv2DTranspose, Flatten, Reshape
import tensorflow_datasets as tfds

app.flags.DEFINE_integer("latent_dim", 64, "latent dimension")
app.flags.DEFINE_integer("batch_size", 100, "batch size")
app.flags.DEFINE_integer("epochs", 50, "epochs")


class InferenceNet(Model):
    def __init__(self, latent_dim, **kwargs):
        super(InferenceNet, self).__init__(**kwargs)

        self.conv1 = Conv2D(
            filters=32, kernel_size=3, strides=(2, 2), activation="relu")
        self.conv2 = Conv2D(
            filters=64, kernel_size=3, strides=(2, 2), activation="relu")
        self.dense = Dense(latent_dim + latent_dim, activation=None)

    def call(self, inp):
        net = self.conv1(inp)
        net = self.conv2(net)
        net = Flatten()(net)
        net = self.dense(net)
        return net


class GenerativeNet(Model):
    def __init__(self, latent_dim, **kwargs):
        super(GenerativeNet, self).__init__(**kwargs)

        self.dense = Dense(7 * 7 * 32)
        self.convt1 = Conv2DTranspose(
            filters=64,
            kernel_size=3,
            strides=(2, 2),
            padding="SAME",
            activation="relu")
        self.convt2 = Conv2DTranspose(
            filters=32,
            kernel_size=3,
            strides=(2, 2),
            padding="SAME",
            activation="relu")
        self.convt3 = Conv2DTranspose(
            filters=1,
            kernel_size=3,
            strides=(1, 1),
            padding="SAME",
            activation=None)

    def call(self, inp):
        net = self.dense(inp)
        net = Reshape((7, 7, 32))(net)
        net = self.convt1(net)
        net = self.convt2(net)
        net = self.convt3(net)
        return net


def process_dataset(dataset, batch_size):
    dataset = dataset.map(lambda f: f["image"])
    dataset = dataset.map(lambda f: tf.cast(f, tf.float32) / 255.0)
    dataset = dataset.map(lambda f: tf.cast(
        tf.greater_equal(f, 0.5), tf.float32))
    dataset = dataset.batch(batch_size)
    return dataset


def train(latent_dim, batch_size, epochs):
    """
    NOTE: from https://www.tensorflow.org/alpha/guide/keras/saving_and_serializing, 
    saving a imperative model is not recommended on 2.0. For creating a saved_model, "non-subclassing" branch.
    """
    train_dataset, test_dataset = tfds.load(
        "mnist", split=[tfds.Split.TRAIN, tfds.Split.TEST])

    train_dataset = process_dataset(train_dataset, batch_size)
    test_dataset = process_dataset(test_dataset, batch_size)

    encoder = InferenceNet(latent_dim)
    decoder = GenerativeNet(latent_dim)

    def compute_loss(x):
        enc = encoder(x)
        mean, logvar = tf.split(enc, num_or_size_splits=2, axis=1)
        z = reparameterize(mean, logvar)
        x_logit = decoder(z)

        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = log_normal_pdf(z, 0.0, 0.0)
        logqz_x = log_normal_pdf(z, mean, logvar)
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)

    def reparameterize(mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * 0.5) + mean

    def log_normal_pdf(sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2.0 * np.pi)
        return tf.reduce_sum(
            -0.5 * ((sample - mean)**2.0 * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis)

    def compute_gradients(x):
        with tf.GradientTape() as tape:
            loss = compute_loss(x)
        return tape.gradient(
            loss,
            encoder.trainable_variables + decoder.trainable_variables), loss

    def apply_gradients(optimizer, gradients, variables):
        optimizer.apply_gradients(zip(gradients, variables))

    optimizer = tf.keras.optimizers.Adam(1e-4)

    for epoch in range(1, epochs + 1):
        for train_x in train_dataset:
            gradients, loss = compute_gradients(train_x)
            apply_gradients(
                optimizer, gradients,
                encoder.trainable_variables + decoder.trainable_variables)

        loss = tf.keras.metrics.Mean()
        for test_x in test_dataset:
            loss = compute_loss(test_x)
        elbo = -loss.result()
        print("Epoch %d, Test ELBO: %f" % (epoch, elbo))

    return encoder, decoder


def main(_):
    FLAGS = flags.FLAGS
    train(FLAGS.latent_dim, FLAGS.batch_size, FLAGS.epochs)


if __name__ == "__main__":
    app.run(main)
