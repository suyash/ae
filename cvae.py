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
app.flags.DEFINE_string("model_dir", "models/cvae", "model dir")


def create_inference_net(latent_dim):
    inp = Input((28, 28, 1))
    net = Conv2D(
        filters=32, kernel_size=3, strides=(2, 2), activation="relu")(inp)
    net = Conv2D(
        filters=64, kernel_size=3, strides=(2, 2), activation="relu")(net)
    net = Flatten()(net)
    net = Dense(latent_dim + latent_dim, activation=None)(net)
    return Model(inp, net)


def create_generative_net(latent_dim):
    inp = Input((latent_dim, ))
    net = Dense(7 * 7 * 32)(inp)
    net = Conv2DTranspose(
        filters=64,
        kernel_size=3,
        strides=(2, 2),
        padding="SAME",
        activation="relu")(net)
    net = Conv2DTranspose(
        filters=32,
        kernel_size=3,
        strides=(2, 2),
        padding="SAME",
        activation="relu")(net)
    net = Conv2DTranspose(
        filters=1,
        kernel_size=3,
        strides=(1, 1),
        padding="SAME",
        activation=None)(net)
    return Model(inp, net)


@tf.function
def reparameterize(mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * 0.5) + mean


@tf.function
def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2.0 * np.pi)
    return tf.reduce_sum(
        -0.5 * ((sample - mean)**2.0 * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)


def process_dataset(dataset, batch_size):
    dataset = dataset.map(lambda f: f["image"])
    dataset = dataset.map(lambda f: tf.cast(f, tf.float32) / 255.0)
    dataset = dataset.map(lambda f: tf.cast(
        tf.greater_equal(f, 0.5), tf.float32))
    dataset = dataset.batch(batch_size)
    return dataset


@tf.function
def compute_loss(x, encoder, decoder):
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


@tf.function
def compute_gradients(x, encoder, decoder):
    with tf.GradientTape() as tape:
        loss = compute_loss(x, encoder, decoder)
    return tape.gradient(
        loss, encoder.trainable_variables + decoder.trainable_variables), loss


@tf.function
def apply_gradients(optimizer, gradients, variables):
    optimizer.apply_gradients(zip(gradients, variables))


@tf.function
def train_step(train_x, encoder, decoder, optimizer):
    gradients, loss = compute_gradients(train_x, encoder, decoder)
    apply_gradients(optimizer, gradients,
                    encoder.trainable_variables + decoder.trainable_variables)
    return loss


def train(latent_dim, batch_size, epochs, model_dir):
    """
    NOTE: from https://www.tensorflow.org/alpha/guide/keras/saving_and_serializing, 
    saving a imperative model is not recommended on 2.0. For creating a saved_model, "non-subclassing" branch.
    """

    with tf.summary.create_file_writer(model_dir).as_default():
        train_dataset, test_dataset = tfds.load(
            "mnist", split=[tfds.Split.TRAIN, tfds.Split.TEST])

        train_dataset = process_dataset(train_dataset, batch_size)
        test_dataset = process_dataset(test_dataset, batch_size)

        encoder = create_inference_net(latent_dim)
        decoder = create_generative_net(latent_dim)

        optimizer = tf.keras.optimizers.Adam(1e-4)

        for epoch in range(1, epochs + 1):
            for i, train_x in enumerate(train_dataset):
                loss_value = train_step(train_x, encoder, decoder, optimizer)
                tf.summary.scalar("Train ELBO", -loss_value, step=epoch * (60000 // batch_size) + i)

            loss = tf.keras.metrics.Mean()
            for test_x in test_dataset:
                loss(compute_loss(test_x, encoder, decoder))
            elbo = -loss.result()
            print("Epoch %d, Test ELBO: %f" % (epoch, elbo))

            tf.summary.scalar("Test ELBO", elbo, step=epoch)

        return encoder, decoder


def main(_):
    FLAGS = flags.FLAGS
    encoder, decoder = train(FLAGS.latent_dim, FLAGS.batch_size, FLAGS.epochs,
                             FLAGS.model_dir)

    # tf.keras.experimental.export_saved_model(
    #     encoder, "%s/export/encoder" % FLAGS.model_dir, serving_only=True)
    # tf.keras.experimental.export_saved_model(
    #     decoder, "%s/export/decoder" % FLAGS.model_dir, serving_only=True)


if __name__ == "__main__":
    app.run(main)
