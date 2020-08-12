import os, sys
sys.path.append(os.getcwd())

import numpy as np
from tensorflow.keras import Model, layers, Sequential
from tensorflow.nn import relu
import tensorflow as tf
import tensorflow_probability as tfp

from vae.config import config

class VAE(tf.keras.Model):

  def __init__(self, config):
    super(VAE, self).__init__()
    self.config = config

    self.encoder = Sequential([
      layers.Conv2D(filters=32, kernel_size=3, strides=(2,2), 
        activation='relu',input_shape=config.img_shape),
      layers.Conv2D(
        filters=64, kernel_size=3, strides=(2,2), activation='relu'),
      layers.Flatten(),
      layers.Dense(config.latent_dim + config.latent_dim)
    ])

    self.decoder = Sequential([
      layers.Dense(units=7*7*32, activation=relu, 
        input_shape=(config.latent_dim, )),
      layers.Reshape(target_shape=(7, 7, 32)),
      layers.Conv2DTranspose(
        filters=64, kernel_size=3, strides=2, padding='same', 
        activation='relu'),
      layers.Conv2DTranspose(
        filters=32, kernel_size=3, strides=2, padding='same', 
        activation='relu'),
      layers.Conv2DTranspose(
        filters=1, kernel_size=3, strides=1, padding='same')
    ])

  @tf.function
  def sample(self, eps=None):
    if eps is None:
      eps = tf.random.normal(shape=(100, self.config.latent_dim))
    return self.decode(eps, apply_sigmoid=True)

  def encode(self, x):
    mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
    return mean, logvar

  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean

  def decode(self, z, apply_sigmoid=False):
    logits = self.decoder(z)
    if apply_sigmoid:
      probs = tf.sigmoid(logits)
      return probs
    return logits


optimizer = tf.keras.optimizers.Adam(1e-4)

def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(
    -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
    axis=raxis)

def compute_loss(model, x):
  mean, logvar = model.encode(x)
  z = model.reparameterize(mean, logvar)
  x_logit = model.decode(z)
  cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
    logits=x_logit, labels=x)
  logpx_z = -tf.reduce_sum(cross_entropy, axis=[1,2,3])
  logpz = log_normal_pdf(z, 0., 0.)
  logqz_x = log_normal_pdf(z, mean, logvar)
  return -tf.reduce_mean(logpx_z + logpz - logqz_x)


@tf.function
def train_step(model, x, optimizer):
  """A single training step"""
  with tf.GradientTape() as tape:
    loss = compute_loss(model, x)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

