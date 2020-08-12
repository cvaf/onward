import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.layers import Dense, Input, BatchNormalization,\
  Concatenate

"""
https://github.com/geohot/ai-notebooks/blob/master/muzero/model.py
"""

def one_hot(action, action_dim):
  action_enc = np.zeros([action_dim])
  if action > -1: action_enc[action] = 1
  # return action_enc.reshape(1,-1)
  return action_enc


class Network:

  def __init__(self, config):
    self.observation_dim = config.environment.observation_space.shape
    self.action_dim = config.action_space_size
    self.state_dim = config.state_dim
    self.losses = []
    self.config = config
    self.layer_count = config.layer_count
    self.layer_dim = config.layer_dim
    self.batch_normalize = config.batch_normalize

    # h: the representation function
    # h(observation) = hidden_state
    x = observation = Input(self.observation_dim)
    for i in range(config.layer_count):
      x = Dense(config.layer_dim, activation='relu')(x)
      if i != config.layer_count-1 and config.batch_normalize:
        x = BatchNormalization()(x)
    hidden_state = Dense(self.state_dim, name='s_0')(x)
    self.h = Model(observation, hidden_state, name='h')

    # g: the dynamics function
    # g(hidden_state, action) = hidden_state_next, reward
    hidden_state = Input(self.state_dim)
    action = Input(self.action_dim)
    x = Concatenate()([hidden_state, action])
    for i in range(config.layer_count):
      x = Dense(config.layer_dim, activation='relu')(x)
      if i != config.layer_count-1 and config.batch_normalize:
        x = BatchNormalization()(x)
    hidden_state_next = Dense(self.state_dim, name='s_k')(x)
    reward = Dense(1, name='r_k')(x)
    self.g = Model([hidden_state, action], [reward, hidden_state_next], 
      name='g')

    # f: the prediction function
    # policy, value = f(hidden_state)
    x = hidden_state = Input(self.state_dim)
    for i in range(config.layer_count):
      x = Dense(config.layer_dim, activation='relu')(x)
      if i != self.layer_count-1 and self.batch_normalize:
        x = BatchNormalization()(x)
    value = Dense(1, name='v_k')(x)
    policy_logits = Dense(self.action_dim, name='p_k')(x)
    self.f = Model(hidden_state, [policy_logits, value], name='f')

    self.model = self.create_model()

  def initial_inference(self, observation):
    """h, f: o -> v, r(=0), p, s_0"""
    hidden_state = self.h(observation)
    policy_logits, value = self.f([hidden_state])
    return value, 0, policy_logits, hidden_state

  def recurrent_inference(self, hidden_state, action):
    """g, f: s_i, a -> v, r, p, s_i+1"""
    reward, hidden_state_next = self.g([hidden_state, action])
    policy_logits, value = self.f([hidden_state_next])
    return value, reward, policy_logits, hidden_state_next

  def create_model(self):

    def softmax_ce_logits(y_true, y_pred):
      return tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(
        y_true, y_pred)

    observation = Input(self.observation_dim, name='o_0')
    value, reward, policy_logits, hidden_state = self.initial_inference(
      observation)

    actions_all, mu_all, loss_all = [], [], []
    mu_all += [value, policy_logits] # TODO: Shouldn't we also add the reward?
    loss_all += ['mse', softmax_ce_logits]

    for k in range(self.config.num_unroll_steps):
      action = Input(self.action_dim, name=f'a_{k}')
      actions_all.append(action)

      value, reward, policy_logits, hidden_state_n = self.recurrent_inference(
        hidden_state, action)
      mu_all += [value, reward, policy_logits]
      loss_all += ['mse', 'mse', softmax_ce_logits]

      hidden_state = hidden_state_n # Passback

    model = Model([observation] + actions_all, mu_all)
    model.compile(Adamax(self.config.learning_rate), loss_all)
    return model

  def load_pretrained(self, file):
    model = tf.keras.models.load_model(file, compile=False)
    
    def softmax_ce_logits(y_true, y_pred):
      return tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(
        y_true, y_pred)

    loss_all = ['mse', softmax_ce_logits]
    for k in range(self.config.num_unroll_steps):
      loss_all += ['mse', 'mse', softmax_ce_logits]

    model.compile(Adamax(self.config.learning_rate), loss_all)

    self.model = model


  def save(self, file):
    tf.keras.models.save_model(self.model, file)

  def train_batch(self, batch):
    X, Y = reformat_batch(batch, self.action_dim)
    loss = self.model.train_on_batch(X, Y)
    self.losses.append(loss)
    return loss

# def reshape_X(X):
#   obs = [x[0] for x in X]
#   acts = [x[1:] for x in X]
#   return [obs, acts]

# def reshape_Y(Y):
#   for y in Y:
#     y.pop(1)
#   return Y


def reshape(array):
  array_t = [[x] for x in array[0]]
  for i in range(1, len(array)):
    for j in range(len(array[i])):
      array_t[j].append(array[i][j])
  return [np.array(x) for x in array_t]

def reformat_batch(batch, action_dim):

  X, Y = [], []
  for image, actions, targets in batch:
    # x = [image.reshape(1,-1)] + [one_hot(a, action_dim) for a in actions]
    x = [image] + [one_hot(a, action_dim) for a in actions]
    y = []
    for target in [list(t) for t in targets]:
      # target_value, target_reward, target_policy = target
      y += target
    X.append(x)
    Y.append(y)
  X = reshape(X)
  Y = reshape(Y)
  Y = [Y[0]] + Y[2:]
  return X, Y

def train_network(config, storage, replay_buffer):
  network = storage.latest_network(config)
  for i in range(config.training_steps):
    if (i % config.checkpoint_interval == 0) or (i == config.training_steps-1):
      storage.save_network(i, network)
    batch = replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps)
    network.train_batch(batch)

def extract_output(network_output):
  value, reward, policy_logits, hidden_state = network_output
  if isinstance(value, tf.Tensor): 
    value = value.numpy()[0][0]
  if isinstance(reward, tf.Tensor): 
    reward = reward.numpy()[0][0]
  if isinstance(policy_logits, tf.Tensor): 
    policy_logits = policy_logits.numpy()[0]
  if isinstance(hidden_state, tf.Tensor):
    hidden_state = hidden_state.numpy()[0]

  return value, reward, policy_logits, hidden_state