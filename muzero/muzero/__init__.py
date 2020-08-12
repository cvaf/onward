import random
import pickle
import numpy as np
from numpy import array, copy, log
from copy import deepcopy
from muzero.model import Network, one_hot

class MuZeroConfig:
  """Store parameters"""

  def __init__(self, environment, max_moves, discount, 
    dirichlet_alpha, num_simulations, batch_size, td_steps, num_actors, 
    known_bounds=None):

    # Self-play attributes
    self.environment = environment
    self.action_space_size = environment.action_space.n # number of actions
    self.num_actors = num_actors                # number of parallel game sims.
    self.max_moves = max_moves
    self.num_simulations = num_simulations
    self.discount = discount

    # Root prior exploration noise
    self.root_dirichlet_alpha = dirichlet_alpha
    self.root_exploration_fraction = 0.25

    # UCB formula
    self.pb_c_base = 19652
    self.pb_c_init = 1.25

    # Bounds
    self.known_bounds = known_bounds

    # Training
    # self.training_steps = int(1e6)
    self.training_steps = 1000
    # self.checkpoint_interval = int(1e3)
    self.checkpoint_interval = 100
    self.window_size = int(1e6)  # num of selfplay games to keep in the buffer
    self.batch_size = batch_size # num of games in each batch
    self.num_unroll_steps = 5 # num of game moves to keep for every batch elmnt 
    self.td_steps = td_steps  # num steps in future used to calculate trgt val
    self.weight_decay = 1e-4
    self.momentum = 0.9
    self.layer_count = 5
    self.layer_dim = 128
    self.batch_normalize = True
    self.state_dim = 128

    # Learning rate schedule
    self.learning_rate = 0.01
    # self.lr_init = 0.01
    # self.lr_decay_rate = 1 # Set to 1 for constant learning rate
    # self.lr_decay_steps = lr_decay_steps

    # Random seed
    self.seed = 42

  def new_game(self):
    return Game(self.action_space_size, self.discount, self.environment)

class SharedStorage:
  """
  Methods for saving versions of neural nets and retrieving the latest
  ones from storage.
  """

  def __init__(self):
    self._networks = {}

  def latest_network(self, config):
    if self._networks:
      return self._networks[max(self._networks.keys())]
    else:
      return Network(config)

  def save_network(self, step, network):
    self._networks[step] = network

class ReplayBuffer:
  """Store data from previous games played."""

  def __init__(self, config):
    self.window_size = config.window_size
    self.batch_size = config.batch_size
    self.buffer = []

  def save_game(self, game):
    """Add the game to our buffer"""
    if len(self.buffer) > self.window_size:
      self.buffer.pop(0)
    self.buffer.append(game)

  def sample_batch(self, num_unroll_steps, td_steps):
    games = [self.sample_game() for _ in range(self.batch_size)]
    game_positions = [(g, self.sample_position(g)) for g in games]
    return [(g.make_image(i),                    # image
             g.sample_history(i, num_unroll_steps),  # actions
             g.make_target(i, num_unroll_steps, td_steps, g.to_play(i))) # trgts
             for (g, i) in game_positions]


  def sample_game(self):
    """Sample game from buffer either uniformly or with priority"""
    # TODO: Add priority option

    p_sample = np.array([np.abs(np.sum(g.root_values) - np.sum(g.rewards))\
                         for g in self.buffer])
    p_sample -= np.min(p_sample)
    p_sample /= np.sum(p_sample)
    return np.random.choice(self.buffer, p=p_sample)

  def sample_position(self, game):
    """Sample position from a game
    Returns: position index in game history
    """
    return np.random.randint(len(game.root_values))

  def save_buffer(self, file):
    filehandler = open(file, 'wb')
    pickle.dump(self, filehandler, protocol=4)

  def load_buffer(self, file):
    filehandler = open(file, 'rb')
    rbuffer = pickle.load(filehandler)
    self.buffer.extend(rbuffer.buffer)



class Game:
  """A single episode of interaction with the environment"""

  def __init__(self, action_space_size, discount, environment):
    self.environment = deepcopy(environment)
    self.history = []
    self.rewards = []
    self.observation = self.environment.reset() # Current observation
    self.observations = [copy(self.observation)]
    self.child_visits = []         # Proportions of children visitations
    self.root_values = []          # Contains the value of each node
    self.action_space_size = action_space_size
    self.discount = discount
    self.done = False

  def terminal(self):
    """Game specific termination rules"""
    # TODO: Define termination case
    return self.done

  def apply(self, action):
    # Store the current observation
    self.observation, reward, done, _ = self.environment.step(action)
    self.observations.append(copy(self.observation))
    self.rewards.append(reward)
    self.history.append(action)
    self.done = done

  def store_search_statistics(self, root):
    """Update the search statistics"""
    sum_visits = sum(child.visit_count for child in root.children.values())
    action_space = (Action(index) for index in range(self.action_space_size))
    self.child_visits.append([
      root.children[a].visit_count / sum_visits if a in root.children else 0
      for a in action_space])
    self.root_values.append(root.value())

  def make_image(self, state_index):
    """Game specific feature planes
    Returns the corresponding observation
    """
    # TODO: We might want to catch all observations up to state_index
    return self.observations[state_index]

  def sample_history(self, i, num_unroll_steps):
    history_length = len(self.history)
    actions = [self.history[i+j] if i+j < history_length else -1
               for j in range(num_unroll_steps)]
    return actions

  def make_target(self, state_idx, num_unroll_steps, td_steps, to_play):
    """
    Using ideas from TD-learning to calculate the target value of each state
    in positions from range(state_idx, state_idx + num_unroll_steps + 1)

    Value target is the discounted root value of the search tree N steps into
    the future, plus the discounted sum of all rewards until then.
    Example:
    td_steps = 6
              c_i                           b_i
    values = [5.4, 5.1, 4.5, 3.8, 4.0, 3.2, 2.7, 2.3, 2.4, 2.0, 1.1, 0.8]
    rewards = [  0,   1,   1,   0,   0,   1,   0,   0,   0,   1,   0,   1]

    TD-learning w no discount:
      target val at c_i = val of b_i + sum(rewards in between) = 2.7 + 3 = 5.7
    MC w no discount:
      target_val at c_i = sum(all future rewards) = 5
    """
    targets = []
    for current_idx in range(state_idx, state_idx + num_unroll_steps + 1):
      
      # Get the discounted root value of the search tree td_steps in the future
      bootstrap_idx = current_idx + td_steps
      if bootstrap_idx < len(self.root_values):
        value = self.root_values[bootstrap_idx] * (self.discount**td_steps)
      else:
        value = 0

      # There's a possibility this might throw an index error for bootstrap_idx
      for i, reward in enumerate(self.rewards[current_idx:bootstrap_idx]):
        # Add the discounted reward from each position until the bootstrap idx
        value += reward * self.discount**i

      if current_idx > 0 and current_idx <= len(self.rewards):
        # TODO: Why do we subtract 1 here? The rewards are misaligned since
        # we don't observe a reward for the very first position now do we?
        last_reward = self.rewards[current_idx-1]
      else:
        last_reward = 0

      if current_idx < len(self.root_values):
        # The calculated TD target value, true reward and policy from the MCTS
        # are appended to the target list
        targets.append((value, last_reward, self.child_visits[current_idx]))
      else:
        # States past the end of games are treated as absorbing states
        targets.append((0, last_reward, array([0]*self.action_space_size)))

    return targets

  def to_play(self, i=None):
    # TODO: Generalize this
    return 1
    # if i:
    #   return self.observations[i][-1]
    # else:
    #   return self.observation[-1]

  def action_history(self):
    return ActionHistory(self.history, self.action_space_size)

  def legal_actions(self):
    try:
      return self.environment.legal_actions(self.observation)
    except:
      return [i for i in range(self.action_space_size)]

  def close(self):
    self.environment.close()

  def render(self):
    self.environment.render()
    # input('Press enter to take a step')

class Node:

  def __init__(self, prior):
    self.visit_count = 0
    self.to_play = -1
    self.prior = prior
    self.value_sum = 0
    self.children = {} # Key: action, Value: Node
    self.hidden_state = None
    self.reward = 0

  def expanded(self):
    """Boolean if node has been explored yet"""
    return len(self.children) > 0

  def value(self):
    if self.visit_count == 0:
      return 0
    return self.value_sum / self.visit_count

class MinMaxStats:
  """Holds the min-max values of the tree"""

  def __init__(self, known_bounds):
    self.maximum = known_bounds.max if known_bounds else -float('inf')
    self.minimum = known_bounds.min if known_bounds else float('inf')

  def update(self, value):
    self.maximum = max(self.maximum, value)
    self.minimum = min(self.minimum, value)

  def normalize(self, value):
    if self.maximum > self.minimum:
      return (value - self.minimum) / (self.maximum - self.minimum)
    return value

class ActionHistory:
  """History container used inside the MCTS"""

  def __init__(self, history, action_space_size):
    self.history =  list(history)
    self.action_space_size = action_space_size

  def clone(self):
    return ActionHistory(self.history, self.action_space_size)

  def add_action(self, action):
    self.history.append(action)

  def last_action(self):
    return self.history[-1]

  def action_space(self):
    return [i for i in range(self.action_space_size)]

class Action:
  # TODO: Remove this 

  def __init__(self, index):
    self.index = index

  def __hash__(self):
    return self.index

  def __eq__(self, other):
    try:
      return self.index == other.index
    except AttributeError:
      return self.index == other
    else:
      return False

  def __gt__(self, other):
    return self.index > other.index