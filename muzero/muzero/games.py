import numpy as np

class TicTacToe:
  observation_dim = (11,)

  def __init__(self, state=None):
    self.players = [-1, 1]
    self.reset()
    if state is not None:
      self.state = state

  def reset(self):
    self.done = False
    self.state = np.repeat(0, 11)
    self.state[-1] = 1
    return self.state

  def render(self):
    print(f'Turn {self.state[-1]}')
    print(np.array(self.state[0:9]).reshape(3,3))
    if self.done:
        if self.value(self.state) == -1:
            print(f'Winner: {-self.state[-1]}')
        else:
            print('Tie')
    
  def legal_actions(self, state):
    return [i for i, s in enumerate(state[:9]) if s==0]
  
  def value(self, state):
    reward = 0
    # Check all winning positions to determine if someone won
    for turn in self.players:
      for i in range(3):
        if all([x==turn for x in state[3*i:3*i+3]]):
          reward = turn
        if all([x==turn for x in [state[i], state[3+i], state[6+i]]]):
          reward = turn
        if all([x==turn for x in [state[0], state[4], state[8]]]):
          reward = turn
        if all([x==turn for x in [state[2], state[4], state[6]]]):
          reward = turn
    
    return reward * state[-1]

  def dynamics(self, state, action):
    """Isn't the model supposed to learn this though? Yes"""
    reward = 0
    state_next = state.copy()
    
    if state[action] != 0 or state[-2] != 0:
      # If game is over or we overwrite an action
      reward = -10 # TODO: Why hardcode the reward? low enough to disincentivize illegal moves
    else:
      state_next[action] = state_next[-1]
      reward += self.value(state_next)
    
    if state_next[-2] != 0:
      reward = 0
    else:
      state_next[-2] = self.value(state_next)

    state_next[-1] = -state_next[-1]  # Update whose turn it is
    return reward, state_next
    
  def step(self, action):
    reward, self.state = self.dynamics(self.state, action)
    if reward != 0:
      self.done = True
    if all([x!=0 for x in self.state[:9]]):
      self.done = True
    return self.state, reward, self.done, None

  def random_game(self):
    self.reset()
    while not self.done:
      random_action = np.random.choice(self.legal_actions(self.state))
      state, reward, done, _ = self.step(random_action)
#     self.render()
    
  def check_winner(self):
    assert self.done
    winner = 0
    if self.value(self.state) == -1:
      winner = -self.state[-1]
    return winner 