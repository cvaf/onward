from numpy import array, copy
from deepcopy import deepcopy


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


class Node:
    """
    A specific stage in the environment episode, storing some relevant statistics.
    """

    def __init__(self, prior: float) -> None:
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children = {}  # {action: Node}
        self.hidden_state = None
        self.reward = 0

    def expanded(self) -> bool:
        """Boolean if node has been explored yet"""
        return len(self.children) > 0

    def value(self) -> float:
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count


class ActionHistory:
    """History container used inside the MCTS"""

    def __init__(self, history, action_space_size):
        self.history = list(history)
        self.action_space_size = action_space_size

    def clone(self):
        return ActionHistory(self.history, self.action_space_size)

    def add_action(self, action: Action) -> None:
        self.history.append(action)

    def last_action(self) -> Action:
        return self.history[-1]

    def action_space(self) -> list:
        return [i for i in range(self.action_space_size)]


class Game:
    """A single episode of interaction with the environment"""

    def __init__(self, action_space_size: int, discount: float, environment) -> None:
        self.environment = deepcopy(environment)
        self.history = []
        self.rewards = []
        self.observation = self.environment.reset()  # Current observation
        self.observations = [copy(self.observation)]
        self.child_visits = []  # Proportions of children visitations
        self.root_values = []  # Contains the value of each node
        self.action_space_size = action_space_size
        self.discount = discount
        self.done = False

    def terminal(self) -> bool:
        """Game specific termination rules"""
        # TODO: Define termination case
        return self.done

    def apply(self, action: Action) -> None:
        """
        Apply an action in our environment and store the returned observation,
        reward and whether the environment has terminated.
        """
        self.observation, reward, done, _ = self.environment.step(action)
        self.observations.append(copy(self.observation))
        self.rewards.append(reward)
        self.history.append(action)
        self.done = done

    def store_search_statistics(self, root: Node) -> None:
        """Update the search statistics"""
        sum_visits = sum(child.visit_count for child in root.children.values())
        action_space = (Action(index) for index in range(self.action_space_size))
        self.child_visits.append(
            [
                root.children[a].visit_count / sum_visits if a in root.children else 0
                for a in action_space
            ]
        )
        self.root_values.append(root.value())

    def make_image(self, state_index: int) -> array:
        """Game specific feature planes
        Returns the corresponding observation
        """
        # TODO: We might want to catch all observations up to state_index
        return self.observations[state_index]

    def sample_history(self, i: int, num_unroll_steps: int) -> list:
        """
        Return the list of actions that happened between i and i+num_unroll_steps
        """
        history_length = len(self.history)
        actions = [
            self.history[i + j] if i + j < history_length else -1
            for j in range(num_unroll_steps)
        ]
        return actions

    def make_target(self, state_idx: int, num_unroll_steps: int, td_steps: int) -> list:
        """
        Using ideas from TD-learning to calculate the target value of each state
        in positions from range(state_idx, state_idx + num_unroll_steps + 1)

        Input:

        :state_idx: int; first state index
        :num_unroll_steps: int; how many states to find the target for
        :td_steps: int; for how many steps to use actual rewards before switching
            to the discounted root value.

        Output:

        :targets: list; list of targets of each state.

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
                value = self.root_values[bootstrap_idx] * (self.discount ** td_steps)
            else:
                value = 0

            # There's a possibility this might throw an index error for bootstrap_idx
            for i, reward in enumerate(self.rewards[current_idx:bootstrap_idx]):
                # Add the discounted reward from each position until the bootstrap idx
                value += reward * self.discount ** i

            if current_idx > 0 and current_idx <= len(self.rewards):
                # TODO: Why do we subtract 1 here? The rewards are misaligned since
                # we don't observe a reward for the very first position now do we?
                last_reward = self.rewards[current_idx - 1]
            else:
                last_reward = 0

            if current_idx < len(self.root_values):
                # The calculated TD target value, true reward and policy from the MCTS
                # are appended to the target list
                targets.append((value, last_reward, self.child_visits[current_idx]))
            else:
                # States past the end of games are treated as absorbing states
                targets.append((0, last_reward, array([0] * self.action_space_size)))

        return targets

    def to_play(self, i=None) -> int:
        # TODO: Generalize this
        return 1

    def action_history(self) -> ActionHistory:
        return ActionHistory(self.history, self.action_space_size)

    def legal_actions(self) -> list:
        """
        Find the possible actions at the current observation.
        """
        try:
            return self.environment.legal_actions(self.observation)
        except Exception:
            return [i for i in range(self.action_space_size)]

    def close(self) -> None:
        self.environment.close()

    def render(self) -> None:
        self.environment.render()
        # input('Press enter to take a step')


class MinMaxStats:
    """Holds the min-max values of the tree"""

    def __init__(self, known_bounds) -> None:
        self.maximum = known_bounds.max if known_bounds else -float("inf")
        self.minimum = known_bounds.min if known_bounds else float("inf")

    def update(self, value: float) -> None:
        """Updates the max and min bounds based on the input value."""
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float):
        """Scales the value input according to our max/min."""
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value
