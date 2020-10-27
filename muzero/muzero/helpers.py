import pickle
import numpy as np

from .model import Network
from .config import MuZeroConfig
from .game import Game


class SharedStorage:
    """
    Methods for saving versions of neural nets and retrieving the latest
    ones from storage.
    """

    def __init__(self) -> None:
        self._networks = {}

    def latest_network(self, config: MuZeroConfig) -> Network:
        """
        Return the latest available network - if none are available,
        then initiate one and return that.
        """
        if self._networks:
            return self._networks[max(self._networks.keys())]
        else:
            return Network(config)

    def save_network(self, step: int, network: Network) -> None:
        """
        Save a network in our storage.
        """
        self._networks[step] = network


class ReplayBuffer:
    """Store data from previous games played."""

    def __init__(self, config: MuZeroConfig) -> None:
        self.window_size = config.window_size
        self.batch_size = config.batch_size
        self.buffer = []

    def save_game(self, game: Game) -> None:
        """Add the game to our buffer"""
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        self.buffer.append(game)

    def sample_batch(self, num_unroll_steps: int, td_steps: int) -> list:
        """
        Create and return a batch of (image, actions, targets) from our buffer.

        Input:
        :num_unroll_steps: number of game moves to keep for each batch element.
        :td_steps: number of steps in future used to calculate target value.

        Output:
        batch_size number of (image, actions, targets) tuples.
        """
        games = [self.sample_game() for _ in range(self.batch_size)]
        game_positions = [(g, self.sample_position(g)) for g in games]
        return [
            (
                g.make_image(i),  # image
                g.sample_history(i, num_unroll_steps),  # actions
                g.make_target(i, num_unroll_steps, td_steps),  # targets
            )
            for (g, i) in game_positions
        ]

    def sample_game(self, priority=True) -> Game:
        """
        Sample game from buffer either uniformly or with priority.
        The priority would be based on the absolute difference between a game's
        root values and its rewards.

        Input:
        :priority: whether to sample uniformly or with priority.

        Output:
        a game
        """

        if priority:
            p_sample = np.array(
                [np.abs(np.sum(g.root_values) - np.sum(g.rewards)) for g in self.buffer]
            )
            p_sample -= np.min(p_sample)
            p_sample /= np.sum(p_sample)
            return np.random.choice(self.buffer, p=p_sample)

        else:
            return np.random.choice(self.buffer)

    def sample_position(self, game: Game) -> int:
        """
        Sample a random position from a game.

        Output:
        a position index in the game's history.
        """
        return np.random.randint(len(game.root_values))

    def save_buffer(self, file: str) -> None:
        """Save a copy of the buffer"""
        with open(file, "wb") as f:
            pickle.dump(self, f, protocol=4)

    def load_buffer(self, file: str) -> None:
        """Load a saved copy of the buffer"""
        with open(file, "rb") as f:
            rbuffer = pickle.load(f)
        self.buffer.extend(rbuffer.buffer)
