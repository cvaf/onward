from .game import Game


class MuZeroConfig:
    """Store parameters"""

    def __init__(
        self,
        environment,
        max_moves,
        discount,
        dirichlet_alpha,
        num_simulations,
        batch_size,
        td_steps,
        num_actors,
        known_bounds=None,
    ):

        # Self-play attributes
        self.environment = environment
        self.action_space_size = environment.action_space.n  # number of actions
        self.num_actors = num_actors  # number of parallel game sims.
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
        self.training_steps = 1000
        self.checkpoint_interval = 100
        self.window_size = int(1e6)  # num of selfplay games to keep in the buffer
        self.batch_size = batch_size  # num of games in each batch
        self.num_unroll_steps = 5  # num of game moves to keep for every batch element
        self.td_steps = td_steps  # num steps in future used to calculate target value
        self.weight_decay = 1e-4
        self.momentum = 0.9
        self.layer_count = 5
        self.layer_dim = 128
        self.batch_normalize = True
        self.state_dim = 128

        # Learning rate schedule
        self.learning_rate = 0.01

        # Random seed
        self.seed = 42

    def new_game(self):
        return Game(self.action_space_size, self.discount, self.environment)
