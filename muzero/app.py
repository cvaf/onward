"""Entrypoint"""

from muzero import SharedStorage, ReplayBuffer
from muzero.play import run_selfplay
from muzero.model import train_network

# TODO: Update imports
from tqdm import tqdm


def muzero(config):
    storage = SharedStorage()
    replay_buffer = ReplayBuffer(config)

    for _ in tqdm(range(config.num_actors)):
        launch_job(run_selfplay, config, storage, replay_buffer)  # TODO: Parallelize

    # TODO: Shouldn't this whole thing be looped?
    train_network(config, storage, replay_buffer)
    return storage.latest_network()


def launch_job(f, *args):
    f(args)
