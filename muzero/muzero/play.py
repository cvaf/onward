from . import Node, MinMaxStats
import numpy as np
import random
from .model import one_hot, extract_output


def run_selfplay(config, storage, replay_buffer):

    # Question: Why do we need a while statement here?
    # while True:
    network = storage.latest_network(config)
    game = play_game(config, network)
    replay_buffer.save_game(game)


def play_game(config, network):
    game = config.new_game()

    while not game.terminal() and len(game.history) < config.max_moves:

        root = Node(0)  # Initiate a new root
        current_observation = game.make_image(-1)
        expand_node(
            root,
            game.to_play(),
            game.legal_actions(),
            network.initial_inference(current_observation.reshape(1, -1)),
        )
        add_exploration_noise(config, root)

        run_mcts(config, root, game.action_history(), network)
        action = select_action(config, len(game.history), root, network)
        game.apply(action)
        game.store_search_statistics(root)

    return game


def expand_node(node, to_play, actions, network_output):
    node.to_play = to_play
    _, node.reward, policy_logits, node.hidden_state = extract_output(network_output)
    policy = {a: np.exp(policy_logits[a]) for a in actions}
    policy_sum = sum(policy.values())
    for action, p in policy.items():
        prior = p / policy_sum if policy_sum > 0 else 0
        node.children[action] = Node(prior)


def add_exploration_noise(config, node):
    actions = list(node.children.keys())
    noise = np.random.dirichlet([config.root_dirichlet_alpha] * len(actions))
    frac = config.root_exploration_fraction
    for a, n in zip(actions, noise):
        node.children[a].prior = node.children[a].prior * (1 - frac) * (n * frac)


def run_mcts(config, root, action_history, network):
    min_max_stats = MinMaxStats(config.known_bounds)

    for _ in range(config.num_simulations):
        history = action_history.clone()  # We copy the original action history
        node = root
        search_path = [node]

        while node.expanded():
            action, node = select_child(config, node, min_max_stats)
            history.add_action(action)
            search_path.append(node)

        # We run this once we reach a node that we haven't expanded before
        parent = search_path[-2]
        network_output = network.recurrent_inference(
            parent.hidden_state[None],
            one_hot(history.last_action(), config.action_space_size)[None],
        )
        # Why is parent.to_play negative? What happens if only 1 player?
        expand_node(node, -parent.to_play, history.action_space(), network_output)
        value, _, _, _ = extract_output(network_output)

        backpropagate(search_path, value, -parent.to_play, config.discount, min_max_stats)


def select_child(config, node, min_max_stats):
    """Selecting a child within the MCTS tree by choosing the child with
    the highest UCB score.

    UCB is a measure that balances the estimated value of the action Q(s,a)
    with an exploration bonus based on the prior probability of selecting the
    action P(s,a) and the number of times the action has already been taken N(s,a)
    """
    ext = [
        (ucb_score(config, node, child, min_max_stats), action, child)
        for action, child in node.children.items()
    ]

    max_score = max([x[0] for x in ext])
    max_children = list(filter(lambda x: x[0] == max_score, ext))
    _, action, child = random.choice(max_children)

    return action, child


def ucb_score(config, parent, child, min_max_stats):
    """
    The score for a node is based on its value, plus an exploration bonus based
    on the prior.
    """
    pb_c = (
        np.log((parent.visit_count + config.pb_c_base + 1) / config.pb_c_base)
        + config.pb_c_init
    )
    pb_c *= np.sqrt(parent.visit_count) / (child.visit_count + 1)

    prior_score = pb_c * child.prior
    if child.visit_count > 0:
        if not min_max_stats:
            value_score = child.reward + config.discount * min_max_stats.normalize(
                child.value()
            )
        else:
            value_score = child.reward + config.discount * child.value()
    else:
        value_score = 0
    return prior_score + value_score


def backpropagate(search_path, value, to_play, discount, min_max_stats):
    for node in search_path[::-1]:
        # node.value_sum += value if node.to_play == to_play else -value
        node.value_sum += value
        node.visit_count += 1
        min_max_stats.update(node.value())

        value = node.reward + discount * value


def select_action(config, num_moves, node, network):
    """Action is chosen based on the number of times each child node was visited"""

    visit_counts = [
        (child.visit_count, action) for action, child in node.children.items()
    ]

    visits = np.array([i[0] for i in visit_counts])
    actions = np.array([i[1] for i in visit_counts])
    if num_moves < 30:
        p_a = visits / np.sum(visits)
        action = np.random.choice(actions, p=p_a)
    else:
        action = actions[np.argmax(visits)]

    return action
