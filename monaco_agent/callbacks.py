import os
import pickle
import random

import numpy as np
import tensorflow as tf


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.load_model = True

    if (self.train and not self.load_model) or not os.path.isfile('my-saved-model.h5'):
        self.logger.info("Setting up model from scratch.")
        self.model = None
    else:
        self.logger.info("Loading model from saved state.")
        self.model = tf.keras.models.load_model('my-saved-model.h5')

    self.ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
    self.eps = .0000001


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    self.logger.debug("Querying model for action.")
    state = state_to_features(game_state)
    state = tf.convert_to_tensor(state)
    state = tf.expand_dims(state, 0)
    action_probability_distribution = self.model.predict(state).flatten()
    action_probability_distribution += np.ones(len(self.ACTIONS)) * self.eps  # adding a small number that avoids determinism
    action_probability_distribution = action_probability_distribution / np.sum(action_probability_distribution)
    return np.random.choice(self.ACTIONS, p=action_probability_distribution)


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # For example, you could construct several channels of equal shape, ...
    channels = []
    # channels.append(...)

    _, _, _, (x, y) = game_state['self']
    coin_dirs = closest_coin_directions((x, y), game_state['coins'])
    channels.append(abs(game_state['field'][x + 1, y]))
    channels.append(coin_dirs[0])
    channels.append(abs(game_state['field'][x, y - 1]))
    channels.append(coin_dirs[1])
    channels.append(abs(game_state['field'][x - 1, y]))
    channels.append(coin_dirs[2])
    channels.append(abs(game_state['field'][x, y + 1]))
    channels.append(coin_dirs[3])

    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    return stacked_channels.reshape(-1)

def closest_coin_directions(agent, coins):
    if len(coins) == 0:
        return [0, 0, 0, 0]
    mdist = 30 ** 5
    for coords in coins:
        hdist = abs(agent[0] - coords[0])
        vdist = abs(agent[1] - coords[1])
        if hdist + vdist < mdist:
            mdist = hdist + vdist
            mcoords = coords
    dirs = []
    dirs.append(max(0, mcoords[0] - agent[0]))
    dirs.append(max(0, agent[1] - mcoords[1]))
    dirs.append(max(0, agent[0] - mcoords[0]))
    dirs.append(max(0, mcoords[1] - agent[1]))
    return dirs


def crates_in_range(agent, field):
    x, y = agent
    num_crates = 0
    width, height = field.shape
    for i in range(1, 4):
        if x + i < width:
            num_crates += max(0, field[x + i, y])
        if y - i >= 0:
            num_crates += max(0, field[x, y - 1])
        if x - i >= 0:
            num_crates += max(0, field[x - i, y])
        if y + i < height:
            num_crates += max(0, field[x, y + 1])