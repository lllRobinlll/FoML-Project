import pickle
import random
from typing import List

import events as e
from .callbacks import state_to_features

import numpy as np
import tensorflow as tf

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.num_inputs = 8
    self.gamma = .95  # discount factor   try: .95, .99
    self.learning_rate = .001  # try: .001, .01
    self.eps = .0000001  # small number that avoids zero div

    if self.model is None:
        self.model = create_model(self)

    # setup arrays that will note (s, a, r) transitions of each step
    self.states, self.probs, self.gradients, self.actions, self.rewards = [], [], [], [], []

    self.scores = []
    self.current_round = 1


def create_model(self):
    state_inputs = tf.keras.layers.Input(shape=(self.num_inputs,))
    x = tf.keras.layers.Dense(24, kernel_initializer="he_uniform", activation='relu')(state_inputs)
    x = tf.keras.layers.Dense(24, kernel_initializer="he_uniform", activation='relu')(x)
    outputs = tf.keras.layers.Dense(len(self.ACTIONS), kernel_initializer="he_uniform", activation="softmax")(x)
    model = tf.keras.models.Model(inputs=state_inputs, outputs=outputs)
    model.summary()
    # compile model using categorical crossentropy as loss
    adam = tf.keras.optimizers.Adam(lr=self.learning_rate)
    model.compile(loss="categorical_crossentropy", optimizer=adam)
    return model


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Idea: Add your own events to hand out rewards
    # if ...:
    #     events.append(PLACEHOLDER_EVENT)

    # state_to_features is defined in callbacks.py
    if not old_game_state is None: 
        memorize(self, old_game_state, self_action, events)


def memorize(self, state, action, events):
    encoded_action = np.zeros((len(self.ACTIONS),), np.float32)
    encoded_action[self.ACTIONS.index(action)] = 1
    state = state_to_features(state)
    self.states.append(state)
    state = tf.convert_to_tensor(state)
    state = tf.expand_dims(state, 0)
    prob = self.model.predict(state).flatten()
    self.probs.append(prob)
    self.gradients.append(encoded_action - prob)
    self.actions.append(action)
    self.rewards.append(reward_from_events(self, events))


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    memorize(self, last_game_state, last_action, events)

    self.episode_length = len(self.states)
    self.reward_sum = np.sum(self.rewards)
    self.scores.append(self.reward_sum)

    # train the model
    update_policy(self)
    self.states, self.probs, self.gradients, self.actions, self.rewards = [], [], [], [], []

    # store the model
    if self.current_round % 50 == 0:
        self.model.save('my-saved-model.h5')

    # print("episode:", self.current_round, "  score:", self.reward_sum, "  length:", self.episode_length)
    print("episode:", self.current_round, "  score:", np.mean(self.scores[-10:]))

    self.current_round += 1


def discount_rewards(self, rewards):
    discounted_rewards = np.zeros_like(rewards)
    discounted_sum = 0
    for t in reversed(range(len(rewards))):
        discounted_sum = rewards[t] + discounted_sum * self.gamma
        discounted_rewards[t] = discounted_sum

    discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + self.eps)
    return discounted_rewards


def update_policy(self):
    discounted_rewards = discount_rewards(self, self.rewards)
    update_inputs = np.zeros((self.episode_length, self.num_inputs))
    adv = np.zeros((self.episode_length, len(self.ACTIONS)))
    for i in range(self.episode_length):
        update_inputs[i] = self.states[i]
        adv[i] = self.probs[i] + self.learning_rate * self.gradients[i] * discounted_rewards[i]

    self.model.fit(update_inputs, adv, epochs=1, verbose=0)
    return


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        # PLACEHOLDER_EVENT: -.1  # idea: the custom event is bad
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
