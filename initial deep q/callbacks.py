import os
#import pickle
#import random

import numpy as np
import tensorflow as tf
from keras.models import load_model


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


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
    
    self.episode_step_count = 0
    self.step_count = 0    #counting steps over entire training
    
    if self.train or not os.path.isfile('my_model.h5'):
        self.logger.info("Setting up model from scratch.")
        

    #if not set to train, load weights from model 
    else:
        self.logger.info("Loading model from saved state.")
        self.model = load_model('my_model.h5')


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    

    # todo Exploration vs exploitation

    self.step_count += 1
    self.episode_step_count +=1
    

    if self.train  :
   
        if (np.random.rand(1)[0] <= self.epsilon):
            #print('random action')

            self.action_taken = np.random.choice(ACTIONS)
     
            return  self.action_taken 
        else:
            #predict action Q-values
            state_tensor = tf.convert_to_tensor(state_to_features(game_state))
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = self.model(state_tensor, training=False)
            # Take best action
            int_action = tf.argmax(action_probs[0]).numpy()
            self.action_taken = int_to_action(int_action)
            return  self.action_taken

    #predict action Q-values
    state_tensor = tf.convert_to_tensor(state_to_features(game_state))
    state_tensor = tf.expand_dims(state_tensor, 0)
    action_probs = self.model(state_tensor, training=False)
    # Take best action
    int_action = tf.argmax(action_probs[0]).numpy()
    self.action_taken = int_to_action(int_action)

    return  self.action_taken
    
    

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
        
    
    #game state, this is the input for the NN


    arena = game_state['field']
    feature_field = arena
    
    
    others = [xy for (n, s, b, xy) in game_state['others']]
    for o in others:
        feature_field[o[0]][o[1]] = 4
        
    explosions = game_state['explosion_map']
    
    bombs = game_state['bombs']

    bomb_map = np.ones(arena.shape) * 5
    for (xb, yb), t in bombs:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i, j] = min(bomb_map[i, j], t)
    
    #bomb timers -2 to -5, explosions -6    
    for i in range (0, arena.shape[0]):
        for j in range (0, arena.shape[1]):
            if feature_field[i][j] == 0:
                feature_field[i][j] = bomb_map[i][j] - 5
            if explosions[i][j] !=0 :
                feature_field[i][j] = -6
                     
                  
    #self xy   3
    _, score, bombs_left, (x, y) = game_state['self']
    feature_field[x][y] = 3   
        
    
    #coins 2 -  what about explosion and coin on same tile?
    coins = game_state['coins']
    for c in coins:
        feature_field[c[0]][c[1]] = 2     
    

    
    # flattened input array
    
    features_flattened = feature_field.flatten()
    

    
    
    return features_flattened


def int_to_action(index):
    actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'BOMB']
    return actions[index]
    
