'''
The code in this initial implementation relies heavily on the example in the keras documentation under
https://keras.io/examples/rl/deep_q_network_breakout/
'''
from collections import namedtuple, deque
from typing import List

import events as e
from .callbacks import state_to_features


import os
import numpy as np
import pandas as pd
import tensorflow as tf
#from tensorflow import keras
#from tensorflow.keras import layers

import keras
from keras.layers import Dense
from keras.layers import Input
from keras.optimizers import Adam
from keras.models import Sequential
from keras.models import load_model
from keras import backend as K

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"


###############################


# Configuration paramaters for the whole setup

gamma = 0.99  # Discount factor for past rewards


epsilon_min = 0.1  # Minimum epsilon greedy parameter
epsilon_max = 1.0  # Maximum epsilon greedy parameter


epsilon_interval = (epsilon_max - epsilon_min)  
epsilon_random_steps = 50000


# Number of frames for exploration
epsilon_greedy_steps = 1000000.0

batch_size = 32  # Size of batch taken from memory



max_memory_length = 100000 

# Train the model after 4 actions
update_after_actions = 4


# How often to update the target network

update_target_network = 2000


#############################################################

num_actions = 6  #['RIGHT', 'LEFT', 'UP', 'DOWN', 'BOMB'] +WAIT?

    
# Using huber loss for stability  
loss_function = keras.losses.Huber()    
    
optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

#####################################################




#start with 2 layers, 32 and 63 nodes

def create_model():
    inputs = Input(shape = (289, ))
    layer1 = Dense(32, activation="relu")(inputs)
    layer2 = Dense(64, activation="relu")(layer1)
    action = Dense(num_actions, activation = "linear")(layer2)

    return keras.Model(inputs=inputs, outputs = action)


def action_to_int(action):
    actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'BOMB']
    return actions.index(action)

#########################

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    
    print("agent is training")
    if os.path.isfile('my_model.h5'):
        self.model = load_model('my_model.h5')
        self.model_target = load_model('my_model.h5')
        print('model loaded')
        print(self.model.summary())
        #print(self.model.weights)
        
        #print('optimizer states')
        #loaded_optimizer_states = [K.eval(w) for w in self.model.optimizer.weights]
        #print(loaded_optimizer_states)
    else:
        print('no saved model, creating model')
        self.model = create_model()
        self.model_target = create_model()
        self.model.compile(loss='huber', optimizer=optimizer)
    #print(self.model.summary())
    

    
    self.epsilon = epsilon_max   #start with 1.0
    


    
    # Experiences of the agent
    self.action_history = []
    self.state_history = []
    self.state_history_last = []
    self.rewards_history = []
    self.done_history = []
    self.episode_reward_history = []
    ####################################
    self.episode_reward = 0  
    
    self.rewards_data = []

    #########################################
    self.running_reward = 0
    self.episode_count = 0
   
    self.total_reward = 0
    self.action_taken = None
    
    
    ###########################################
    
    
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)


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
    #########################
    #update state 
    
    state_current= state_to_features(new_game_state)
    
    state_last = state_to_features(old_game_state)


    # update exploration probability epsilon 
    #print(self.episode_step_count)
    

    if self.step_count > epsilon_random_steps:
        self.epsilon -= epsilon_interval / epsilon_greedy_steps
        self.epsilon = max(self.epsilon, epsilon_min)
        

    
    # if step > 0 update current reward and total reward
    
    reward_received = reward_from_events(self, events)
    self.episode_reward += reward_received
    self.total_reward += reward_received
    
    #add experience to replay buffer
    
    #done_history: history of bools episode_over - whether it's time to reset the episode again
    #do i need done_history??
    '''
    if len(self.action_history) > 0:
        self.done_history.append(self.action_history[-1])
    '''
    if self.episode_step_count > 0:
        self.rewards_history.append(reward_received)
            
        self.action_history.append(action_to_int(self.action_taken))
        
        self.state_history_last.append(state_last)    
        
        self.state_history.append(state_current)
    


    if self.step_count % update_after_actions == 0 and len(self.state_history_last) > batch_size: 
        # Get indices of samples for replay buffers
        indices = np.random.choice(range(len(self.state_history_last)), size=batch_size)
        
  
        state_sample_last = np.array([self.state_history_last[i] for i in indices])
        

 
        state_sample = np.array([self.state_history[i] for i in indices])
        
        rewards_sample = [self.rewards_history[i] for i in indices]
        action_sample = [self.action_history[i] for i in indices]

        


        
        future_rewards = self.model_target.predict(state_sample, batch_size = batch_size)

        updated_q_values = rewards_sample + gamma * tf.reduce_max(future_rewards, axis=1)
    
        

        
        # train the Q-network

        
  
        
        ######################

        # Create a mask so we only calculate loss on the updated Q-values
        masks = tf.one_hot(action_sample, num_actions)
        
        with tf.GradientTape() as tape:
                # Train the model on the states and updated Q-values
                q_values = self.model(state_sample_last)

                # Apply the masks to the Q-values to get the Q-value for action taken
                q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                # Calculate loss between new Q-value and old Q-value
                loss = loss_function(updated_q_values, q_action)     
        
        # backpropagate and update DQN with the minibatch
        grads = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    
    # if number of updates to DQN since last update to target network, then 
    #     update the target Q-networtk
    
    
    if self.step_count > 0 and self.step_count % update_target_network == 0:
        # update the the target network with new weights
        self.model_target.set_weights(self.model.get_weights())
        # Log details
        template = "running reward: {:.2f} at episode {}, total step count {}"
        print(template.format(self.running_reward, self.episode_count, self.step_count))
    

    # Limit the state and reward history
    if len(self.rewards_history) > max_memory_length:
        del self.rewards_history[:1]
        del self.state_history[:1]
        del self.state_history_last[:1]
        del self.action_history[:1]
        #del done_history[:1]
    
    
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Idea: Add your own events to hand out rewards
    if ...:
        events.append(PLACEHOLDER_EVENT)

    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))
    

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
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))


    self.model.save('my_model.h5')   
    
    ###########################

    self.episode_reward_history.append(self.episode_reward)
    if len(self.episode_reward_history) > 100:
        del self.episode_reward_history[:1]
    self.running_reward = np.mean(self.episode_reward_history)  

    
    #save data about episode and rewards
    episode_data = []
    episode_data.append(self.episode_count)
    episode_data.append(self.episode_step_count)
    episode_data.append(self.step_count)
    episode_data.append(self.epsilon)
    episode_data.append(events)
    episode_data.append(self.episode_reward)
    episode_data.append(self.running_reward)
    
    self.rewards_data.append(episode_data)
    df = pd.DataFrame(self.rewards_data)
    df.to_csv("rewards_data.csv")
    
    
    self.episode_count += 1
    self.episode_step_count = 0
    self.episode_reward = 0
    ###########################



def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """

    
    game_rewards = {
        e.COIN_COLLECTED: 10,
        e.KILLED_OPPONENT: 5,
        #e.SURVIVED_ROUND: 20,
        
        e.INVALID_ACTION: -.3,        
        #e.GOT_KILLED: -0.3,
        #PLACEHOLDER_EVENT: -.1  # idea: the custom event is bad
        e.KILLED_SELF: -10,
        
        
        
        e.MOVED_LEFT: 1,
        e.MOVED_DOWN: 1,
        e.MOVED_RIGHT: 1,
        e.MOVED_UP: 1,
        e.WAITED: -.3,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    

    
    return reward_sum
