import pickle
import random
from collections import namedtuple, deque
from typing import List

import events as e
from .callbacks import state_to_features


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np

import sys 
import os
sys.path.append(os.path.abspath("./agent_code/my_agent"))
print(os.path.abspath("./agent_code/my_agent"))
from imitationAgent import imitAgent


Transition = namedtuple('Transition',
                        ('state', 'action', 'next__to', 'reward'))

TRANSITION_HISTORY_SIZE = 20000  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    print("setup train")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    if not torch.cuda.is_available():
        print("cuda not available")
    
    
    # hyperparameter
    self.imitationSteps = 0
    self.useCheckpoint = True
    self.eval = 10
    self.BATCH_SIZE = 1024
    self.GAMMA = 0.95
    self.EPS_START = 0.9
    self.EPS_END = 0.05
    self.EPS_DECAY = 6000000
    self.TARGET_UPDATE = 100
    self.saveModel = 20    
    self.optimizer = optim.Adam(self.policy_net.parameters(),lr=0.001)
    
    
    
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)      
    self.i_episode = 0
    self.steps_done = 0
    self.episode_durations = []  
    # create rule based agent if imitation learining is used
    if self.imitationSteps > 0:
        self.imitationAgent = imitAgent()
    # parameter for eval
    self.doEval = False
    self.evalReward = 0
    self.rewardList = []
    self.augRewardList = []
    self.accList = [] 
    self.accuracy = None
    # load checkpoint
    if self.useCheckpoint:
        load_ckp("my-checkpoint.pt", self)
        self.model.load_state_dict(self.policy_net.state_dict())
        self.model.eval()
    
      

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

    # state_to_features is defined in callbacks.py
    if old_game_state != None:
        if gotCloserToClosestCoin(old_game_state["coins"], old_game_state["self"][3], new_game_state["self"][3]) > 0:
            events.append("coinDistance-")
        if gotCloserToClosestCoin(old_game_state["coins"], old_game_state["self"][3], new_game_state["self"][3]) < 0:
            events.append("coinDistance+")
        
        if self.doEval == False:
            self_action = toInt(self_action)
            self.transitions.append(Transition(state_to_features(self, old_game_state), self_action, state_to_features(self, new_game_state), reward_from_events(self, events)))
        else:
            self.evalReward += reward_from_events(self, events)
    if self.doEval == False:
        optimize_model(self)
    

    
    
    

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
    if self.doEval == False:
        last_action = toInt(last_action)
        self.transitions.append(Transition(state_to_features(self, last_game_state), last_action, None, reward_from_events(self, events)))
    else:
        self.evalReward += reward_from_events(self, events)
    self.i_episode += 1
    
    # ppdate target model
    if self.i_episode % self.TARGET_UPDATE == 0:
        self.model.load_state_dict(self.policy_net.state_dict())
        
    # store the model and checkpoint
    if self.i_episode % self.saveModel == 0: 
        with open("my-saved-model.pt", "wb") as file:
            torch.save(self.policy_net.state_dict(), file)
        with open("my-checkpoint.pt", "wb") as file:
            checkpoint = {
            'epoch': self.i_episode,
            'state_dict': self.policy_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps': self.steps_done,
            'rewardList': self.rewardList,
            'augRewardList': self.augRewardList,
            'accList': self.accList
            }
            torch.save(checkpoint, file)


    #run one episode without training and collect eval data
    if self.i_episode % self.eval == 1:
        #print(self.i_episode)
        self.policy_net.eval()
        self.doEval = True
        self.evalReward = 0
    #return to training and save eval data
    if self.i_episode % self.eval == 2:
        #print(self.i_episode)
        self.policy_net.train()
        self.doEval = False
        
        self.rewardList.append(last_game_state["self"][1])
        self.augRewardList.append(self.evalReward)
        if self.steps_done <= self.imitationSteps:
            if self.accuracy != None:
                self.accList.append(self.accuracy)
            


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        e.KILLED_SELF: -5,
        e.INVALID_ACTION: -1,
        "coinDistance-": 0.2,
        "coinDistance+": -0.2,
        
        e.CRATE_DESTROYED: 0.2,
        #e.WAITED: -0.01
        #e.BOMB_DROPPED: -5
    }
    reward_sum = 0
    
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")

    return reward_sum


def gotCloserToClosestCoin(coins, oldPos, newPos):
    dist_1 = 0
    dist_2 = 0
    if not not coins:
        coins = np.asarray(coins)
        dist_1 = np.sum((coins - oldPos)**2, axis=1)
        dist_1 = np.min(dist_1)
        dist_2 = np.sum((coins - newPos)**2, axis=1)
        dist_2 = np.min(dist_2)
    return dist_1 - dist_2

def optimize_model(self):
    if len(self.transitions) < self.BATCH_SIZE:
        return
    batch = random.sample(list(self.transitions), self.BATCH_SIZE)
    batch = Transition(*zip(*batch))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    
    
    # compute a mask of non-final states and prepare the batch elements    
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next__to)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next__to
                                                if s is not None])
    state_batch = torch.cat(batch.state).to(device)
    try:
        action_batch = torch.from_numpy(np.asarray(batch.action).astype("int64")).to(device)
    except ValueError:
        print(batch.action)
    reward_batch = torch.from_numpy(np.asarray(batch.reward)).to(device)
    reward_batch = reward_batch.float()

    # compute Q(s_t, a)
    state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
    

    # compute V(s_{t+1}) for all next states..
    next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = self.model(non_final_next_states).max(1)[0].detach()

    # compute the expected Q values
    expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

    # compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    
    # compute imitation loss and combine losses
    if self.steps_done % 4000 == 0:
        print("Episode: ", self.i_episode)
        print("steps done: ", self.steps_done)
        print("Huber loss: ", loss)
    if self.steps_done <= self.imitationSteps:
        if self.steps_done == self.imitationSteps:
            print("end imitation learning")
        if self.steps_done == self.BATCH_SIZE:
            print("start imitation learning")
        values = self.model(state_batch)
        a = values.gather(1, action_batch.unsqueeze(1))
        output = values.clone()
        output += 0.8
        output[torch.arange(self.BATCH_SIZE),action_batch] -= 0.8
        output = values.max(1)[0].detach().unsqueeze(1)
        imitationLoss = output - a
        self.accuracy = 1 - (torch.count_nonzero(imitationLoss) / self.BATCH_SIZE)
        imitationLoss = torch.sum(imitationLoss)
        if self.steps_done % 4000 == 0:
            print("imitation loss: ", imitationLoss)
        loss += imitationLoss
    
    # Optimize the model
    self.optimizer.zero_grad()
    loss.backward()
    for param in self.policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    self.optimizer.step()
    
    
def toInt(action):
    switcher={
                "UP":0,
                "RIGHT":1,
                "DOWN":2,
                "LEFT":3,
                "WAIT":4,
                "BOMB":5,
                "Invalid": 4,
             }
    return switcher.get(action,"Invalid")
    

def load_ckp(checkpoint_path, self):
    """
    checkpoint_path: path to save checkpoint
    model: model to load checkpoint parameters into       
    optimizer: optimizer to use
    scheduler: scheduler to use
    """
    # load check point
    checkpoint = torch.load(checkpoint_path)
    # initialize
    self.policy_net.load_state_dict(checkpoint['state_dict'])
    self.optimizer.load_state_dict(checkpoint['optimizer'])
    self.steps_done = checkpoint['steps']
    self.i_episode = checkpoint['epoch']
    self.rewardList = checkpoint['rewardList']
    self.augRewardList = checkpoint['augRewardList']
    self.accList = checkpoint['accList']
    print(self.i_episode)
    return 0
            
            