import os
import pickle
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.autograd import Variable
import math
import itertools

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
    print("setup")
    
    self.modelToUse = 1
    if self.train:
        print("train")
    else:
        print("no train")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.envSize = 17
    
    #init model
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        if self.modelToUse == 0:
            self.policy_net = Model_global_view(self.envSize, self.envSize, 6).to(device)
            self.model = Model_global_view(self.envSize, self.envSize, 6).to(device)
        elif self.modelToUse == 1:
            self.policy_net = Model_local_view(self.envSize, self.envSize, 6).to(device)
            self.model = Model_local_view(self.envSize, self.envSize, 6).to(device)
        else:
            self.policy_net = Model_combined_view(self.envSize, self.envSize, 6).to(device)
            self.model = Model_combined_view(self.envSize, self.envSize, 6).to(device)
        self.model.load_state_dict(self.policy_net.state_dict())
        self.model.eval()
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            if self.modelToUse == 0:
                self.model = Model_global_view(self.envSize, self.envSize, 6)
            elif self.modelToUse == 1:
                self.model = Model_local_view(self.envSize, self.envSize, 6)
            else:
                self.model = Model_combined_view(self.envSize, self.envSize, 6)
            if torch.cuda.is_available():
                self.model.load_state_dict(torch.load(file))
                self.model.to(device)
            else:
                self.model.load_state_dict(torch.load(file, map_location=device))
            



def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    #print("act")
    self.logger.debug("Querying model for action.")
    if self.train and self.steps_done < self.imitationSteps:
        self.steps_done += 1
        act = self.imitationAgent.act(game_state)
        return act
    else:
        act = ACTIONS[select_action(self, state_to_features(self, game_state))]
        return act


def state_to_features(self, game_state: dict) -> np.array:
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    
    #get global information as a 17x17 channel
    x = game_state['field']
    x = np.swapaxes(x,0,1)
    for i in range(len(game_state['coins'])):
        a = game_state['coins'][i][1]
        b = game_state['coins'][i][0]
        x[a][b] = 4
    for i in range(len(game_state['bombs'])):
        a = game_state['bombs'][i][0][1]
        b = game_state['bombs'][i][0][0]
        x[a][b] = -(5+game_state['bombs'][i][1])
    for i in game_state['others']:
        if i[2]:
            x[i[3][1]][i[3][0]] = -10
        else:
            x[i[3][1]][i[3][0]] = -11
    if game_state['self'][2]:
        x[game_state['self'][3][1]][game_state['self'][3][0]] = 5
    else:
        x[game_state['self'][3][1]][game_state['self'][3][0]] = 6
    expl_List = np.argwhere(game_state['explosion_map'] != 0)
    for i in expl_List:
        x[i[1]][i[0]] = -4
    channel1 = x.copy()
    
    
    #prep local channel
    if self.modelToUse != 0:
        #get simpele direction to and aways from closest coin or crate if no coin on the field
        x_axis,y_axis,coin_creat_encoding = directionToNearestCoin_Crate(game_state['coins'], game_state['self'][3], game_state['field'])
        if x_axis == "left":
            if x[game_state['self'][3][1]][game_state['self'][3][0]-1] == 0:
                x[game_state['self'][3][1]][game_state['self'][3][0]-1] = coin_creat_encoding
            if x[game_state['self'][3][1]][game_state['self'][3][0]+1] == 0:
                x[game_state['self'][3][1]][game_state['self'][3][0]+1] = -2
        if x_axis == "right":
            if x[game_state['self'][3][1]][game_state['self'][3][0]-1] == 0:
                x[game_state['self'][3][1]][game_state['self'][3][0]-1] = -2
            if x[game_state['self'][3][1]][game_state['self'][3][0]+1] == 0:
                x[game_state['self'][3][1]][game_state['self'][3][0]+1] = coin_creat_encoding
        if y_axis == "up":
            if x[game_state['self'][3][1]-1][game_state['self'][3][0]] == 0:
                x[game_state['self'][3][1]-1][game_state['self'][3][0]] = coin_creat_encoding
            if x[game_state['self'][3][1]+1][game_state['self'][3][0]] == 0:
                x[game_state['self'][3][1]+1][game_state['self'][3][0]] = -2
        if y_axis == "down":
            if x[game_state['self'][3][1]-1][game_state['self'][3][0]] == 0:
                x[game_state['self'][3][1]-1][game_state['self'][3][0]] = -2
            if x[game_state['self'][3][1]+1][game_state['self'][3][0]] == 0:
                x[game_state['self'][3][1]+1][game_state['self'][3][0]] = coin_creat_encoding
                    
        
        
        #get information of bombs: on which position the explotion will be and how far away the bomb is
        bombs = game_state['bombs']
        bombs.sort(key=lambda x: x[1],reverse=True)
        x = np.pad(x, (3,3), 'constant', constant_values=(-1))
        for i in (bombs):
            y_bomb = i[0][1] + 3
            x_bomb = i[0][0] + 3
            for j in range(4):
                if abs(x[y_bomb,x_bomb+j]) != 1 and x[y_bomb,x_bomb+j] != -4:
                    blocked = False
                    for l in range(j):
                        if x[y_bomb,x_bomb+j-l] == -1:
                            blocked = True
                    if blocked == False:
                        x[y_bomb,x_bomb+j] = -(9-j)
                        #print("test1")
                if abs(x[y_bomb,x_bomb-j]) != 1 and x[y_bomb,x_bomb-j] != -4:
                    blocked = False
                    for l in range(j):
                        if x[y_bomb,x_bomb-j+l] == -1:
                            blocked = True
                    if blocked == False:
                        x[y_bomb,x_bomb-j] = -(9-j)
                        #print("test2")
                if abs(x[y_bomb+j,x_bomb]) != 1 and x[y_bomb+j,x_bomb] != -4:
                    blocked = False
                    for l in range(j):
                        if x[y_bomb+j-l,x_bomb] == -1:
                            blocked = True
                    if blocked == False:
                        x[y_bomb+j,x_bomb] = -(9-j)
                        #print("test3")
                if abs(x[y_bomb-j,x_bomb]) != 1 and x[y_bomb-j,x_bomb] != -4:
                    blocked = False
                    for l in range(j):
                        if x[y_bomb-j+l,x_bomb] == -1:
                            blocked = True
                    if blocked == False:
                        x[y_bomb-j,x_bomb] = -(9-j)
                        #print("test4")
        x = x[3:-3,3:-3]
    
        
        #get local view and concatenate it with channel 1 (will be sliced apart in the model later)
        z = np.zeros(17)
        y = x[game_state['self'][3][1]-1:game_state['self'][3][1]+2,game_state['self'][3][0]-1:game_state['self'][3][0]+2]
        y = y.flatten()
        z[0:9] = y
    #get correct input for the model used
    if self.modelToUse == 2:
        z = Variable(torch.from_numpy(z)).to(device).to(torch.float)
        z = z.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        channel1 = Variable(torch.from_numpy(channel1)).to(device).to(torch.float)
        channel1 = channel1.unsqueeze(0).unsqueeze(0)
        return torch.cat((channel1,z),2)
    elif self.modelToUse == 1:
        y = Variable(torch.from_numpy(y)).to(device).to(torch.float)
        y = y.unsqueeze(0)
        return y
    else:
        channel1 = Variable(torch.from_numpy(channel1)).to(device).to(torch.float)
        channel1 = channel1.unsqueeze(0).unsqueeze(0)
        return channel1
    return


def oldestBombInRange(pos, bombs):
    timer = 10
    for i in bombs:
        for j in range(3):
            if i[0][1] == pos[1]:
                if i[0][0] >= pos[0]-j-1 and i[0][0] <= pos[0]+j+1:
                    if timer > i[1]:
                        timer = i[1]
            if i[0][0] == pos[0]:
                if i[0][1] >= pos[1]-j-1 and i[0][1] <= pos[1]+j+1:
                    if timer > i[1]:
                        timer = i[1]
        return - (timer+5)
        

def select_action(self, state):
    with torch.no_grad():
        if self.train:
            sample = random.random()
            eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                math.exp(-1. * self.steps_done / self.EPS_DECAY)
            self.steps_done += 1
            if sample > eps_threshold or self.doEval:
                x = self.policy_net(state)
                z = torch.nn.functional.softmax(x,dim = 1).data
                y = np.random.choice([0,1,2,3,4,5], p=z.view(6).cpu().numpy())
                return y
            else:
                # 80%: walk in any direction. 10% wait. 10% bomb.
                return np.random.choice([0,1,2,3,4,5], p=[.2, .2, .2, .2, .1, .1])           
        else:
            x = self.model(state)
            y = torch.argmax(x)
            return y
        
def directionToNearestCoin_Crate(coins, pos, field):
    x_axis = 0
    y_axis = 0
    coin_creat_encoding = 0
    if not not coins:
        coin_creat_encoding = 2
        coins = np.asarray(coins)
        dist_1 = np.sum((coins - pos)**2, axis=1)
        argDist = np.argmin(dist_1)
        if coins[argDist][0] < pos[0]:
            x_axis = "left"
        if coins[argDist][0] > pos[0]:
            x_axis = "right"
        if coins[argDist][1] < pos[1]:
            y_axis = "up"
        if coins[argDist][1] > pos[1]:
            y_axis = "down"
    else:
        cratesList = np.argwhere(field == 1)
        if not cratesList.size == 0:
            coin_creat_encoding = 3
            cratesList = np.asarray(cratesList)
            dist_1 = np.sum((cratesList - pos)**2, axis=1)
            argDist = np.argmin(dist_1)
            if cratesList[argDist][0] < pos[0]:
                x_axis = "left"
            if cratesList[argDist][0] > pos[0]:
                x_axis = "right"
            if cratesList[argDist][1] < pos[1]:
                y_axis = "up"
            if cratesList[argDist][1] > pos[1]:
                y_axis = "down"
    return x_axis,y_axis, coin_creat_encoding


class Model_combined_view(nn.Module):

    def __init__(self, h, w, outputs):
        super(Model_combined_view, self).__init__()
        self.padding = nn.ConstantPad2d(1, -1)
        self.conv1 = nn.Conv2d(1, 66, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(66)
        self.conv2 = nn.Conv2d(66, 66, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(66)
        self.conv3 = nn.Conv2d(66, 66, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(66)

        def conv2d_size_out(size, kernel_size = 3, stride = 1, padding = 1):
            return (size + 2*padding - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 66 + 9
        self.head = nn.Linear(linear_input_size, 200)
        self.fc1 = nn.Linear(200, outputs)

    def forward(self, x):
        #print(x.size())
        y = x[:,:,-1,:]
        y = y.view(y.size(0), -1)
        y = y[:,:9]
        x = x[:,:,:-1,:]
        #print(x.size())
        #print(y.size())
        x = F.relu(self.bn1(self.conv1(self.padding(x))))
        x = F.relu(self.bn2(self.conv2(self.padding(x))))
        x = F.relu(self.bn3(self.conv3(self.padding(x))))
        #print(x.size())
        x = F.relu(self.head(torch.cat((x.view(x.size(0), -1),y),1)))
        x = self.fc1(x)
        return x  
    
    
class Model_local_view(nn.Module):

    def __init__(self, h, w, outputs):
        super(Model_local_view, self).__init__()
        self.fc1 = nn.Linear(9, 200)
        self.fc2 = nn.Linear(200, outputs)

    def forward(self, x):
        x = F.relu(self.fc1(x))#.view(x.size(0), -1)))
        x = self.fc2(x)
        return x  
    
    
class Model_global_view(nn.Module):

    def __init__(self, h, w, outputs):
        super(Model_global_view, self).__init__()
        self.padding = nn.ConstantPad2d(1, -1)
        self.conv1 = nn.Conv2d(1, 66, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(66)
        self.conv2 = nn.Conv2d(66, 66, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(66)
        self.conv3 = nn.Conv2d(66, 66, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(66)

        def conv2d_size_out(size, kernel_size = 3, stride = 1, padding = 1):
            return (size + 2*padding - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 66
        self.head = nn.Linear(linear_input_size, 200)
        self.fc1 = nn.Linear(200, outputs)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(self.padding(x))))
        x = F.relu(self.bn2(self.conv2(self.padding(x))))
        x = F.relu(self.bn3(self.conv3(self.padding(x))))
        x = F.relu(self.head(x.view(x.size(0), -1)))
        x = self.fc1(x)
        return x  
