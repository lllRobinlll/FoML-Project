# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 03:05:20 2021

@author: User
"""
import torch
import matplotlib
import matplotlib.pyplot as plt
import pickle
from os import makedirs
import numpy as np

def load_ckp(checkpoint_path):
    """
    checkpoint_path: path to save checkpoint
    model: model to load checkpoint parameters into       
    optimizer: optimizer to use
    scheduler: scheduler to use
    """
    # load check point
    checkpoint = torch.load(checkpoint_path)
    # initialize
    rewardList = checkpoint['rewardList']
    augRewardList = checkpoint['augRewardList']
    accList = checkpoint["accList"]
    return rewardList, augRewardList, accList


def plot_history(data, name, y_label = "Reward"):
    try:
        makedirs('./plots')
    except Exception as e:
        None
    legendList = []
    plt.rc('xtick',labelsize=15)
    plt.rc('ytick',labelsize=15)
    fig = plt.figure()
    plt.xlabel('Episodes', fontsize=30)
    plt.ylabel(y_label, fontsize=30)

    plt.plot(data)
    #legendList.append('real')
    #plt.legend(legendList, loc='upper left', fontsize=15)
    #plt.plot(augmented)
    #legendList.append('augmented')
    #plt.legend(legendList, loc='upper left', fontsize=15)
    fig.set_figheight(7.2*2)
    fig.set_figwidth(15)
    fig.savefig("./plots/" + name + ".png")
    plt.close(fig)
    


reward, augReward, accTensors = load_ckp("my-checkpoint.pt")
acc = []
for i in accTensors:
    acc.append(i.cpu().numpy())
plot_history(reward, "reward", "reward")
plot_history(augReward, "augmented_reward", "augmented_reward")
plot_history(acc, "accuracy_on_imitation", "accuracy_on_imitation")



reward = np.array(reward)
augReward = np.array(augReward)
acc = np.array(acc)
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

reward = moving_average(reward,50)
plot_history(reward, "reward_smoothen", "reward")

augReward = moving_average(augReward,50)
plot_history(augReward, "augmented_reward_smothen", "reward")
