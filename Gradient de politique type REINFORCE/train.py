import os
import sys
import random
import numpy as np

from copy import deepcopy

import torch
import torch.optim as optim
import torch.nn.functional as F
from model import QNet
from tensorboardX import SummaryWriter

from memory import Memory
from config import goal_score, log_interval, device, lr, gamma


def main():
    env = deepcopy(L)
    torch.manual_seed(500)

    num_inputs = 2
    num_actions = 4
    print('state size:', num_inputs)
    print('action size:', num_actions)

    net = QNet(num_inputs, num_actions)

    optimizer = optim.Adam(net.parameters(), lr=lr)
    writer = SummaryWriter('logs')

    net.to(device)
    net.train()
    running_score = 0
    steps = 0
    loss = 0

    for e in range(500):
        done = False
        memory = Memory()

        score = 0
        state = np.array(mouse_initial_indices)
        state = torch.Tensor(state).to(device)
        state = state.unsqueeze(0)
        #print("step,state,score")
        #print(steps,",",state[0].data.numpy(),",",score)
        while not done:
            steps += 1
            
            #print(state.size())
            #print(state[0].data.numpy())

            action = net.get_action(state)
            
            #print("action = ",action)
            
            newstate = state + torch.Tensor(np.array(actions_list[action])).to(device)
            
            if env[int(newstate[0][0].tolist()),int(newstate[0][1].tolist())] != 0:
                next_state = newstate
                reward = rewardlist[env[int(newstate[0][0].tolist()),int(newstate[0][1].tolist())]]
                if env[int(newstate[0][0].tolist()),int(newstate[0][1].tolist())] == 2:
                    done = True
            else :
                next_state = state
                reward = rewardlist[0]

            

            mask = 0 if done else 1
            reward = reward if not done or score == 499 else -1

            action_one_hot = np.zeros(4)
            action_one_hot[action] = 1
            memory.push(state, next_state, action_one_hot, reward, mask)
    
            score += reward
            state = next_state
            
            #print(steps,",",state[0].data.numpy(),",",score)
            
            
        #print(net)
        #print(memory.sample())
        loss = QNet.train_model(net, memory.sample(), optimizer)
        #print(loss)    

        score = score if score == 500.0 else score + 1
        running_score = 0.99 * running_score + 0.01 * score
        if e % log_interval == 0:
            print('{} episode | score: {:.2f}'.format(
                e, running_score))
            writer.add_scalar('log/score', float(running_score), e)
            writer.add_scalar('log/loss', float(loss), e)

        if running_score > goal_score:
            break


if __name__=="__main__":
    
    L = np.array([[0,0,0,0,0],[0,1,1,3,0],[0,1,3,1,0],[0,1,1,2,0],[0,1,1,4,0],[0,0,0,0,0]]).T #labyrinth utilisé (0=mur, 1=vide, 2= arrivée, 3=électricité, 4=eau)
    mouse_initial_indices = [1,1]
    rewardlist = [-5,-1,50,-10,20,-10] #se prendre un mur, se déplacer, arriver au fromage, se prendre l'électricité, boire de l'eau, revenir sur de l'eau
    actions_list = [[1,0],[-1,0],[0,1],[0,-1]]
    
    main()
