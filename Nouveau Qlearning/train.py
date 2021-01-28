import os
import sys
import gym
import random
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
from model import QNet
from memory import Memory
from tensorboardX import SummaryWriter

from config import initial_exploration, batch_size, update_target, goal_score, log_interval, device, replay_memory_capacity, lr

def get_action(state, target_net, epsilon, env,test=False):
    if np.random.rand() <= epsilon and test==False:
        return torch.tensor([[random.randrange(4)]], device=device, dtype=torch.long)
    else:
        return target_net.get_action(state)

def update_target_model(online_net, target_net):
    # Target <- Net
    target_net.load_state_dict(online_net.state_dict())


def main():
    L=np.array([[0,0,0,0,0],[0,1,1,3,0],[0,1,3,1,0],[0,1,1,2,0],[0,0,0,0,0]]) #labyrinthe utilisé (0=mur, 1=vide, 2= arrivée, 3=électricité, 4=eau)
    dep=[1,1]
    env = L
    torch.manual_seed(500)

    num_inputs = 2
    num_actions = 4
    print('state size:', num_inputs)
    print('action size:', num_actions)

    online_net = QNet(num_inputs, num_actions)
    target_net = QNet(num_inputs, num_actions)
    update_target_model(online_net, target_net)

    optimizer = optim.Adam(online_net.parameters(), lr=lr)
    writer = SummaryWriter('logs')

    online_net.to(device)
    target_net.to(device)
    online_net.train()
    target_net.train()
    memory = Memory(replay_memory_capacity)
    running_score = 0
    epsilon = 1.0
    steps = 0
    loss = 0
    rewardlist=[-5,-1,50,-10,5,-10] #se prendre un mur, se déplacer, arriver au fromage, se prendre l'électricité, boire de l'eau, revenir sur de l'eau
    def_action=[[1,0],[-1,0],[0,1],[0,-1]]

    for e in range(1000):
        done = False

        score = 0
        state = np.array(dep)
        state = torch.Tensor(state).to(device)
        state = state.unsqueeze(0)

        while not done:
            steps += 1

            action = get_action(state, target_net, epsilon, env)
            newstate=state+torch.Tensor(np.array(def_action[action])).to(device)
            if env[int(newstate[0][0].tolist()),int(newstate[0][1].tolist())]!=0:
                next_state=newstate
                reward=rewardlist[env[int(newstate[0][0].tolist()),int(newstate[0][1].tolist())]]
                if env[int(newstate[0][0].tolist()),int(newstate[0][1].tolist())]==2:
                    done=True
            else :
                next_state=state
                reward=rewardlist[0]

            mask = 0 if done else 1
            action_one_hot = np.zeros(4)
            action_one_hot[action] = 1
            memory.push(state, next_state, action_one_hot, reward, mask)

            score += reward
            state = next_state

            if steps > initial_exploration:
                epsilon -= 0.00005
                epsilon = max(epsilon, 0.1)

                batch = memory.sample(batch_size)
                loss = QNet.train_model(online_net, target_net, optimizer, batch)

                if steps % update_target == 0:
                    update_target_model(online_net, target_net)

        running_score = 0.99 * running_score + 0.01 * score
        # running_score=score
        if e % log_interval == 0:
            print('{} episode | score: {:.2f} | epsilon: {:.2f}'.format(
                e, running_score, epsilon))
            writer.add_scalar('log/score', float(running_score), e)
            writer.add_scalar('log/loss', float(loss), e)

        if running_score > goal_score:
            break
    torch.save(online_net.state_dict(), "./qlearning_model")

def test():
    online_net = QNet(2, 4).to(device)
    online_net.load_state_dict(torch.load("./qlearning_model", map_location=device))
    L=np.array([[0,0,0,0,0],[0,1,1,3,0],[0,1,3,1,0],[0,1,1,2,0],[0,0,0,0,0]]) #labyrinthe utilisé (0=mur, 1=vide, 2= arrivée, 3=électricité, 4=eau)
    dep=[1,1]
    env = L
    rewardlist=[-5,-1,50,-10,5,-10] #se prendre un mur, se déplacer, arriver au fromage, se prendre l'électricité, boire de l'eau, revenir sur de l'eau
    def_action=[[1,0],[-1,0],[0,1],[0,-1]]
    done = False
    steps = 0
    score = 0
    state = np.array(dep)
    state = torch.Tensor(state).to(device)
    state = state.unsqueeze(0)
    while not done:
        steps += 1

        action = get_action(state, online_net, 1, env,test=True)
        newstate=state+torch.Tensor(np.array(def_action[action])).to(device)
        if env[int(newstate[0][0].tolist()),int(newstate[0][1].tolist())]!=0:
            next_state=newstate
            reward=rewardlist[env[int(newstate[0][0].tolist()),int(newstate[0][1].tolist())]]
            if env[int(newstate[0][0].tolist()),int(newstate[0][1].tolist())]==2:
                done=True
        else :
            next_state=state
            reward=rewardlist[0]


        score += reward
        state = next_state
        print('position : ',state.tolist()[0],score)




if __name__=="__main__":
    m=main()
    test()