#import os
#import sys
#import gym
import random
import numpy as np
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from model import QNet
from memory import Memory
#from tensorboardX import SummaryWriter
from display import Displayer

from config import initial_exploration, batch_size, update_target, goal_score, log_interval, device, replay_memory_capacity, lr


def get_action(state, target_net, epsilon, env,test=False):
    if np.random.rand() <= epsilon and test==False:
        return torch.tensor([[random.randrange(4)]], device=device, dtype=torch.long)
    else:
        return target_net.get_action(state)

def update_target_model(online_net, target_net):
    # Target <- Net
    target_net.load_state_dict(online_net.state_dict())


def main(L, mouse_initial_indices, rewardlist, actions_list):
    
    
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
    #writer = SummaryWriter('logs')

    online_net.to(device)
    target_net.to(device)
    online_net.train()
    target_net.train()
    memory = Memory(replay_memory_capacity)
    running_score = 0
    epsilon = 1.0
    steps = 0
    loss = 0

    for e in range(30000):
        steps = 0
        done = False

        score = 0
        state = np.array(mouse_initial_indices)
        state = torch.Tensor(state).to(device)
        state = state.unsqueeze(0)

        while not done:
            steps += 1
            if steps > 100:
                done = True

            action = get_action(state, target_net, epsilon, env)
            
            new_state=state+torch.Tensor(np.array(actions_list[action])).to(device)
            
            if env[int(new_state[0][0].tolist()),int(new_state[0][1].tolist())]!=0:  # if the mouse doesn't step on a wall
                next_state=new_state
                reward=rewardlist[env[int(new_state[0][0].tolist()),int(new_state[0][1].tolist())]]
                if env[int(new_state[0][0].tolist()),int(new_state[0][1].tolist())]==2:  # if the mouse reaches the end
                    done=True
            else : # if the mouse steps on a wall
                next_state=state
                reward=rewardlist[0]

            mask = 0 if done else 1
            action_one_hot = np.zeros(4)
            action_one_hot[action] = 1
            memory.push(state, next_state, action_one_hot, reward, mask)

            score += reward
            state = next_state

        if e > initial_exploration:
            epsilon -= 0.00005
            epsilon = max(epsilon, 0.1)

            batch = memory.sample(batch_size)
            loss = QNet.train_model(online_net, target_net, optimizer, batch)

            if e % update_target == 0:
                update_target_model(online_net, target_net)

        running_score = 0.99 * running_score + 0.01 * score
        # running_score=score
        if e % log_interval == 0:
            print('{} episode | score: {:.2f} | epsilon: {:.2f}'.format(
                e, running_score, epsilon))
            #writer.add_scalar('log/score', float(running_score), e)
            #writer.add_scalar('log/loss', float(loss), e)

        if running_score > goal_score:
            break
    torch.save(online_net.state_dict(), "./qlearning_model")
    


def test(L, mouse_initial_indices, rewardlist, actions_list):
    
    online_net = QNet(2, 4).to(device)
    online_net.load_state_dict(torch.load("./qlearning_model", map_location=device))
    
    env = L
    
    done = False
    steps = 0
    score = 0
    state = np.array(mouse_initial_indices)
    state = torch.Tensor(state).to(device)
    state = state.unsqueeze(0)
    
    
    def progress_loop(done, steps, state, score):
        
        steps += 1

        action = get_action(state, online_net, 1, env, test=True)
        
        displacement = np.array(actions_list[action])
        
        new_state = state+torch.Tensor(displacement).to(device)
        
        if env[int(new_state[0][0].tolist()),int(new_state[0][1].tolist())] != 0:
            
            next_state=new_state
            
            displayer.main_canva.move(
                displayer.mouse,
                *(displacement  * displayer.square_size)
            )
            
            reward=rewardlist[env[int(new_state[0][0].tolist()),int(new_state[0][1].tolist())]]
            if env[int(new_state[0][0].tolist()),int(new_state[0][1].tolist())] == 2:
                done=True
        else :
            next_state=state
            reward=rewardlist[0]


        score += reward
        
        state = next_state
        print('position : ', state.tolist()[0], score)
        
        if done is False:
            displayer.window.after(100, lambda: progress_loop(done, steps, state, score))
    
        
    displayer = Displayer()
    
    displayer.create_labyrinth(L, mouse_initial_indices)
    progress_loop(done, steps, state, score)
    
    displayer.window.mainloop()




if __name__=="__main__":
    
    #L=np.array([[0,0,0,0,0],[0,1,1,3,0],[0,1,3,1,0],[0,1,1,2,0],[0,0,0,0,0]]) #labyrinthe utilisé (0=mur, 1=vide, 2= arrivée, 3=électricité, 4=eau)
    
#    L = np.array([ [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                   [0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0],
#                   [0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0],
#                   [0, 3, 1, 0, 0, 0, 1, 1, 1, 1, 0],
#                   [0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0],
#                   [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
#                   [0, 1, 1, 0, 1, 3, 1, 1, 0, 1, 0],
#                   [0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0],
#                   [0, 1, 1, 1, 1, 1, 1, 1, 0, 2, 0],
#                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#                 ])
    
    L=np.array([
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0, 0, 0, 0],
                [2, 1, 3, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 0, 0, 0],
                [0, 1, 0, 0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0]
              ]).T
    
    mouse_initial_indices = [5,5]
    rewardlist = [-10,-1,100,-50,5,-10] #se prendre un mur, se déplacer, arriver au fromage, se prendre l'électricité, boire de l'eau, revenir sur de l'eau
    actions_list = [[1,0],[-1,0],[0,1],[0,-1]]
    
    
    #main(L, mouse_initial_indices, rewardlist, actions_list)

    test(L, mouse_initial_indices, rewardlist, actions_list)
    
    
