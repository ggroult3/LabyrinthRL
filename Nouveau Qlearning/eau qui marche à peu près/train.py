import os
import sys
import gym
import random
import numpy as np
from copy import deepcopy
import torch
import torch.optim as optim
import torch.nn.functional as F
from model import QNet
from memory import Memory
# from tensorboardX import SummaryWriter
from display import Displayer


from config import initial_exploration, batch_size, update_target, goal_score, log_interval, device, replay_memory_capacity, lr

def get_action(state, target_net, epsilon, env,test=False,seprendrelesmurs=False,eaubue=0.):
    stateaugmente=torch.cat((state,torch.tensor(eaubue).unsqueeze(0).unsqueeze(0).to(device)),1)
    if not seprendrelesmurs:
        actions_list=[[1,0],[-1,0],[0,1],[0,-1]]
        if np.random.rand() <= epsilon and test==False: #au hasard
            action=torch.tensor([[random.randrange(4)]], device=device, dtype=torch.long)
            newstate=state+torch.Tensor(np.array(actions_list[action])).to(device)
            if env[int(newstate[0][0].tolist()),int(newstate[0][1].tolist())]!=0:
                return action
            while env[int(newstate[0][0].tolist()),int(newstate[0][1].tolist())]==0:
                action=torch.tensor([[random.randrange(4)]], device=device, dtype=torch.long)
                newstate=state+torch.Tensor(np.array(actions_list[action])).to(device)
            return action #Renvoie ici un tenseur valant de 0 à 3, qui n'entraîne pas un déplacement vers un mur
        else:
            qvalue=target_net.get_action(stateaugmente)
            listeindice=qvalue.topk(4)[1][0]
            action=listeindice[0].cpu().numpy()
            newstate=state+torch.Tensor(np.array(actions_list[action])).to(device)
            if env[int(newstate[0][0].tolist()),int(newstate[0][1].tolist())]!=0:
                return action
            action=listeindice[1].cpu().numpy()
            newstate=state+torch.Tensor(np.array(actions_list[action])).to(device)
            if env[int(newstate[0][0].tolist()),int(newstate[0][1].tolist())]!=0:
                return action
            action=listeindice[2].cpu().numpy()
            newstate=state+torch.Tensor(np.array(actions_list[action])).to(device)
            if env[int(newstate[0][0].tolist()),int(newstate[0][1].tolist())]!=0:
                return action
            action=listeindice[3].cpu().numpy()
            return action

    if np.random.rand() <= epsilon and test==False:
        return torch.tensor([[random.randrange(4)]], device=device, dtype=torch.long)
    else:
        _, action = torch.max(target_net.get_action(stateaugmente), 1)
        return action.cpu().numpy()[0]

def update_target_model(online_net, target_net):
    # Target <- Net
    target_net.load_state_dict(online_net.state_dict())


def main(L, mouse_initial_indices, rewardlist, actions_list,seprendrelesmurs=True):

    env = deepcopy(L)
    torch.manual_seed(2020)

    num_inputs = 2+1
    num_actions = 4
    print('state size:', num_inputs)
    print('action size:', num_actions)

    online_net = QNet(num_inputs, num_actions)
    target_net = QNet(num_inputs, num_actions)
    update_target_model(online_net, target_net)

    optimizer = optim.Adam(online_net.parameters(), lr=lr)
    # writer = SummaryWriter('logs')

    online_net.to(device)
    target_net.to(device)
    online_net.train()
    target_net.train()
    memory = Memory(replay_memory_capacity)
    running_score = 0
    epsilon = 1.0
    steps = 0
    loss = 0

    for e in range(10000):
        done = False
        env = deepcopy(L)
        eaubue=0.
        score = 0
        state = np.array(mouse_initial_indices)
        state = torch.Tensor(state).to(device)
        state = state.unsqueeze(0)

        while not done:
            steps += 1

            action = get_action(state, target_net, epsilon, env,seprendrelesmurs=seprendrelesmurs,eaubue=eaubue)
            newstate=state+torch.Tensor(np.array(actions_list[action])).to(device)
            if env[int(newstate[0][0].tolist()),int(newstate[0][1].tolist())]!=0:
                next_state=newstate
                new_eaubue=eaubue
                reward=rewardlist[env[int(newstate[0][0].tolist()),int(newstate[0][1].tolist())]]
                if env[int(newstate[0][0].tolist()),int(newstate[0][1].tolist())]==2:
                    done=True
                if env[int(newstate[0][0].tolist()),int(newstate[0][1].tolist())]==4: #if the mouse is in the water
                    env[int(newstate[0][0].tolist()),int(newstate[0][1].tolist())]=5 #there is no more water
                    new_eaubue=1.
            else :
                next_state=state
                reward=rewardlist[0]
                new_eaubue=eaubue


            mask = 0 if done else 1
            action_one_hot = np.zeros(4)
            action_one_hot[action] = 1
            memory.push(torch.cat((state,torch.tensor(eaubue).unsqueeze(0).unsqueeze(0).to(device)),1), torch.cat((next_state,torch.tensor(new_eaubue).unsqueeze(0).unsqueeze(0).to(device)),1), action_one_hot, reward, mask)

            score += reward
            state = next_state
            eaubue=new_eaubue

            if steps > initial_exploration:
                epsilon -= 0.00005
                epsilon = max(epsilon, 0.1)

                batch = memory.sample(batch_size)
                loss = QNet.train_model(online_net, target_net, optimizer, batch)


                if steps % update_target == 0:
                    update_target_model(online_net, target_net)

        # if score>47:
        #     print(score)
        running_score = 0.99 * running_score + 0.01 * score
        # running_score=score
        if e % log_interval == 0:
            print('{} episode | score: {:.2f} | epsilon: {:.2f}'.format(
                e, running_score, epsilon))
            # writer.add_scalar('log/score', float(running_score), e)
            # writer.add_scalar('log/loss', float(loss), e)
            torch.save(online_net.state_dict(), "./qlearning_model")

        if running_score > goal_score:
            break


def test(L, mouse_initial_indices, rewardlist, actions_list,seprendrelesmurs):
    online_net = QNet(3, 4).to(device)
    online_net.load_state_dict(torch.load("./qlearning_model", map_location=device))
    env = deepcopy(L)

    done = False
    eaubue=0.
    steps = 0
    score = 0
    state = np.array(mouse_initial_indices)
    state = torch.Tensor(state).to(device)
    state = state.unsqueeze(0)
    def progress_loop(done, steps, state, score,eaubue):
        steps += 1

        action = get_action(state, online_net, 1, env,True,seprendrelesmurs,eaubue)
        displacement = np.array(actions_list[action])
        newstate=state+torch.Tensor(displacement).to(device)
        if env[int(newstate[0][0].tolist()),int(newstate[0][1].tolist())]!=0:
            next_state=newstate

            displayer.main_canva.move(
                displayer.mouse,
                *(displacement  * displayer.square_size)
            )

            reward=rewardlist[env[int(newstate[0][0].tolist()),int(newstate[0][1].tolist())]]
            if env[int(newstate[0][0].tolist()),int(newstate[0][1].tolist())]==2:
                done=True
            if env[int(newstate[0][0].tolist()),int(newstate[0][1].tolist())]==4: #if the mouse is in the water
                env[int(newstate[0][0].tolist()),int(newstate[0][1].tolist())]=5 #there is no more water
                eaubue=1.
        else :
            next_state=state
            reward=rewardlist[0]


        score += reward
        state = next_state
        print('position : ',state.tolist()[0],score)

        if done is False:
            displayer.window.after(100, lambda: progress_loop(done, steps, state, score,eaubue))
    displayer = Displayer()

    displayer.create_labyrinth(L, mouse_initial_indices)
    progress_loop(done, steps, state, score,0.)

    displayer.window.mainloop()



if __name__=="__main__":
    # L = np.array(  [[0,0,0,0,0,0,0,0,0,0],
    #                 [0,1,1,3,0,0,1,1,1,0],
    #                 [0,0,1,1,1,1,1,0,1,0],
    #                 [0,1,1,0,1,0,0,0,1,0],
    #                 [0,1,0,0,1,1,1,0,1,0],
    #                 [0,1,0,0,0,0,3,0,1,0],
    #                 [0,1,0,1,1,0,1,0,1,0],
    #                 [0,1,1,1,0,0,1,1,1,0],
    #                 [0,0,0,1,0,0,0,1,0,0],
    #                 [0,1,1,1,1,1,1,1,2,0],
    #                 [0,0,0,0,0,0,0,0,0,0]
    #                 ]).T #Ne pas oublier de repasser l'exploration à 10000
    L=np.array([[0,0,0,0,0],[0,1,1,3,0],[0,1,3,1,0],[0,1,1,2,0],[0,1,1,4,0],[0,0,0,0,0]]).T #labyrinth utilisé (0=mur, 1=vide, 2= arrivée, 3=électricité, 4=eau)
    mouse_initial_indices=[1,1]
    rewardlist=[-5,-1,50,-10,20,-10] #se prendre un mur, se déplacer, arriver au fromage, se prendre l'électricité, boire de l'eau, revenir sur de l'eau
    actions_list=[[1,0],[-1,0],[0,1],[0,-1]]

    # m=main(L, mouse_initial_indices, rewardlist, actions_list,True)
    test(L, mouse_initial_indices, rewardlist, actions_list,True)
