
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


# L=np.array([[0,0,2,0,0,0,0],[0,1,1,0,1,1,0],[0,1,0,0,1,0,0],[0,1,1,1,1,0,0],[0,0,1,0,1,1,0],[0,0,0,0,0,1,0],[0,0,0,0,0,1,0]])
# dep=[6,6]

L=np.array([[0,0,0,0,0],[0,1,1,3,0],[0,1,3,1,0],[0,1,1,2,0],[0,0,0,0,0]]) #labyrinthe utilisé (0=mur, 1=vide, 2= arrivée, 3=électricité, 4=eau)
dep=[1,1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class QNet(nn.Module):

    def __init__(self):
        super(QNet, self).__init__()
        # VOTRE CODE
        ############
        # Définition d'un réseau avec une couche cachée (à 256 neurones par exemple)
        self.fc1=nn.Linear(2, 256)
        self.fc2=nn.Linear(256, 4)

    def forward(self, x):
        # VOTRE CODE
        ############
        # Calcul de la passe avant :
        # Fonction d'activation relu pour la couche cachée
        # Fonction d'activation linéaire sur la couche de sortie
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

class Agent:
    def __init__(self,lab,dep):

        self.env = lab
        self.depart=dep
        self.batch_size = 128
        self.gamma = 0.999
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 200
        self.target_update = 10
        self.num_episodes = 500

        self.n_actions = 4
        self.episode_durations = []

        self.rewardlist=[-5,-1,50,-10,5] #se prendre un mur, se déplacer, arriver au fromage, se prendre l'électricité, boire de l'eau

        self.cumulated_reward = []

        self.policy_net = QNet().to(device)
        self.target_net = QNet().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(10000)

        self.steps_done = 0

    def select_action(self,state):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            # VOTRE CODE
            ############
            # Calcul et renvoi de l'action fournie par le réseau
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1,1) #On renvoie sous forme de tenseur de taille 1x1 l'action maximisant le Q en sortie du réseau

        else:
            # VOTRE CODE
            ############
            # Calcul et renvoi d'une action choisie aléatoirement
            return torch.tensor([[random.randrange(self.n_actions)]], device=device, dtype=torch.long) #Renvoie ici un tenseur valant de 0 à 3

    def process_state(self,state):
        return torch.from_numpy(state).unsqueeze(0).float().to(device)

    def plot_durations(self):
        plt.figure(2)
        plt.clf()
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Reward cumulé')
        plt.plot(self.cumulated_reward)


        plt.pause(0.001)  # pause a bit so that plots are updated
        if is_ipython:
            display.clear_output(wait=True)
            display.display(plt.gcf())

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)

        batch = Transition(*zip(*transitions))


        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # VOTRE CODE
        ############
        # Calcul de Q(s_t,a) : Q pour l'état courant

        state_action_values  = self.policy_net(state_batch).gather(1,action_batch) #on récupère la valeur associée à l'action choisie

        # VOTRE CODE
        ############
        # Calcul de Q pour l'état suivant
        next_state_values= torch.zeros(self.batch_size, device=device)#initialisation des états à 0
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0] #On prend la valeur de Q maximale (sauf si l'on est à l'état final)

        # VOTRE CODE
        ############
        # Calcul de Q future attendue cumulée
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # VOTRE CODE
        ############
        # Calcul de la fonction de perte de Huber
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1)) #unsqueeze pour mettre les valeurs en colonne

        # VOTRE CODE
        ############
        # Optimisation du modèle
        self.optimizer.zero_grad()
        loss.backward() #backpropagation
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1) #on borne les gradients entre -1 et 1 pour éviter l'explosion des gradients
        self.optimizer.step() #on réalise une étape de l'optimisation


    def train_policy_model(self):

        for i_episode in range(self.num_episodes):

            state = np.array(self.depart)
            done=False
            c_reward=0

            for t in count():
                action = self.select_action(self.process_state(state))
                if action==0 :
                    newstate=state+[1,0]
                if action==1 :
                    newstate=state+[-1,0]
                if action==2 :
                    newstate=state+[0,1]
                if action==3 :
                    newstate=state+[0,-1]
                if self.env[newstate[0],newstate[1]]!=0:
                    next_state=newstate
                    reward=self.rewardlist[self.env[newstate[0],newstate[1]]]
                    if self.env[newstate[0],newstate[1]]==2:
                        done=True
                else :
                    next_state=state
                    reward=self.rewardlist[0]

                c_reward += reward
                reward = torch.tensor([reward], device=device)

                if done:
                    next_state = None

                self.memory.push(self.process_state(state), action, self.process_state(next_state) if not next_state is None else None, reward)

                state = next_state

                self.optimize_model()

                if done:
                    self.episode_durations.append(t + 1)
                    self.cumulated_reward.append(c_reward)
                    self.plot_durations()
                    break

            if i_episode % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
                self.save_model()

        self.save_model()
        print('Training completed')
        plt.show()
        print


    def save_model(self):
        torch.save(self.policy_net.state_dict(), "./qlearning_model")

    def load_model(self):
        self.policy_net.load_state_dict(torch.load("./qlearning_model", map_location=device))

    def test(self):
        print('Testing model:')
        state = np.array(self.depart)
        done=False
        c_reward=0

        for t in count():
            action = self.policy_net(self.process_state(state)).max(1)[1].view(1,1).detach() #Calcul de l'action par le réseau

            if action==0 :
                newstate=state+[1,0]
            if action==1 :
                newstate=state+[-1,0]
            if action==2 :
                newstate=state+[0,1]
            if action==3 :
                newstate=state+[0,-1]
            if self.env[newstate[0],newstate[1]]!=0:
                next_state=newstate
                reward=self.rewardlist[self.env[newstate[0],newstate[1]]]
                if self.env[newstate[0],newstate[1]]==2:
                    done=True
            else :
                next_state=state
                reward=self.rewardlist[0]
            c_reward += reward

            state = next_state
            print('position : ',state)

            time.sleep(0.05)
            if done:
                break

        print('Testing completed, reward : ',c_reward)

if __name__ == '__main__':

    # set up matplotlib
    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        from IPython import display

    plt.ion()

    agent = Agent(L,dep)

    # Training phase
    agent.train_policy_model()

    #Testing phase
    agent.load_model()
    agent.test()
