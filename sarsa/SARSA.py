import numpy as np
from display import Displayer
import matplotlib.pyplot as plt
import time
import random



def epsilon_greedy(Q, epsilon, n_actions, s, test):
    """
    Renvoie l'action à choisir selon la politique epsilon-greedy
    ------------------------------------------------------------
    Q : Q-value
    epsilon : probabilité de choisir une action au hasard
    n_actions : nombre d'actions possibles
    s' : état
    ------------------------------------------------------------
    """
    if not test and np.random.rand() < epsilon:
        action = np.random.randint(0, n_actions)
        
    else:
        action = np.argmax(Q[s, :])
    return action

def map_state(L):
    """
    Renvoie la liste des états possibles de l'agent'
    ------------------------------------------------
    L : labyrinthe
    ------------------------------------------------
    """ 
    state_list = []
    for i in range(L.shape[0]):
        for j in range(L.shape[1]):
            if L[i,j] != 0 : state_list.append([i,j])
    return state_list


def reset_eau(L):
    """
    Remet de l'eau sur les cases où elle a été bue'
    ------------------------------------------------
    L : labyrinthe
    ------------------------------------------------
    """
    for i in range(L.shape[0]):
        for j in range(L.shape[1]):
            if L[i,j] == 5 : L[i,j] = 4
    return L
    
def sarsa(alpha, gamma, epsilon, episodes, max_steps, L, test):
    """
    Phase d'apprentissage
    ------------------------------------------------------------
    alpha : learning rate
    gamma : discount factor
    epsilon : probabilité de choisir une action au hasard
    episodes : : nombre d'épisodes pour la phase d'apprentissage
    max_steps : nombre maximum d'itérations par épisode'
    test : bouléen pour effectuer un test sur l'interface ou non
    
    
    ------------------------------------------------------------
    """
    
    state_list = map_state(L)
    n_states = len(state_list)
    n_actions = 4
    Q = np.ones((n_states, n_actions))
    best_scores = [0]
    scores = []
    running_score = 0
    running_score_list = [0]
   
    for episode in range(episodes):
        
        L = reset_eau(L)
        total_reward = 0
        s = 0 # état de départ en haut à gauche
        #s = random.randint(0, len(state_list) - 1)  #état de départ aléatoire
        coord_s = state_list[s] 
        a = epsilon_greedy(Q, epsilon, n_actions, s, test)
        t = 0
        done = False
        
        while t < max_steps:
            
            t += 1
            time.sleep(0)
            coord_s_ = []
            coord_s_.append(coord_s[0] + actions_list[a][0])
            coord_s_.append(coord_s[1] + actions_list[a][1])
            case = L[coord_s_[0],coord_s_[1]]
            
            if case !=0:
                next_state=state_list.index(coord_s_)
                if case==2:
                    done=True
                if case==4: #if the mouse is in the water
                    L[coord_s_[0],coord_s_[1]] = 5 #there is no more water
            else :
                next_state=s
            
            reward=rewardlist[case]
            total_reward += reward
            s_ = next_state
            a_ = epsilon_greedy(Q, epsilon, n_actions, s_, test)
            
            if done:
                Q[s, a] += alpha * ( reward  - Q[s, a] )
            else:
                Q[s, a] += alpha * ( reward + (gamma * Q[s_, a_] ) - Q[s, a] )
                
            s, a = s_, a_
            coord_s = state_list[s]
            
            if done:
                print(f"Cet épisode a prit {t} étapes pour une récompense de : {total_reward}")
                                    
                scores.append(total_reward)
                running_score = 0.99 * running_score + 0.01 * total_reward
                running_score_list.append(running_score)
                
                if total_reward > best_scores[-1]:
                    best_scores.append(total_reward)
                else :
                    best_scores.append(best_scores[-1])
                break
            
    """ sauvegarde de la courbe de convergence lissée"""
    plot_convergence(episodes, running_score_list, best_scores, 'courbe_apprentissage_lissée')
    
        
    if test:
        test_agent(Q, L,n_actions)
        
    return Q


    
def test_agent(Q, L, n_actions, delay=1):
    """
    Permet d'effectuer un test sur l'interface'
    -----------------------------------------------
    Q : Q-fonction trouvée suite à l'apprentissage
    L : labyritnhe
    n_actions : nombre d'actions
    
    ------------------------------------------------
    """
    
    state_list = map_state(L)
    s = 0
    mouse_initial_indices = state_list[s]
    done = False
    epsilon = 0
    total_reward = 0
    steps = 0
    L = reset_eau(L)
        
    def progress_loop(done, steps, s, total_reward,eaubue,L):
            coord_s = state_list[s]
            coord_s_ = []
            a = epsilon_greedy(Q, epsilon, n_actions, s, test=False)
            coord_s_.append(coord_s[0] + actions_list[a][0])
            coord_s_.append(coord_s[1] + actions_list[a][1])
            case = L[coord_s_[0],coord_s_[1]]
            steps += 1
            displacement = np.array(actions_list[a])
            if case !=0:
                next_state=state_list.index(coord_s_)
                displayer.main_canva.move(
                displayer.mouse,
                *(displacement  * displayer.square_size)
            )
                if case==2:
                    done=True
                if case==4: #if the mouse is in the water
                    L[coord_s_[0],coord_s_[1]] = 5
                    #there is no more water
            else :
                next_state=s
                
            reward=rewardlist[case]
            total_reward += reward
            s = next_state
            
            if done == True : 
                
                print("Phase de test terminée --> Score : " + str(total_reward))
                displayer.window.destroy()
            
            if done is False:
                displayer.window.after(500, lambda: progress_loop(done, steps, s, total_reward,eaubue,L))
    displayer = Displayer()

    displayer.create_labyrinth(L, mouse_initial_indices)
    progress_loop(done, steps, s, total_reward,0,L)

    displayer.window.mainloop()


def plot_convergence(number_episodes, scores,best_scores, name_fig):

    
    plt.plot(np.linspace(1, number_episodes, len(scores)), scores, lw=1)
    plt.plot(np.linspace(1, number_episodes, len(best_scores)), best_scores, 'k', label='best')

    plt.xlim(1, number_episodes)
    plt.ylim(max(-1500, 1.2*min(scores)), 1.2*max(best_scores))


    plt.xlabel('episode', fontsize=15)
    plt.ylabel('score', fontsize=15)

    plt.legend(prop={'size':15})
    plt.savefig(str(name_fig)+'.png')
    


if __name__=="__main__":
    
    
    """ Définition du grand labyrinthe """
    
    L = np.array(  [[0,0,0,0,0,0,0,0,0,0],
                    [0,1,1,3,0,0,1,1,1,0],
                    [0,0,1,1,1,1,1,0,4,0],
                    [0,1,1,0,1,0,0,0,1,0],
                    [0,1,0,0,1,1,1,0,1,0],
                    [0,1,0,0,0,0,3,0,1,0],
                    [0,1,0,1,1,0,1,0,1,0],
                    [0,1,1,1,0,0,1,1,1,0],
                    [0,0,0,1,0,0,0,1,0,0],
                    [0,1,1,1,1,1,1,1,2,0],
                    [0,0,0,0,0,0,0,0,0,0]
                    ]).T
    
    """ Définition du petit labyrinhte"""
    
    """
    
    L = np.array([[0,0,0,0,0],
                 [0,1,1,3,0],
                 [0,1,3,1,0],
                 [0,1,1,2,0],
                 [0,1,1,4,0],
                 [0,0,0,0,0]]).T
    """
    
    rewardlist=[-5,-1,50,-10,20,-10] #se prendre un mur, se déplacer, arriver au fromage, se prendre l'électricité, boire de l'eau, revenir sur de l'eau
    actions_list=[[1,0],[-1,0],[0,1],[0,-1]]
    test = True
    alpha = 0.4
    gamma = 0.9999
    epsilon = 0.1
    episodes = 2000
    max_steps = 10000

    timestep_reward = sarsa(alpha, gamma, epsilon, episodes, max_steps, L, test)

