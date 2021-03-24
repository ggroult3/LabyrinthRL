# -*- coding: utf-8 -*-

import random,operator

def argmax(seq,fn):
    best = seq[0]
    best_score = fn(best)
    for x in seq:
        x_score = fn(x)
        if x_score > best_score:
            best, best_score = x, x_score
    return best

def vector_add(a,b):
    return tuple(map(operator.add,a,b))

orientations = [(1,0),(0,1),(-1,0),(0,-1)]

def turn_right(orientation):
    return orientations[orientations.index(orientation)-1]

def turn_left(orientation):
    return orientations[(orientations.index(orientation)+1) % len(orientations)]

def isnumber(x):
    return hasattr(x,'__init__')

class MDP:
    def __init__(self,init_pos,actlist,terminals,transitions={},states=None,gamma = 0.99):
        if not (0 < gamma <= 1):
            raise ValueError("MDP should have 0 < gamma <=1 values")
        if states : 
            self.states = states
        else:
            self.states = set()
            self.init_pos = init_pos
            self.actlist = actlist
            self.terminals = terminals
            self.transitions = transitions
            self.gamma = gamma
            self.reward = {}
    
    def R(self,state):
        """ Returns a reward for the state """
        return self.reward[state]

    def T(self, state, action):
        """ 
        Transition model
        Inputs : a state & an action
        Returns a list of tuple (probability,result-state) for each state 
        """
        if (self.transitions == {}):
            raise ValueError("Transition model is missing")
        else : 
            return self.transitions[state][action]
    
    def actions(self,state):
        if state in self.terminals:
            return [None]
        else:
            return self.actlist

class GridMDP(MDP):
    def __init__(self,grid,terminals,init_pos=(0,0),gamma=0.99):
        #grid.reverse()
        MDP.__init__(self,init_pos,actlist=orientations,terminals=terminals,gamma=gamma)
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        for x in range(self.cols):
            for y in range(self.rows):
                self.reward[x,y] = grid[y][x]
                if grid[y][x] is not None:
                    self.states.add((x,y))
    
    def T(self,state,action):
        if action is None:
            return [(0.0,state)]
        else : 
            return [(0.8,self.go(state,action)),(0.1,self.go(state,turn_right(action))),(0.1,self.go(state,turn_left(action)))]
    
    def go(self,state,direction):
        state1 = vector_add(state,direction)
        return state1 if state1 in self.states else state

    def to_grid(self,mapping):
        return list(reversed([[mapping.get((x,y),None) for x in range(self.cols)] for y in range(self.rows)]))

    def to_arrows(self,policy):
        chars = {(1,0):'>',(0,1):'^',(-1,0):'<',(0,-1):'v',None:'.'}
        return self.to_grid({s:chars[a] for (s,a) in policy.items()})
    
def value_iteration(mdp,epsilon=0.001):
    STSN = {s:0 for s in mdp.states}
    R, T, gamma = mdp.R, mdp.T, mdp.gamma
    count = 0
    while True:
        count += 1
        STS = STSN.copy()
        delta = 0
        for s in mdp.states:
            STSN[s] = R(s) + gamma * max([sum([p * STS[s1] for (p,s1) in T(s,a)]) for a in mdp.actions(s)])
            delta = max(delta,abs(STSN[s] - STS[s]))
        if delta < epsilon * (1 - gamma) / gamma:
            return STS,count

def best_policy(mdp,tupleSTS):
    STS,count = tupleSTS
    pi = {}
    for s in mdp.states: #Pour chaque etat, on détermine la politique utilisee
        pi[s] = argmax(mdp.actions(s),lambda a : expected_utility(a,s,STS,mdp))
    return pi,count

def expected_utility(a,s,STS,mdp):
    return sum([p * STS[s1] for (p,s1) in mdp.T(s,a)])

def policy_iteration(mdp):
    STS = {s:0 for s in mdp.states} # Dictionnaire contenant les states
    pi = {s: random.choice(mdp.actions(s)) for s in mdp.states} # Dictionnaire contenant les politique pi
    count = 0
    while True:
        count += 1
        STS = policy_evaluation(pi,STS,mdp)
        unchanged = True
        for s in mdp.states:
            a = argmax(mdp.actions(s),lambda a : expected_utility(a,s,STS,mdp))
            if a != pi[s]:
                pi[s] = a
                unchanged = False
        if unchanged:
            return pi,count

def policy_evaluation(pi,STS,mdp,k=20):
    R, T, gamma = mdp.R, mdp.T, mdp.gamma
    for i in range(k):
        for s in mdp.states:
            STS[s] = R(s) + gamma * sum([p * STS[s1] for (p,s1) in T(s,pi[s])])
    return STS

def print_table(table, header=None, sep=' ', numfmt='{}'):
    justs = ['rjust' if isnumber(x) else 'ljust' for x in table[0]]
    if header:
        table.insert(0, header)
    table = [[numfmt.format(x) if isnumber(x) else x for x in row] for row in table]
    sizes = list(map(lambda seq: max(map(len, seq)), list(zip(*[map(str, row) for row in table]))))
    for row in table:
        print(sep.join(getattr(str(x), j)(size) for (j, size, x) in zip(justs, sizes, row)))



# L = np.array([[0,0,0,0,0],[0,1,1,3,0],[0,1,3,1,0],[0,1,1,2,0],[0,1,1,1,0],[0,0,0,0,0]]).T #labyrinth utilisé (0=mur, 1=vide, 2= arrivée, 3=électricité, 4=eau)
# mouse_initial_indices = [1,1]
# rewardlist = [-5,-1,50,-10,20,-10] #se prendre un mur, se déplacer, arriver au fromage, se prendre l'électricité, boire de l'eau, revenir sur de l'eau
# actions_list = [[1,0],[-1,0],[0,1],[0,-1]]

def findEnd(M):
    A = []
    for i in range(len(M)):
        for j in range(len(M[0])):
            if M[i][j] == 2:
                A.append((i,j))
    return A

# L = [[0,0,0,0,0,0,0,0,0,0],
#     [0,1,1,3,0,0,1,1,1,0],
#     [0,0,1,1,1,1,1,0,1,0],
#     [0,1,1,0,1,0,0,0,1,0],
#     [0,1,0,0,1,1,1,0,1,0],
#     [0,1,0,0,0,0,3,0,1,0],
#     [0,1,0,1,1,0,1,0,1,0],
#     [0,1,1,1,0,0,1,1,1,0],
#     [0,0,0,1,0,0,0,1,0,0],
#     [0,1,1,1,1,1,1,1,2,0],
#     [0,0,0,0,0,0,0,0,0,0]]

# L = [[0,0,0,0,0,0],
#       [0,1,1,1,1,0],
#       [0,1,3,1,1,0],
#       [0,3,1,2,1,0],
#       [0,0,0,0,0,0]]



L = [[0,0,0,0,0,0,0,0,0,0],
    [0,1,1,3,0,0,1,1,1,0],
    [0,0,1,1,1,1,1,0,1,0],
    [0,1,1,0,1,0,0,0,1,0],
    [0,1,0,0,1,1,1,0,1,0],
    [0,1,0,0,0,0,3,0,1,0],
    [0,1,0,1,1,0,1,0,1,0],
    [0,1,1,1,0,0,1,1,1,0],
    [0,0,0,1,0,0,0,1,0,0],
    [0,1,1,1,1,1,1,1,2,0],
    [0,0,0,0,0,0,0,0,0,0]]


begin = (1,1)
end = findEnd(L)
print(end)
reward_list = [-5,-1,50,-10,20,-10]



def construct_grid(M,R):
    n = len(M)
    p = len(M[0])
    G = []
    for i in range(n):
        G_row = []
        for j in range(p):
            G_row.append(R[M[i][j]])
        G.append(G_row)
    return G
            
grid = construct_grid(L,reward_list)

# sequential_decision_environment = GridMDP([[-0.02, -0.02, -0.02, +1],
#                                            [-0.02,None,-0.02,-1],
#                                            [-0.02, -0.02, -0.02, -0.02]],
                                          # terminals = [(3,2),(3,1)])

sequential_decision_environment = GridMDP(grid,end,begin)

value_iter,value_count = best_policy(sequential_decision_environment,value_iteration(sequential_decision_environment,.01))

print("\n",value_count,"\n")

print("\n Optimal Policy based on Value Iteration\n")
print_table(sequential_decision_environment.to_arrows(value_iter))

policy_iter,policy_count = policy_iteration(sequential_decision_environment)

print("\n",policy_count,"\n")

print("\n Optimal Policy based on Policy Iteration & Evaluation\n")
print_table(sequential_decision_environment.to_arrows(policy_iter))