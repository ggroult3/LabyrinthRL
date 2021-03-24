import torch

# #Pour petit labyrinthe avec eau
# gamma = 0.99
# batch_size = 32
# lr = 0.00003
# initial_exploration = 3000
# goal_score = 200
# log_interval = 10
# update_target = 100
# replay_memory_capacity = 1000
# device = torch.device("cuda")

# #Grand labyrinthe sans eau
# gamma = 0.99
# batch_size = 32
# lr = 0.001
# initial_exploration = 50000
# goal_score = 200
# log_interval = 10
# update_target = 100
# replay_memory_capacity = 1000
# device = torch.device("cuda")

#Grand labyrinthe avec eau
gamma = 0.99
batch_size = 32
lr = 0.001
initial_exploration = 50000
goal_score = 200
log_interval = 10
update_target = 100
replay_memory_capacity = 5000
try:
    device = torch.device("cuda")
except:
    device = torch.device("cpu")