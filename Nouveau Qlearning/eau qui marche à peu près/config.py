import torch

gamma = 0.99
batch_size = 32
lr = 0.00003
initial_exploration = 3000
goal_score = 200
log_interval = 20
update_target = 100
replay_memory_capacity = 1000

#device = torch.device("cuda")
device = torch.device("cpu")