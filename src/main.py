import sys, os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn.functional import mse_loss
from simulator import GraphArena, ArenaEnv
from agent import Actor, Evaluator, get_value, select_action

arena = ArenaEnv()

import agent 

def set_position_an_direction(origin=None):
    origin = origin if origin is not None else np.zeros(2)
    distance_from_origin = np.random.uniform(0.8, 1.5)
    direction_from_origin = np.random.uniform(0, 2*np.pi)

    position = distance_from_origin*np.array([
        np.cos(direction_from_origin),
        np.sin(direction_from_origin) ])
    direction = np.random.uniform(-np.pi, np.pi)
    return position, direction

plt.close("all")
plt.ion()
DEVICE = "cpu"

# create environment

episodes = 20000
N = np.prod(arena.retina_dims)
stime = 400
lr = 0.001
gamma = 0.99
I = 1

history = {"episodes": np.zeros([episodes]), "scores": np.zeros(episodes)}

# create perceptron
actor = Actor(N).to(DEVICE)
evaluator = Evaluator(N).to(DEVICE)

# Define the optimizer
ActorOptimizer = torch.optim.SGD(actor.parameters(), lr=lr)
EvaluatorOptimizer = torch.optim.SGD(evaluator.parameters(), lr=lr)

# Train the model
for episode in range(episodes):

    if os.path.exists("PLOT"):
        if type(arena) == ArenaEnv:
            garena = GraphArena()
            garena.copy(arena)
            arena = garena
    else:
        if type(arena) == GraphArena:
            ngarena = ArenaEnv()
            ngarena.copy(arena)
            arena.close()
            arena = ngarena
    

    score = 0
    arena.reset(*set_position_an_direction())
    state, reward = arena.step(0, 0)
    for t in range(stime):
        
        prev_value = get_value(evaluator, state)

        # Forward pass
        action, noise = select_action(actor, state)
        state, reward = arena.step(*(action*0.2))

        value = get_value(evaluator, state)
        
        discounted_curr_value = reward + gamma*value 
        td = discounted_curr_value - prev_value

        policy_loss = -td*torch.mean(noise**2)
        value_loss = mse_loss(discounted_curr_value, value) 
        
        # Backward pass
        policy_loss.backward(retain_graph=True)
        value_loss.backward(retain_graph=True)

        # Zero the gradients
        ActorOptimizer.zero_grad()
        EvaluatorOptimizer.zero_grad()
        # Update the weights
        ActorOptimizer.step()
        EvaluatorOptimizer.step()

        if reward > 0: 
            history["episodes"][episode] = t
            history["scores"][episode] = 1
            break

    print(episode, "*"*history["scores"][episode].astype(int))
