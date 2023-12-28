import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys, os
import torch
from torch.distributions import Normal
from torch.nn.functional import mse_loss
from agent import Actor, Evaluator
from simulator import GraphArena, ArenaEnv
matplotlib.use("agg")
plt.ion()

#device to run model on 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def set_position_an_direction(origin=None):
    origin = origin if origin is not None else np.zeros(2)
    distance_from_origin = np.random.uniform(2.0, 2.5)
    direction_from_origin = np.random.uniform(0, 2*np.pi)

    position = distance_from_origin*np.array([
        np.cos(direction_from_origin),
        np.sin(direction_from_origin) ])
    direction = np.random.uniform(-np.pi, np.pi)
    return position, direction



def action_selection(actor, state, exploration = True):

    osize = actor.output_size
    
    # Get the action parameters from the actor
    action_params = actor(state)
    action_mean = action_params[:, :osize]
    action_std = action_params[:, osize:]

    if not exploration:
        action_std = 0.01
    else:
        action_std += 0.01

    # Create a normal distribution with the action mean and standard deviation
    mean_ampls = torch.tensor([0.1, 1]).reshape(1,-1).to(DEVICE)
    prob = torch.distributions.Normal(action_mean*mean_ampls, action_std)

    # Sample an action from the distribution
    action = prob.sample()

    return action.ravel().tolist(), action_mean.ravel().tolist(), action_std.ravel().tolist()

# Initialize arena
arena = GraphArena()

# Set hyperparameters
N = np.prod(arena.retina_dims)
stime = 200


# Create actor and evaluator instances
actor = Actor(N, 2, symmetry=(1, 0)).to(DEVICE)
evaluator = Evaluator(N).to(DEVICE)

actor.load_state_dict(torch.load("params/actor_params", map_location=DEVICE)())
evaluator.load_state_dict(torch.load("params/evaluator_params", map_location=DEVICE)())

# %%


for episode in range(50):

    c_state = arena.reset(*set_position_an_direction())

    # Time step loop
    for t in range(stime):
        
        # Convert current state to torch tensor
        state = torch.tensor(c_state.reshape(1, -1)).float().to(DEVICE)
            
        # Perform action selection using the actor network
        action, action_mean, action_std = action_selection(actor, state)
        
        # Update the current state based on the action
        c_state, reward = arena.step(*action)


        state.detach()

        if reward > 0:
            break
arena.save("demo")
arena.close()



