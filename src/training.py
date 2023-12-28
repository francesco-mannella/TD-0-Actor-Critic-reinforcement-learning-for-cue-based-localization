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


def create_or_switch_environment(arena):
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
    return arena

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
    log_action = prob.log_prob(action)

    return action.ravel().tolist(), log_action, (action_mean, action_std) 

# Initialize arena
arena = ArenaEnv()

# Set hyperparameters
N = np.prod(arena.retina_dims)
episodes = 500
stime = 200
lr = 0.002
gamma = 0.99
I = 1 

scores = np.ones(episodes)*stime

# Create actor and evaluator instances
actor = Actor(N, 2, symmetry=(1, 0)).to(DEVICE)
evaluator = Evaluator(N).to(DEVICE)

# Set up optimizers
actor_optimizer = torch.optim.SGD(actor.parameters(), lr=lr)
evaluator_optimizer = torch.optim.SGD(evaluator.parameters(), lr=lr)

# %%

# Train the model
for episode in range(episodes):

    arena = create_or_switch_environment(arena)
    c_state = arena.reset(*set_position_an_direction())
        
    # Convert current state to torch tensor
    state = torch.tensor(c_state.reshape(1, -1)).float().to(DEVICE)
    
    # Get the initial value of the state from the evaluator
    prev_val = evaluator(state)

    # Time step loop
    for t in range(stime):
        
        # Convert current state to torch tensor
        state = torch.tensor(c_state.reshape(1, -1)).float().to(DEVICE)
            
        # Perform action selection using the actor network
        action, log_action, (action_mean, action_std) = action_selection(actor, state)
        
        # Update the current state based on the action
        c_state, reward = arena.step(*action)

        # Convert the new state to torch tensor
        new_state = torch.tensor(c_state.reshape(1, -1)).float().to(DEVICE)
        
        # Get the value of the new state from the evaluator
        val = evaluator(new_state)
        if reward > 0:
            val *= 0
        
        # Compute the temporal difference (TD) error
        td = reward + gamma*val.item() - prev_val.item()
        
        # Compute the policy loss and value loss
        policy_loss = -td*I*log_action.sum()
        value_loss = I*mse_loss(reward + gamma*val, prev_val)
        
        # Update the actor's parameters
        actor_optimizer.zero_grad()
        policy_loss.backward()
        actor_optimizer.step()
        
        # Update the evaluator's parameters
        evaluator_optimizer.zero_grad()
        value_loss.backward(retain_graph=True)
        evaluator_optimizer.step()

        # Detach the states and values to prevent gradients from flowing backward
        state.detach()
        new_state.detach()
        prev_val = val

        if reward > 0:
            scores[episode] = t
            break

    print(f"{episode}: " + "*"*int(80*scores[episode]/stime))

np.save("scores", scores)
torch.save(actor.state_dict, "actor_params")
torch.save(evaluator.state_dict, "evaluator_params")
