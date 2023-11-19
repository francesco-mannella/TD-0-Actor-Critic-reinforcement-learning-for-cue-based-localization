import sys
import torch
from torch.distributions import Normal
from  torch.nn.functional import mse_loss
import numpy as np
import matplotlib.pyplot as plt

#device to run model on 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"

class Actor(torch.nn.Module):
    def __init__(self, input_size):
        super(Actor, self).__init__()
        self.fc = torch.nn.Linear(input_size, 4)
        
    def forward(self, x):
        x = self.fc(x)
        return x

class Evaluator(torch.nn.Module):
    def __init__(self, input_size):
        super(Evaluator, self).__init__()
        self.fc = torch.nn.Linear(input_size, 1)
        
    def forward(self, x):
        x = self.fc(x)
        return x

def select_action(network, state):
    ''' Selects an action given state
    Args:
    - network (Pytorch Model): neural network used in forward pass
    - state (Array): environment state
    
    Return:
    - action.items (Array): continuous action
    - noise (Array): noise added to get stochastic action 
    
    '''
    
    #create state tensor
    state_tensor = torch.from_numpy(state.copy()).float().unsqueeze(0).to(DEVICE)
    state_tensor.required_grad = True
    
    #forward pass through network
    action_parameters = network(state_tensor)
    
    action_mean = action_parameters[:,:2]
    noise_sigma = torch.exp(action_parameters[:,2:])
    #get normal distribution
    m = Normal(action_mean, noise_sigma)

    #sample action, get log probability
    action = m.sample()
    noise = action - action_mean
    action_items = action.cpu().detach().numpy().ravel()

    return action_items, noise

def get_value(network, state):
    ''' Gives a value given state
    Args:
    - network (Pytorch Model): neural network used in forward pass
    - state (Array): environment state
    
    Return:
    - value (float): current value of the state
    
    '''
    #create state tensor
    state_tensor = torch.from_numpy(state.copy()).float().unsqueeze(0).to(DEVICE)
    state_tensor.required_grad = True
    
    #forward pass through network
    value = network(state_tensor)

    return value

class FakeArena:

    def __init__(self):

        self.reward_radius = 0.04
        self.reset()

    def reset(self):
        self.state = np.array([0.1, 0.1]) # 0.1*np.random.uniform(-1,1, 2)
        self.direction = 1.*np.pi# 2*np.pi*np.random.uniform(-1, 1)

    def step(self, action):

        speed, direction = action
        speed = 0.03*np.tanh(np.exp(speed)) 
        
        self.direction += 1*(4*np.pi*np.tanh(direction) -self.direction)
        self.state += speed*np.hstack([np.cos(self.direction), np.sin(self.direction)])
        reward = 1*(np.linalg.norm(self.state) < self.reward_radius)
        return self.state, reward

# %%

if __name__ == "__main__":

    episodes = 2000
    N = 2
    stime = 40
    lr = 0.06
    gamma = 0.99
    I_init = 1
    noise_sigma_init = 0.1
    plotting = True

    history = {"episodes": np.zeros([episodes, stime, 2]), "scores": np.zeros(episodes)}

    # create environment
    arena = FakeArena()

    # create perceptron
    actor = Actor(N).to(DEVICE)
    evaluator = Evaluator(N).to(DEVICE)

    # Define the optimizer
    ActorOptimizer = torch.optim.SGD(actor.parameters(), lr=lr)
    EvaluatorOptimizer = torch.optim.SGD(evaluator.parameters(), lr=lr)

   
    if plotting:
        plt.ion()
        nose_length = 0.02
        fig, ax = plt.subplots(1, 1, figsize=(6,6))
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        agent_pos = ax.scatter(0, 0, s=100, fc="white", ec="black", zorder=1)
        agent_nose, = ax.plot([0, nose_length], [0, 0], c="black") 
        agent_path, = ax.plot([0, 0], [0, 0], c="black")
        ts = ax.text(.3, .3, "ts: 0")
        preward = plt.Circle((0, 0), arena.reward_radius, color="green", zorder=0)
        ax.add_patch(preward)

    # Train the model
    for episode in range(episodes):
        
        I = I_init*np.exp(-5*episode/episodes)

        score = 0
        arena.reset()
        state, reward = arena.step([0, 0])
        for t in range(stime):
            
            prev_value = get_value(evaluator, state)

            # Forward pass
            action, noise = select_action(actor, state)
            state, reward = arena.step(action)

            value = get_value(evaluator, state)
            if t == stime - 1: value *= 0
            
            discounted_curr_value = reward + gamma*value 
            td = discounted_curr_value - prev_value

            policy_loss = I*td*torch.mean(noise**2)
            value_loss = I*mse_loss(discounted_curr_value, value) 
            
            history["episodes"][episode, t, :] = arena.state.copy()
            
            # Backward pass
            policy_loss.backward(retain_graph=True)
            value_loss.backward(retain_graph=True)

            # Zero the gradients
            ActorOptimizer.zero_grad()
            EvaluatorOptimizer.zero_grad()
            # Update the weights
            ActorOptimizer.step()
            EvaluatorOptimizer.step()

            if plotting and episode%50==0:
                agent_pos.set_offsets(arena.state)
                agent_nose.set_data(np.vstack([
                    arena.state,
                    arena.state + 
                    nose_length*np.array([
                        np.cos(arena.direction), 
                        np.sin(arena.direction)])
                    ]).T)
                history["episodes"][episode, :t].shape
                agent_path.set_data(*history["episodes"][episode, :t].T)
                ts.set_text(f"ts: {t}")
                plt.pause(0.0001)

                if reward > 0: 
                    if plotting: plt.pause(0.1)

            if reward > 0: 
                history["scores"][episode] += 1

        print(episode, history["scores"][episode])
