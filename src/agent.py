import sys
import torch
from torch.distributions import Normal
from  torch.nn.functional import mse_loss
import numpy as np
import matplotlib.pyplot as plt

#device to run model on 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
    - action.item() (float): continuous action
    - log_action (float): log of probability density of action
    
    '''
    
    #create state tensor
    state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
    state_tensor.required_grad = True
    
    #forward pass through network
    action_parameters = network(state_tensor)
    
    #get mean and std, get normal distribution
    mu, sigma = action_parameters[:, :2], torch.exp(action_parameters[:, 2:])
    m = Normal(mu, sigma)

    #sample action, get log probability
    action = m.sample()
    log_action = m.log_prob(action)
    action_items = action.detach().numpy().ravel()

    return action_items, log_action

def get_value(network, state):
    ''' Gives a value given state
    Args:
    - network (Pytorch Model): neural network used in forward pass
    - state (Array): environment state
    
    Return:
    - value (float): current value of the state
    
    '''
    #create state tensor
    state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
    state_tensor.required_grad = True
    
    #forward pass through network
    value = network(state_tensor)

    return value

class FakeArena:

    def __init__(self):

        self.reset()

    def reset(self):
        self.state = 0.1*np.random.uniform(-1,1, 2)
        self.direction = 2*np.pi*np.random.uniform(-1, 1)

    def step(self, action):

        speed, direction = action
        speed = np.exp(speed) 
        speed *= 0.01
        
        self.direction += 0.3*direction 
        self.state += speed*np.hstack([np.cos(self.direction), np.sin(self.direction)])
        reward = 1*(np.linalg.norm(self.state) < 0.01)
        return self.state, reward

# %%

if __name__ == "__main__":

    epochs = 100000
    N = 2
    stime = 20
    lr = 0.001
    gamma = 0.99
    plotting = False

    history = {"episodes": np.zeros([epochs, stime, 2]), "timesteps": stime*np.ones(epochs)}

    # create environment
    arena = FakeArena()

    # create perceptron
    actor = Actor(N)
    evaluator = Evaluator(N)

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
        preward = plt.Circle((0, 0), 0.01, color="green", zorder=0)
        ax.add_patch(preward)

    # Train the model
    for epoch in range(epochs):

        arena.reset()
        # actor.noise = noise*np.exp(-epoch/epochs)

        state, reward = arena.step([0, 0])
        value = get_value(evaluator, state)
        
        for t in range(stime):
            
            prev_value = value

            # Forward pass
            action, logprob = select_action(actor, state)
            state, reward = arena.step(action)
            value = get_value(evaluator, state)
            
            discounted_curr_value = reward + gamma*value 
            td = discounted_curr_value - prev_value

            policy_loss = -td*torch.sum(logprob) 
            value_loss = mse_loss(discounted_curr_value, value) 
            
            history["episodes"][epoch, t, :] = arena.state.copy()
            
            # Backward pass
            policy_loss.backward(retain_graph=True)
            value_loss.backward(retain_graph=True)

            # Zero the gradients
            ActorOptimizer.zero_grad()
            EvaluatorOptimizer.zero_grad()
            # Update the weights
            ActorOptimizer.step()
            EvaluatorOptimizer.step()

            if plotting:
                agent_pos.set_offsets(arena.state)
                agent_nose.set_data(np.vstack([
                    arena.state,
                    arena.state + 
                    nose_length*np.array([
                        np.cos(arena.direction), 
                        np.sin(arena.direction)])
                    ]).T)
                history["episodes"][epoch, :t].shape
                agent_path.set_data(*history["episodes"][epoch, :t].T)
                ts.set_text(f"ts: {t}")
                plt.pause(0.0001)

            if reward > 0: 
                history["timesteps"][epoch] = t
                if plotting: plt.pause(0.1)
                break
        print(epoch, history["timesteps"][epoch])
