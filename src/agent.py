import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys, os
import torch
from torch.distributions import Normal
from torch.nn.functional import mse_loss

#device to run model on 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Actor(torch.nn.Module):
    def __init__(self, input_size, output_size, symmetry=None, sigma_ampl=0.1):
        """
        Initialize the Actor class.
        
        Args:
            input_size (int): The size of the input.
            output_size (int): The size of the output.
            symmetry (ndarray or None, optional): The symmetry values. Defaults to None.
            sigma_ampl (float, optional): The sigma amplification factor. Defaults to 0.1.
        """
        super(Actor, self).__init__()
        self.output_size = output_size
        self.sigma_ampl = sigma_ampl
        self.symmetry = symmetry
        self.idcs = np.arange(self.output_size, dtype=int)
        if symmetry is None:
            symmetry = np.ones(self.output_size)
        self.nsim_idcs = self.idcs[symmetry == 0]
        self.sim_idcs = self.idcs[symmetry == 1]

        self.layer = torch.nn.Linear(input_size, 2*output_size)
        self.layer.weight.data.fill_(0.)
        self.layer.bias.data.fill_(0.)
        
    def forward(self, x):
        """
        Perform forward pass on the network.
        
        Args:
            x (tensor): The input tensor.
            
        Returns:
            tensor: The output tensor.
        """
        x = self.layer(x)
        x[:, self.sim_idcs] = torch.tanh(self.sigma_ampl*x[:,self.sim_idcs])
        x[:, self.nsim_idcs] = torch.sigmoid(self.sigma_ampl*x[:,self.nsim_idcs])

        means = x[:,:self.output_size]
        sigms = torch.sigmoid(self.sigma_ampl*x[:,self.output_size:])
        return torch.cat([means, sigms], dim=1)

class Evaluator(torch.nn.Module):
    def __init__(self, input_size):
        """
        Initialize the Evaluator class.
        
        Args:
            input_size (int): The size of the input.
        """
        super(Evaluator, self).__init__()
        self.layer = torch.nn.Linear(input_size, 1)
        self.layer.weight.data.fill_(0.)
        self.layer.bias.data.fill_(0.)
        
    def forward(self, x):
        """
        Perform forward pass on the network.
        
        Args:
            x (tensor): The input tensor.
            
        Returns:
            tensor: The output tensor.
        """
        x = self.layer(x)
        value = x  
        return value

if __name__ == "__main__":

    def action_selection(actor, state, exploration = True):
        
        # Get the action parameters from the actor
        action_params = actor(state)
        action_mean = action_params[:, 0]
        action_std = action_params[:, 1]

        if not exploration:
            action_std = 0.01

        # Create a normal distribution with the action mean and standard deviation
        prob = torch.distributions.Normal(action_mean, action_std)

        # Sample an action from the distribution
        action = prob.sample()
        log_action = prob.log_prob(action)

        return action, log_action, (action_mean, action_std) 
    
    # Set hyperparameters
    episodes = 200
    stime = 200
    lr = 0.4
    gamma = 0.99
    I = 1 

    # Set up plot
    plt.close("all")
    fig, ax = plt.subplots(1, 1)
    line, = ax.plot(0,0, c="black")
    upp, = ax.plot(0,0, lw=0.4, c="black")
    low, = ax.plot(0,0, lw=0.4, c="black")
    ax.set_xlim(-stime*0.1, stime*1.1)
    ax.set_ylim(-2, 2)

    # Create actor and evaluator instances
    actor = Actor(1, 1).to(DEVICE)
    evaluator = Evaluator(1).to(DEVICE)

    # Set up optimizers
    actor_optimizer = torch.optim.SGD(actor.parameters(), lr=lr)
    evaluator_optimizer = torch.optim.SGD(evaluator.parameters(), lr=lr)

    # Set up history array
    history = np.zeros([stime, 3])

    # Training loop
    for episode in range(episodes):

        print(episode)
        
        # Randomly initialize the current state
        c_state = np.random.uniform(-1, 1)
        
        # Convert current state to torch tensor
        state = torch.tensor([[c_state]]).float().to(DEVICE)
        
        # Get the initial value of the state from the evaluator
        prev_val = evaluator(state)

        # Time step loop
        for t in range(stime):
            
            # Convert current state to torch tensor
            state = torch.tensor([[c_state]]).float().to(DEVICE)
                
            # Perform action selection using the actor network
            action, log_action, (action_mean, action_std) = action_selection(actor, state)
            
            # Update the current state based on the action
            c_state += 0.2*action.cpu().item()
            c_state = np.clip(c_state, -1, 1)

            # Convert the new state to torch tensor
            new_state = torch.tensor([[c_state]]).float().to(DEVICE)
            
            # Get the value of the new state from the evaluator
            val = evaluator(new_state)
            
            # Compute the reward
            s = 0.04
            r = np.exp(-0.5*(s**-2)*(c_state - 0.)**2) 
            r = 1*(r>0.01)
            
            # Compute the temporal difference (TD) error
            td = r + gamma*val.item() - prev_val.item()
            
            # Compute the policy loss and value loss
            policy_loss = -td*I*log_action
            value_loss = I*mse_loss(r + gamma*val, prev_val)
            
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
            
            # Update the history array
            std = action_std.item()
            history[t] = np.ones(3)*c_state + [-std,0, std]
        
        # Update the plot every 20 episodes
        if episode % 20 == 0:
            line.set_data(range(stime), history[:, 1])
            low.set_data(range(stime), history[:, 0])
            upp.set_data(range(stime), history[:, 2])
            plt.pause(0.1)

    # Perform a final evaluation with action standard deviation fixed at 0.01
    with torch.no_grad():
        
        c_state = np.random.uniform(-1, 1)

        # Time step loop
        for t in range(stime):
            
            state = torch.tensor([[c_state]]).float().to(DEVICE)
            action, _, (action_mean, action_std) = action_selection(actor, state, exploration=False)
            c_state += 0.2*action.cpu().item()
            c_state = np.clip(c_state, -1, 1)
            new_state = torch.tensor([[c_state]]).float().to(DEVICE)
            state.detach()
            std = action_std
            history[t] = np.ones(3)*c_state + [-std,0, std]
        
        line.set_data(range(stime), history[:, 1])
        low.set_data(range(stime), history[:, 0])
        upp.set_data(range(stime), history[:, 2])
        plt.pause(0.1)
        input()
