import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

# This file contains the neural network architectures used in the Soft Actor-Critic algorithm.
# The networks include the Actor, Critic, and Value networks, each with their own structure and parameters.

# Critic Network
class CriticNetwork(nn.Module):

    # Initializes the Critic Network
    # learn_rate: Learning rate for the optimizer
    # input_dims: Dimensions of the input state
    # n_actions: Number of actions in the action space
    # layer1_dims: Number of neurons in the first hidden layer
    # layer2_dims: Number of neurons in the second hidden layer
    # name: Name of the network (for saving/loading checkpoints)
    # chkpt_dir: Directory to save the model checkpoints
    def __init__(self, learn_rate, input_dims, n_actions, layer1_dims=256, layer2_dims=256,
            name='critic', chkpt_dir='tmp/sac'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.layer1_dims = layer1_dims
        self.layer2_dims = layer2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        # Define the layers of the network
        # The first layer takes both the state and action as input
        self.layer1 = nn.Linear(self.input_dims[0]+n_actions, self.layer1_dims)
        self.layer2 = nn.Linear(self.layer1_dims, self.layer2_dims)
        self.q = nn.Linear(self.layer2_dims, 1)

        # Define the optimizer for the network
        # Using Adam optimizer with the specified learning rate
        self.optimizer = optim.Adam(self.parameters(), lr=learn_rate)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        # Move the network to the appropriate device (GPU or CPU)
        self.to(self.device)

    # Forward pass through the network
    # state: Input state tensor
    # action: Input action tensor
    # Returns the Q-value for the given state-action pair
    def forward(self, state, action):
        action_value = self.layer1(T.cat([state, action], dim=1))
        action_value = F.relu(action_value)
        action_value = self.layer2(action_value)
        action_value = F.relu(action_value)

        q = self.q(action_value)

        return q

    # Save the model checkpoint to the specified file
    # Creates the directory if it does not exist
    def save_checkpoint(self):
        os.makedirs(os.path.dirname(self.checkpoint_file), exist_ok=True)
        T.save(self.state_dict(), self.checkpoint_file)

    # Load the model checkpoint from the specified file
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

# Value Network
class ValueNetwork(nn.Module):

    # Initializes the Value Network
    # learn_rate: Learning rate for the optimizer
    # input_dims: Dimensions of the input state
    # layer1_dims: Number of neurons in the first hidden layer
    # layer2_dims: Number of neurons in the second hidden layer
    # name: Name of the network (for saving/loading checkpoints)
    # chkpt_dir: Directory to save the model checkpoints
    def __init__(self, learn_rate, input_dims, layer1_dims=256, layer2_dims=256,
            name='value', chkpt_dir='tmp/sac'):
        super(ValueNetwork, self).__init__()
        self.input_dims = input_dims
        self.layer1_dims = layer1_dims
        self.layer2_dims = layer2_dims
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        # Define the layers of the network
        # The first layer takes the state as input
        self.layer1 = nn.Linear(*self.input_dims, self.layer1_dims)
        self.layer2 = nn.Linear(self.layer1_dims, layer2_dims)
        self.v = nn.Linear(self.layer2_dims, 1)

        # Define the optimizer for the network
        # Using Adam optimizer with the specified learning rate
        self.optimizer = optim.Adam(self.parameters(), lr=learn_rate)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        # Move the network to the appropriate device (GPU or CPU)
        self.to(self.device)

    # Forward pass through the network
    # state: Input state tensor 
    # Returns the state value for the given state
    def forward(self, state):
        state_value = self.layer1(state)
        state_value = F.relu(state_value)
        state_value = self.layer2(state_value)
        state_value = F.relu(state_value)

        v = self.v(state_value)

        return v

    # Save the model checkpoint to the specified file
    # Creates the directory if it does not exist
    def save_checkpoint(self):
        os.makedirs(os.path.dirname(self.checkpoint_file), exist_ok=True)
        T.save(self.state_dict(), self.checkpoint_file)

    # Load the model checkpoint from the specified file
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

# Actor Network
class ActorNetwork(nn.Module):

    # Initializes the Actor Network
    # learn_rate: Learning rate for the optimizer
    # input_dims: Dimensions of the input state
    # max_action: Maximum action value (used for scaling actions)
    # layer1_dims: Number of neurons in the first hidden layer
    # layer2_dims: Number of neurons in the second hidden layer
    # n_actions: Number of actions in the action space
    # name: Name of the network (for saving/loading checkpoints)
    # chkpt_dir: Directory to save the model checkpoints
    def __init__(self, learn_rate, input_dims, max_action, layer1_dims=256, 
            layer2_dims=256, n_actions=2, name='actor', chkpt_dir='tmp/sac'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.layer1_dims = layer1_dims
        self.layer2_dims = layer2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
        self.max_action = max_action
        self.reparam_noise = 1e-6

        # Define the layers of the network
        # The first layer takes the state as input
        self.layer1 = nn.Linear(*self.input_dims, self.layer1_dims)
        self.layer2 = nn.Linear(self.layer1_dims, self.layer2_dims)

        # The output layers for mean (mu) and standard deviation (sigma)
        # mu represents the mean of the action distribution
        # sigma represents the standard deviation of the action distribution
        # Both mu and sigma are used to sample actions from a normal distribution
        self.mu = nn.Linear(self.layer2_dims, self.n_actions)
        self.sigma = nn.Linear(self.layer2_dims, self.n_actions)

        # Define the optimizer for the network
        # Using Adam optimizer with the specified learning rate
        self.optimizer = optim.Adam(self.parameters(), lr=learn_rate)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        # Move the network to the appropriate device (GPU or CPU)
        self.to(self.device)

    # Forward pass through the network
    # state: Input state tensor
    def forward(self, state):
        prob = self.layer1(state)
        prob = F.relu(prob)
        prob = self.layer2(prob)
        prob = F.relu(prob)

        mu = self.mu(prob)
        sigma = self.sigma(prob)

        # Clamping the standard deviation (sigma) to avoid numerical issues
        # Ensures sigma is within a reasonable range to prevent instability during training
        sigma = T.clamp(sigma, min=5e-3, max=1)

        return mu, sigma

    # Sample actions from the actor network
    # state: Input state tensor
    def sample_normal(self, state, reparameterize=True):

        # Get the mean (mu) and standard deviation (sigma) from the forward pass
        mu, sigma = self.forward(state)

        # Debug NaNs
        if T.isnan(mu).any() or T.isnan(sigma).any():
            print("NaN detected in mu or sigma!")
            print("mu:", mu)
            print("sigma:", sigma)
            raise ValueError("mu or sigma contains NaNs")
        probabilities = Normal(mu, sigma)

        # Sample actions from the normal distribution
        # If reparameterize is True use rsample to allow gradients to flow through the sampling process
        if reparameterize:
            actions = probabilities.rsample()

        # If reparameterize is False use sample with no gradient flow
        else:
            actions = probabilities.sample()

        # Apply tanh activation to the actions to ensure they are within the range [-1, 1]
        # Scale the actions by the maximum action value
        tanh_actions = T.tanh(actions)
        action = tanh_actions*T.tensor(self.max_action).to(self.device)

        # Calculate the log probabilities of the actions
        # This is used for the policy loss (Lπ)
        log_probs = probabilities.log_prob(actions) # Used for Lπ
        log_probs -= T.log(1-tanh_actions.pow(2)+self.reparam_noise) # for correcting the distribution (noise to avoid log(0))
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs

    # Save the model checkpoint to the specified file
    # Creates the directory if it does not exist
    def save_checkpoint(self):
        os.makedirs(os.path.dirname(self.checkpoint_file), exist_ok=True)
        T.save(self.state_dict(), self.checkpoint_file)

    # Load the model checkpoint from the specified file
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))