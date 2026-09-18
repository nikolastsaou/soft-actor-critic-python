import torch as T
import numpy as np
import torch.nn.functional as F
from ReplayBuffer import ReplayBuffer
from Networks import ActorNetwork, CriticNetwork, ValueNetwork

# Soft Actor-Critic (SAC) Agent
# This agent implements the Soft Actor-Critic algorithm for continuous action spaces
class Agent():

    # Initializes the SAC agent with various parameters
    # temper: Initial temperature for entropy regularization
    # alpha: Learning rate for the actor network and entropy coefficient
    # beta: Learning rate for the critic and value networks
    # input_dims: Dimensions of the input state
    # env: Environment object (used to determine action space)
    # gamma: Discount factor for future rewards
    # n_actions: Number of actions in the action space
    # max_size: Maximum size of the replay buffer
    # tau: Soft update parameter for target networks
    # batch_size: Size of the batch for training
    # reward_scale: Scaling factor for rewards
    # update_temper: Whether to update the temperature parameter automatically
    # chkpt_dir: Directory to save model checkpoints
    def __init__(self,temper=0.2, alpha=0.0003, beta=0.001, input_dims=[8],
            env=None, gamma=0.99, n_actions=2, max_size=1000000, tau=0.005,
            batch_size=256, reward_scale=0.5, update_temper=True,chkpt_dir='tmp/sac'):

        # For analysis purposes
        # Initialize lists to store histories of various metrics
        self.temper_history = []
        self.q1_mean_history = []
        self.q2_mean_history = []
        self.critic_loss_history = []
        self.actor_loss_history = []
        self.value_loss_history = []

        # Add the initial temperature to the history
        self.temper_history.append(temper)

        # Initialize the agent's parameters
        self.gamma = gamma
        self.tau = tau # For soft update of target parameters
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions

        # Initialize the networks
        # Actor network for policy
        # Critic networks for Q-value estimation
        # Value network for state value estimation
        self.actor = ActorNetwork(alpha, input_dims, n_actions=n_actions,
                    name='actor', max_action=env.action_space.high, chkpt_dir = chkpt_dir)
        self.critic_1 = CriticNetwork(beta, input_dims, n_actions=n_actions,
                    name='critic_1', chkpt_dir = chkpt_dir)
        self.critic_2 = CriticNetwork(beta, input_dims, n_actions=n_actions,
                    name='critic_2', chkpt_dir = chkpt_dir)
        self.value = ValueNetwork(beta, input_dims, name='value', chkpt_dir = chkpt_dir)
        self.target_value = ValueNetwork(beta, input_dims, name='target_value', chkpt_dir = chkpt_dir)

        # Set reward scaling and whether to update the temperature automatically
        self.scale = reward_scale
        self.update_temper = update_temper

        # Initialize the target value network with the same parameters as the value network
        self.update_network_parameters(tau=1)

        # For automatic entropy tuning
        self.temper = temper

        self.target_entropy = -T.prod(T.Tensor(env.action_space.shape)).item() # Set target entropy to -dim(A)
        self.log_entropy_alpha = T.tensor(np.log(self.temper), requires_grad=True, device=self.actor.device) # log_alpha = log(temper)
        self.entropy_alpha_optimizer = T.optim.Adam([self.log_entropy_alpha], lr=alpha) # Optimizer for alpha
   

    # Sample action from the actor network
    def choose_action(self, observation, deterministic=False, smooth=False, smooth_samples=10):
        # If the observation is a numpy array, convert it to a tensor
        # If the observation is a list, convert it to a numpy array
        if isinstance(observation, np.ndarray):
            state = T.from_numpy(observation).float().unsqueeze(0).to(self.actor.device)
        else:
            state = T.tensor(observation, dtype=T.float32).unsqueeze(0).to(self.actor.device)

        # Choose action based on the actor network
        # If deterministic is True, use the mean of the action distribution (used for evaluation)
        if deterministic:
            mu, _ = self.actor.forward(state)
            action = T.tanh(mu) * T.tensor(self.actor.max_action)
        
        # Debugging purposes, take the mean of multiple samples
        elif smooth:
            actions = []
            for _ in range(smooth_samples):
                a, _ = self.actor.sample_normal(state, reparameterize=False)
                actions.append(a)
            action = T.mean(T.stack(actions), dim=0)

        # If deterministic is False, sample from the action distribution
        # This allows for exploration during training
        else:
            action, _ = self.actor.sample_normal(state, reparameterize=False)

        return action.cpu().detach().numpy()[0]

    # Store in the replay buffer
    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    # Update the target networks
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        # Get the named parameters of the target value network and the value network
        # This allows us to access the parameters of the networks by their names
        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        # Create dictionaries from the named parameters for easy manipulation
        # This allows us to perform a soft update of the target network parameters
        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        # Perform the soft update: target_value = tau * value + (1 - tau) * target_value
        for name in value_state_dict:
            value_state_dict[name] = tau*value_state_dict[name].clone() + (1-tau)*target_value_state_dict[name].clone()

        # Load the updated state_dict into the target value network
        # This updates the target network parameters to be a mix of the current value network parameters
        self.target_value.load_state_dict(value_state_dict)

    # Save the model checkpoints
    def save_models(self):
        print('.... saving models ....')
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    # Load the model checkpoints
    def load_models(self):
        print('.... loading models ....')
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()

    # Learn from the replay buffer
    # This function samples a batch of transitions from the replay buffer and updates the networks
    def learn(self):
        if self.memory.mem_index < self.batch_size:
            return

        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size) # Sample a batch of transitions from the replay buffer

        # Convert the sampled transitions to tensors and move them to the appropriate device
        reward = T.tensor(reward, dtype=T.float32).to(self.actor.device)
        done = T.tensor(done, dtype=T.bool).to(self.actor.device)
        state_ = T.tensor(new_state, dtype=T.float32).to(self.actor.device)
        state = T.tensor(state, dtype=T.float32).to(self.actor.device)
        action = T.tensor(action, dtype=T.float32).to(self.actor.device)

        ##########################################
        # --------- Update Value Network ---------
        ##########################################
        with T.no_grad():
            next_action, next_log_prob = self.actor.sample_normal(state_, reparameterize=False)
            q1_ = self.critic_1(state_, next_action)
            q2_ = self.critic_2(state_, next_action)
            min_q_ = T.min(q1_, q2_).view(-1)
            v_target = min_q_ - self.log_entropy_alpha.exp() * next_log_prob.view(-1)
            v_target[done] = 0.0

        v = self.value(state).view(-1)
        value_loss = 0.5 * F.mse_loss(v, v_target.detach())

        # For analysis purposes, store the value loss
        self.value_loss_history.append(value_loss.item())

        self.value.optimizer.zero_grad()
        value_loss.backward()
        self.value.optimizer.step()

        ##########################################
        # --------- Update Actor Network ---------
        ##########################################
        new_action, log_prob = self.actor.sample_normal(state, reparameterize=True)
        q1_new = self.critic_1(state, new_action)
        q2_new = self.critic_2(state, new_action)
        min_q_new = T.min(q1_new, q2_new).view(-1)

        actor_loss = (self.log_entropy_alpha.exp() * log_prob.view(-1) - min_q_new).mean()

        # For analysis purposes, store the actor loss
        self.actor_loss_history.append(actor_loss.item())

        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        ##########################################
        # --------- Update Critic Networks -------
        ##########################################
        with T.no_grad():
            target_value = self.target_value(state_).view(-1)
            target_value[done] = 0.0
            q_target = self.scale * reward + self.gamma * target_value

        q1_old = self.critic_1(state, action).view(-1)
        q2_old = self.critic_2(state, action).view(-1)

        # For analysis purposes, store the mean of Q-values
        self.q1_mean_history.append(q1_old.mean().item())
        self.q2_mean_history.append(q2_old.mean().item())

        critic1_loss = 0.5 * F.mse_loss(q1_old, q_target)
        critic2_loss = 0.5 * F.mse_loss(q2_old, q_target)

        # For analysis purposes, store the critic losses
        self.critic_loss_history.append((critic1_loss + critic2_loss).item())

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        (critic1_loss + critic2_loss).backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        ##########################################
        # ---- Update Entropy Coefficient α ------
        ##########################################
        if self.update_temper:
            # Use log_prob from non-reparameterized sample for entropy tuning
            with T.no_grad():
                _, log_prob_no_reparam = self.actor.sample_normal(state, reparameterize=False)
            alpha_loss = -(self.log_entropy_alpha * (log_prob_no_reparam + self.target_entropy).detach()).mean()

            self.entropy_alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.entropy_alpha_optimizer.step()

            self.temper_history.append(self.log_entropy_alpha.exp().item())

        ##########################################
        # ------- Update Target Value Net --------
        ##########################################
        self.update_network_parameters()