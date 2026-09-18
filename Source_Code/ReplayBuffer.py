import numpy as np

# Replay Buffer for storing transitions in reinforcement learning
# This buffer allows the agent to sample past experiences to learn from them
class ReplayBuffer():

    # Initializes the replay buffer with a maximum size, input shape, and number of actions
    # max_size: Maximum number of transitions to store in the buffer
    # input_shape: Shape of the state input (e.g., dimensions of the observation space)
    # n_actions: Number of possible actions the agent can take
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_index = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    # Stores a transition in the replay buffer
    # state: Current state of the environment
    # action: Action taken by the agent
    # reward: Reward received after taking the action
    # next_state: State of the environment after taking the action
    # done: Boolean indicating if the episode has ended
    # The transition is stored at the current index, which wraps around when the buffer is full
    def store_transition(self, state, action, reward, next_state, done):
        index = self.mem_index % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = next_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_index += 1

    # Samples a batch of transitions from the replay buffer
    # batch_size: Number of transitions to sample
    # Returns states, actions, rewards, next_states, and dones for the sampled transitions
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_index, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        next_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, next_states, dones