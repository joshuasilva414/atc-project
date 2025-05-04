import numpy as np

class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_count = 0
        self.state_mem = np.zeros((self.mem_size, *input_shape))
        self.new_state_mem = np.zeros((self.mem_size, *input_shape))
        self.action_mem = np.zeros((self.mem_size, n_actions))
        self.rewared_mem = np.zeros(self.mem_size)
        self.terminal_mem = np.zeros(self.mem_size, dtype=np.bool)
        
    def store_transition(self, state, action reward, state_, done):
        index = self.mem_count % self.mem_size

        self.state_mem[index] = state
        self.new_state_mem[index] = state_
        self.action_mem[index] = action
        self.rewared_mem[index] = reward
        self.terminal_mem[index] = done

        self.mem_count += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_count, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_mem[batch]
        states_ = self.new_state_mem[batch]
        actions = self.action_mem[batch]
        rewards = self.rewared_mem[batch]
        dones = self.terminal_mem[batch]

        return states, actions, rewards, states_, dones