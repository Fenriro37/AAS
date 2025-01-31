import numpy as np

class PPOBuffer:
    def __init__(self, obs_shape, n_envs, buffer_size):
        self.n_envs = n_envs
        self.buffer_size = buffer_size
        
        self.observations = np.zeros((n_envs, buffer_size, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((n_envs, buffer_size), dtype=np.float32)
        self.rewards = np.zeros((n_envs, buffer_size), dtype=np.float32)
        self.dones = np.zeros((n_envs, buffer_size), dtype=np.float32)
        self.values = np.zeros((n_envs, buffer_size), dtype=np.float32)
        self.log_probs = np.zeros((n_envs, buffer_size), dtype=np.float32)
        self.advantages = np.zeros((n_envs, buffer_size), dtype=np.float32)
        self.returns = np.zeros((n_envs, buffer_size), dtype=np.float32)
        
        self.pos = 0 
        
    def store(self, obs, actions, rewards, dones, values, log_probs):
        # All inputs should have first dimension n_envs
        self.observations[:, self.pos] = obs
        self.actions[:, self.pos] = actions
        self.rewards[:, self.pos] = rewards
        self.dones[:, self.pos] = dones
        self.values[:, self.pos] = values
        self.log_probs[:, self.pos] = log_probs
        
        self.pos += 1
        
    def compute_gae(self, next_values, gamma=0.99, lam=0.95):
        for env_idx in range(self.n_envs):
            lastgaelam = 0  
            for t in reversed(range(self.buffer_size)):
                if self.dones[env_idx, t]:
                    lastgaelam = 0
                    next_value = 0
                else:
                    next_value = next_values[env_idx] if t == self.buffer_size - 1 else self.values[env_idx, t + 1]
                
                delta = self.rewards[env_idx,t] + gamma * next_value  - self.values[env_idx,t]
                lastgaelam = delta + gamma * lam * lastgaelam
                self.advantages[env_idx, t] = lastgaelam 

            self.returns[env_idx, :] = self.advantages[env_idx, :] + self.values[env_idx, :]
        
        #self.advantages = (self.advantages - np.mean(self.advantages)) / (np.std(self.advantages) + 1e-8)

    def get_all_samples(self, batch_size):
        total_samples = self.n_envs * self.buffer_size
        indices = np.arange(total_samples)
        np.random.shuffle(indices)
        
        observations = self.observations.reshape(-1, *self.observations.shape[2:])
        actions = self.actions.reshape(-1)
        rewards = self.rewards.reshape(-1)
        dones = self.dones.reshape(-1)
        values = self.values.reshape(-1)
        log_probs = self.log_probs.reshape(-1)        
        advantages = self.advantages.reshape(-1)
        returns = self.returns.reshape(-1)

        for start in range(0, total_samples, batch_size):
            batch_indices = indices[start:start + batch_size]
            yield (
                observations[batch_indices],
                actions[batch_indices],
                rewards[batch_indices],
                dones[batch_indices],
                values[batch_indices],
                log_probs[batch_indices],
                advantages[batch_indices],
                returns[batch_indices]
            )  

    def clear(self):
        self.pos = 0
        self.observations.fill(0)
        self.actions.fill(0)
        self.rewards.fill(0)
        self.dones.fill(0)
        self.values.fill(0)
        self.log_probs.fill(0)
        self.advantages.fill(0)
        self.returns.fill(0)