import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from tqdm import tqdm
from buffer import PPOBuffer
from utils import save_models
from impala import IMPALAPolicyNet, IMPALAValueNet

class PPOAgent:
    def __init__(self, envs, num_actions, obs_shape, impala_filters, dense_units, num_envs, buffer_size, 
                 learning_rate, minibatch, epochs, model_name, gamma=0.99, gae_lambda=0.95, 
                 epsilon=0.2, value_coeff=0.5, entropy_coeff=0.02):
        self.envs = envs
        self.num_envs = num_envs
        self.num_actions = num_actions
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.epsilon = epsilon
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        
        self.policy_network = IMPALAPolicyNet(num_actions=num_actions, impala_filters=impala_filters, dense_units=dense_units)
        self.value_network = IMPALAValueNet(impala_filters=impala_filters, dense_units=dense_units)

        self.policy_optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.value_optimizer = tf.keras.optimizers.Adam(learning_rate)

        self.buffer = PPOBuffer(obs_shape, self.num_envs, buffer_size)
        self.buffer_size = buffer_size
        self.minibatch = minibatch
        self.k_epochs = epochs
        self.model_name = model_name

    def select_action(self, obs, training=False):
        """ Selects an action using the policy network """
        policy_probs = self.policy_network(obs, training=training)  
        value = self.value_network(obs, training=training)  
        dist = tfp.distributions.Categorical(probs=policy_probs)
        actions = dist.sample()

        return actions.numpy(), policy_probs.numpy(), value.numpy()

    def update(self, minibatch_size):
        """ Performs PPO update step """
        dataset = self.buffer.get_all_samples(minibatch_size)
        losses = []

        for minibatch in dataset:
            observations, actions, rewards, dones, old_values, old_log_probs, advantages, returns = minibatch
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

            with tf.GradientTape() as policy_tape, tf.GradientTape() as value_tape:
                policy_probs = self.policy_network(observations, training=True)  
                values = self.value_network(observations, training=True)

                selected_log_probs = tf.gather(tf.math.log(policy_probs), actions.astype(int), batch_dims=1)

                ratio = tf.exp(selected_log_probs - old_log_probs)
                clipped_ratio = tf.clip_by_value(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon)
                policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))
                
                value_loss = tf.reduce_mean(tf.square(tf.squeeze(values) - returns))

                entropy_loss = -tf.reduce_mean(tf.reduce_sum(policy_probs * tf.math.log(policy_probs + 1e-8), axis=-1))

                total_policy_loss = policy_loss - self.entropy_coeff * entropy_loss
                total_value_loss = self.value_coeff * value_loss

            policy_grads = policy_tape.gradient(total_policy_loss, self.policy_network.trainable_variables)
            value_grads = value_tape.gradient(total_value_loss, self.value_network.trainable_variables)

            self.policy_optimizer.apply_gradients(zip(policy_grads, self.policy_network.trainable_variables))
            self.value_optimizer.apply_gradients(zip(value_grads, self.value_network.trainable_variables))

            losses.append(total_policy_loss.numpy() + total_value_loss.numpy())

        return np.mean(losses)

    def train(self, max_timesteps):
        """ Training loop for PPO agent """
        obs, _ = self.envs.reset()

        episode_rewards = [[] for _ in range(self.num_envs)]  
        episode_steps = [[] for _ in range(self.num_envs)]  
        current_episode_rewards = [0] * self.num_envs
        current_episode_timesteps = [0] * self.num_envs
        losses = []
        
        timesteps = 0  
        with tqdm(total=max_timesteps, desc="Training Progress", dynamic_ncols=True) as pbar:
            while timesteps < max_timesteps:
                for _ in range(self.buffer_size):
                    actions, policy_probs, values = self.select_action(obs, training=True)

                    next_obs, rewards, dones, _, infos = self.envs.step(actions)

                    self.buffer.store(
                        obs=obs,
                        actions=actions,
                        rewards=rewards,
                        dones=dones,
                        values=values.reshape(-1),
                        log_probs=tf.gather(tf.math.log(policy_probs), actions.astype(int), batch_dims=1).numpy()
                    )

                    for i in range(self.num_envs):
                        current_episode_rewards[i] += rewards[i]
                        current_episode_timesteps[i] += 1

                        if dones[i]:
                            episode_rewards[i].append(current_episode_rewards[i])
                            episode_steps[i].append(current_episode_timesteps[i])

                            current_episode_rewards[i] = 0.0
                            current_episode_timesteps[i] = 0

                    obs = next_obs

                timesteps += self.num_envs * self.buffer_size
                if timesteps > max_timesteps:
                    break
                
                last_values = self.value_network(obs, training=True).numpy().squeeze(-1)
                self.buffer.compute_gae(last_values, gamma=self.gamma, lam=self.gae_lambda)

                for _ in range(self.k_epochs):
                    loss = self.update(minibatch_size=self.minibatch)
                losses.append(loss)

                self.buffer.clear()
                save_models(self.policy_network, self.value_network, self.model_name, "final_models")


                pbar.update(self.num_envs * self.buffer_size)

        return episode_rewards, episode_steps
    """
    def update(self, minibatch_size):
            dataset = self.buffer.get_all_samples(minibatch_size)
            losses = []

            for minibatch in dataset:
                observations, actions, rewards, dones, old_values, old_log_probs, advantages, returns = minibatch
                advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

                with tf.GradientTape() as tape:
                    policy, values = self.network(observations, training=True)  

                    selected_log_probs = tf.gather(tf.nn.log_softmax(policy), actions.astype(int), batch_dims=1)

                    ratio = tf.exp(selected_log_probs - old_log_probs)
                    
                    clipped_ratio = tf.clip_by_value(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon)
                    policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))
                    
                    value_loss = tf.reduce_mean(tf.square(tf.squeeze(values) - returns))
                    
                    entropy_loss = -tf.reduce_mean(tf.reduce_sum(tf.nn.softmax(policy) * tf.nn.log_softmax(policy + 1e-8), axis=-1))
                    
                    total_loss = policy_loss + self.value_coeff * value_loss - self.entropy_coeff * entropy_loss
                    
                # Compute and apply gradients
                gradients = tape.gradient(total_loss, self.network.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.network.trainable_variables))

                losses.append(total_loss.numpy())
            #print(total_loss,policy_loss,value_loss,entropy_loss)

            return np.mean(losses)

        def train(self, max_timesteps):
            obs, _ = self.envs.reset()

            episode_rewards = [[] for _ in range(self.num_envs)]  
            episode_steps = [[] for _ in range(self.num_envs)]  
            current_episode_rewards = [0] * self.num_envs
            current_episode_timesteps = [0] * self.num_envs
            losses = []
            
            timesteps = 0  
            with tqdm(total=max_timesteps, desc="Training Progress", dynamic_ncols=True) as pbar:
                while timesteps < max_timesteps:
                    for _ in range(self.buffer_size):
                        actions, policy, values = self.select_action(obs,training=True)

                        # Take actions in the environment
                        next_obs, rewards, dones, _, infos = self.envs.step(actions)

                        # Store in buffer
                        self.buffer.store(
                            obs=obs,
                            actions=actions,
                            rewards=rewards,
                            dones=dones,
                            values=values.reshape(-1),
                            log_probs=tf.gather(tf.nn.log_softmax(policy), actions.astype(int), batch_dims=1).numpy()
                        )

                        # Track rewards and timesteps for episodes
                        for i in range(self.num_envs):
                            current_episode_rewards[i] += rewards[i]
                            current_episode_timesteps[i] += 1

                            if dones[i]:
                                episode_rewards[i].append(current_episode_rewards[i])
                                episode_steps[i].append(current_episode_timesteps[i])

                                current_episode_rewards[i] = 0.0
                                current_episode_timesteps[i] = 0

                        obs = next_obs

                    timesteps += self.num_envs * self.buffer_size
                    if timesteps > max_timesteps:
                        break
                    
                    last_values = self.network(obs, training=True)[1].numpy().squeeze(-1)
                    self.buffer.compute_gae(last_values, gamma=self.gamma, lam=self.gae_lambda)

                    for _ in range(self.k_epochs):
                        loss = self.update(minibatch_size=self.minibatch)
                    losses.append(loss)

                    self.buffer.clear()
                    save_models(self.network, self.model_name, "final_models")

                    pbar.update(self.num_envs * self.buffer_size)

            return episode_rewards, episode_steps
    """