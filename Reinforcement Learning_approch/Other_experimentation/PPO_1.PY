import torch as T
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from parameters import FC1_DIMS, FC2_DIMS, GAMMA, ALPHA, GAE_LAMBDA, POLICY_CLIP, BATCH_SIZE, N, N_EPOCHS

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)  # Added shuffling for better learning
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return T.tensor(np.array(self.states), dtype=T.float32), \
               T.tensor(np.array(self.actions)), \
               T.tensor(np.array(self.probs), dtype=T.float32), \
               T.tensor(np.array(self.vals), dtype=T.float32), \
               T.tensor(np.array(self.rewards), dtype=T.float32), \
               T.tensor(np.array(self.dones)), batches
    
    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []
    
class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha, 
                 fc1_dims=FC1_DIMS, fc2_dims=FC2_DIMS,
                 chkpt_dir='C:/Users/Kushagra tiwari/Downloads/INTERIIT_PS1/Files/Files/Model_params'):
        super(ActorNetwork, self).__init__()

        os.makedirs(chkpt_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')
        
        # Simpler network architecture for stability
        self.actor = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.LayerNorm(fc1_dims),
            nn.ReLU(),
            nn.Dropout(0.1),  # Added dropout for regularization
            nn.Linear(fc1_dims, fc2_dims),
            nn.LayerNorm(fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha, weight_decay=1e-5)  # Added weight decay
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        # Ensure proper input shape
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        
        dist = self.actor(state)
        return dist
    
    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha,
                 fc1_dims=FC1_DIMS, fc2_dims=FC2_DIMS,
                 chkpt_dir='C:/Users/Kushagra tiwari/Downloads/INTERIIT_PS1/Files/Files/Model_params'):
        super(CriticNetwork, self).__init__()

        os.makedirs(chkpt_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')
        
        self.critic = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.LayerNorm(fc1_dims),
            nn.ReLU(),
            nn.Dropout(0.1),  # Added dropout for regularization
            nn.Linear(fc1_dims, fc2_dims),
            nn.LayerNorm(fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=alpha, weight_decay=1e-5)  # Added weight decay
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            
        value = self.critic(state)
        return value
    
    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class Agent:
    def __init__(self, input_dims, n_actions, gammma=GAMMA, alpha=ALPHA, gae_lambda=GAE_LAMBDA,
                 policy_clip=POLICY_CLIP, batch_size=BATCH_SIZE, N=N, n_epochs=N_EPOCHS):
        self.gamma = gammma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.actor = ActorNetwork(n_actions, input_dims, alpha)
        self.critic = CriticNetwork(input_dims, alpha)
        self.memory = PPOMemory(batch_size)
        
        # Better exploration management
        self.entropy_coef = 0.001  # Reduced entropy coefficient
        self.max_entropy_coef = 0.001
        self.min_entropy_coef = 0.0001
        self.entropy_decay = 0.999
        
        # Training statistics
        self.train_step = 0

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('...saving models...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('...loading models...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation, wlkfrwd=False, flag=None):
        state = T.tensor(observation, dtype=T.float32).to(self.actor.device)
        
        with T.no_grad():
            action_probs = self.actor(state)
        
        # Ensure proper shape
        if len(action_probs.shape) == 2:
            action_probs = action_probs.squeeze(0)
        
        # More conservative action modification
        action_probs_modified = action_probs.clone()
        hold_index = 1  # Assuming HOLD is at index 1
        
        if hold_index < len(action_probs_modified):
            # Less aggressive hold probability adjustment
            action_probs_modified[hold_index] = action_probs_modified[hold_index] * 0.98
            action_probs_modified = action_probs_modified / (action_probs_modified.sum() + 1e-8)
        
        dist = Categorical(action_probs_modified)
        
        # More controlled exploration strategy
        if wlkfrwd:
            # During walkforward: 90% greedy, 10% exploratory
            if np.random.random() < 0.1:
                action = dist.sample()
            else:
                action = T.argmax(action_probs_modified)
        else:
            # Always sample during training, but with temperature
            action = dist.sample()

        log_prob = dist.log_prob(action)
        
        with T.no_grad():
            value = self.critic(state)
            value = value.squeeze().item()

        return action.item(), log_prob.item(), value

    def learn(self):
        self.train_step += 1
        
        # Decay entropy coefficient
        self.entropy_coef = max(
            self.min_entropy_coef, 
            self.entropy_coef * self.entropy_decay
        )
        
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_probs_arr, vals_arr, reward_arr, done_arr, batches = self.memory.generate_batches()
            
            values = vals_arr.numpy()
            advantages = np.zeros(len(reward_arr), dtype=np.float32)
            last_advantage = 0
            
            # Fixed advantage calculation with proper bootstrapping
            for t in reversed(range(len(reward_arr))):
                if t == len(reward_arr) - 1:
                    # If terminal state, no bootstrap
                    next_value = 0.0
                    next_advantage = 0.0
                else:
                    next_value = values[t+1] * (1 - done_arr[t])
                    next_advantage = advantages[t+1]
                
                delta = reward_arr[t] + self.gamma * next_value - values[t]
                advantages[t] = delta + self.gamma * self.gae_lambda * (1 - done_arr[t]) * next_advantage
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            advantage_tensor = T.tensor(advantages, dtype=T.float32).to(self.actor.device)
            returns_tensor = advantage_tensor + T.tensor(values, dtype=T.float32).to(self.actor.device)
            
            total_actor_loss = 0
            total_critic_loss = 0
            total_entropy = 0
            
            for batch in batches:
                if len(batch) == 0:
                    continue
                    
                states = state_arr[batch].float().to(self.actor.device)
                old_probs = old_probs_arr[batch].float().to(self.actor.device)
                actions = action_arr[batch].to(self.actor.device)
                batch_advantages = advantage_tensor[batch]
                batch_returns = returns_tensor[batch]
                
                # Get new action probabilities
                new_action_probs = self.actor(states)
                dist = Categorical(new_action_probs)
                new_probs = dist.log_prob(actions)
                
                # Calculate entropy
                entropy = dist.entropy().mean()
                
                # Critic value
                critic_value = self.critic(states).squeeze()
                
                # Probability ratio
                prob_ratio = (new_probs - old_probs).exp()
                
                # Clipped objective
                weighted_probs = batch_advantages * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip) * batch_advantages
                
                # Actor loss with entropy
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()
                actor_loss = actor_loss - self.entropy_coef * entropy
                
                # Critic loss
                critic_loss = F.mse_loss(critic_value, batch_returns)
                
                # Total loss
                total_loss = actor_loss + 0.5 * critic_loss
                
                # Track metrics
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.item()
                
                # Update networks
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping
                T.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                T.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                
                self.actor.optimizer.step()
                self.critic.optimizer.step()
            
            # Print training progress occasionally
            if self.train_step % 10 == 0:
                avg_actor_loss = total_actor_loss / len(batches)
                avg_critic_loss = total_critic_loss / len(batches)
                avg_entropy = total_entropy / len(batches)
                print(f"Step {self.train_step}, Actor Loss: {avg_actor_loss:.4f}, "
                      f"Critic Loss: {avg_critic_loss:.4f}, Entropy: {avg_entropy:.4f}")
                
        self.memory.clear_memory()