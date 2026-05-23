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
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return T.tensor(np.array(self.states)), T.tensor(np.array(self.actions)), T.tensor(np.array(self.probs)), T.tensor(np.array(self.vals)), \
            T.tensor(np.array(self.rewards)), T.tensor(np.array(self.dones)), batches
    
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
        
        # Use simple feedforward network for stability
        self.actor = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.LayerNorm(fc1_dims),
            nn.ReLU(),
            nn.Dropout(0.1),  # Added dropout for regularization
            nn.Linear(fc1_dims, fc2_dims),
            nn.LayerNorm(fc2_dims),
            nn.ReLU(),
            nn.Dropout(0.1),  # Added dropout for regularization
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha, weight_decay=1e-5)  # Added weight decay
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
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
        
        # Use simple feedforward network for stability
        self.critic = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.LayerNorm(fc1_dims),
            nn.ReLU(),
            nn.Dropout(0.1),  # Added dropout for regularization
            nn.Linear(fc1_dims, fc2_dims),
            nn.LayerNorm(fc2_dims),
            nn.ReLU(),
            nn.Dropout(0.1),  # Added dropout for regularization
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
        self.positions = [0, 1, -1]
        
        # Dynamic entropy coefficient for better exploration
        self.entropy_coef = 0.02
        self.entropy_decay = 0.9995
        
        # Learning rate schedulers
        self.actor_scheduler = optim.lr_scheduler.ExponentialLR(self.actor.optimizer, gamma=0.999)
        self.critic_scheduler = optim.lr_scheduler.ExponentialLR(self.critic.optimizer, gamma=0.999)
        
        # Training statistics
        self.training_step = 0

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

    def _compute_returns(self, rewards, dones, values):
        """Compute discounted returns"""
        returns = np.zeros_like(rewards)
        running_return = 0
        for t in reversed(range(len(rewards))):
            if dones[t]:
                running_return = 0
            running_return = rewards[t] + self.gamma * running_return
            returns[t] = running_return
        return returns

    def choose_action(self, observation, wlkfrwd, flag):
        state = observation

        action_probs = self.actor(state)
        
        # Handle different tensor shapes
        if len(action_probs.shape) == 2:
            action_probs = action_probs.squeeze(0)
        
        # Dynamic action selection based on training progress
        dist = Categorical(action_probs)
        
        # More exploration early in training, more exploitation later
        exploration_threshold = max(0.1, 0.3 * (0.99 ** self.training_step))
        
        if wlkfrwd:
            # During walkforward: balance based on training progress
            if np.random.random() < exploration_threshold:
                action = dist.sample()
            else:
                action = T.argmax(action_probs)
        else:
            # During optimization: always sample but with temperature
            action = dist.sample()

        log_prob = dist.log_prob(action)
        value = self.critic(state)
        
        # Handle scalar output
        if value.dim() > 0:
            value = value.squeeze()
        value = value.item()

        return action.item(), log_prob.item(), value

    def learn(self):
        self.training_step += 1
        
        # Decay entropy coefficient
        self.entropy_coef *= self.entropy_decay
        
        for epoch in range(self.n_epochs):
            state_arr, action_arr, old_probs_arr, vals_arr, reward_arr, done_arr, batches = self.memory.generate_batches()
            
            if len(state_arr) == 0:
                continue
                
            # Convert to tensors and move to device
            states = state_arr.to(self.actor.device)
            actions = action_arr.to(self.actor.device)
            old_probs = old_probs_arr.to(self.actor.device)
            rewards = T.tensor(reward_arr, dtype=T.float32).to(self.actor.device)
            dones = T.tensor(done_arr, dtype=T.bool).to(self.actor.device)
            old_values = T.tensor(vals_arr, dtype=T.float32).to(self.actor.device)
            
            # Compute returns
            returns_np = self._compute_returns(reward_arr, done_arr, vals_arr)
            returns = T.tensor(returns_np, dtype=T.float32).to(self.actor.device)
            
            # Compute advantages
            advantages = returns - old_values
            
            # Normalize advantages
            if advantages.std() > 0:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            total_actor_loss = 0
            total_critic_loss = 0
            total_entropy = 0
            batch_count = 0
            
            # Update for several mini-batches
            for batch in batches:
                if len(batch) == 0:
                    continue
                    
                # Get batch data
                batch_states = states[batch]
                batch_actions = actions[batch] 
                batch_old_probs = old_probs[batch]
                batch_advantages = advantages[batch]
                batch_returns = returns[batch]
                
                # Get new action probabilities and values
                new_probs = self.actor(batch_states)
                dist = Categorical(new_probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # Critic values
                critic_value = self.critic(batch_states).squeeze()
                
                # Probability ratio (using log probabilities directly)
                ratio = (new_log_probs - batch_old_probs).exp()
                
                # PPO losses with clipped objectives
                surr1 = ratio * batch_advantages
                surr2 = T.clamp(ratio, 1 - self.policy_clip, 1 + self.policy_clip) * batch_advantages
                actor_loss = -T.min(surr1, surr2).mean() - self.entropy_coef * entropy
                
                # Clipped value loss
                value_pred_clipped = old_values[batch] + (critic_value - old_values[batch]).clamp(-self.policy_clip, self.policy_clip)
                value_losses = (critic_value - batch_returns).pow(2)
                value_losses_clipped = (value_pred_clipped - batch_returns).pow(2)
                critic_loss = 0.5 * T.max(value_losses, value_losses_clipped).mean()
                
                # Total loss
                total_loss = actor_loss + critic_loss
                
                # Optimize
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping
                T.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                T.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                
                self.actor.optimizer.step()
                self.critic.optimizer.step()
                
                # Track losses for debugging
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.item()
                batch_count += 1
                
            # Step learning rate schedulers
            if epoch % 2 == 0:  # Step every 2 epochs to be less aggressive
                self.actor_scheduler.step()
                self.critic_scheduler.step()
            
            # Debug logging
            if epoch % 3 == 0 and batch_count > 0:  # Log every 3 epochs
                avg_actor_loss = total_actor_loss / batch_count
                avg_critic_loss = total_critic_loss / batch_count
                avg_entropy = total_entropy / batch_count
                
                print(f"Step {self.training_step}, Epoch {epoch}: Actor: {avg_actor_loss:.4f}, "
                      f"Critic: {avg_critic_loss:.4f}, Entropy: {avg_entropy:.4f}, "
                      f"Entropy Coef: {self.entropy_coef:.4f}")
    
        self.memory.clear_memory()

        