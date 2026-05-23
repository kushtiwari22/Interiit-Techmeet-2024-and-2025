import torch as T
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from parameters2 import (
    FC1_DIMS, FC2_DIMS, GAMMA, ALPHA, GAE_LAMBDA,
    POLICY_CLIP, BATCH_SIZE, N, N_EPOCHS,
    ENTROPY_COEF_START, ENTROPY_COEF_END, ENTROPY_DECAY,
    EXPLORATION_RATE, HOLD_PENALTY, GRAD_CLIP_NORM, LR_DECAY
)


class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []      # old log-probs
        self.vals = []       # value estimates at time of sampling (for diag only)
        self.actions = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)

        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]

        return (
            T.tensor(np.array(self.states), dtype=T.float32),
            T.tensor(np.array(self.actions)),
            T.tensor(np.array(self.probs), dtype=T.float32),
            T.tensor(np.array(self.vals), dtype=T.float32),
            T.tensor(np.array(self.rewards), dtype=T.float32),
            T.tensor(np.array(self.dones), dtype=T.float32),
            batches,
        )

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)   # scalar log-prob
        self.vals.append(vals)     # scalar value (diag)
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
    def __init__(
        self,
        n_actions,
        input_dims,
        alpha,
        fc1_dims=FC1_DIMS,
        fc2_dims=FC2_DIMS,
        chkpt_dir='C:/Users/Kushagra tiwari/Downloads/INTERIIT_PS1/Files/Files/Model_params',
    ):
        super(ActorNetwork, self).__init__()

        os.makedirs(chkpt_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo_optimized')

        self.actor = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.LayerNorm(fc1_dims),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fc1_dims, fc2_dims),
            nn.LayerNorm(fc2_dims),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim=-1),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
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
    def __init__(
        self,
        input_dims,
        alpha,
        fc1_dims=FC1_DIMS,
        fc2_dims=FC2_DIMS,
        chkpt_dir='C:/Users/Kushagra tiwari/Downloads/INTERIIT_PS1/Files/Files/Model_params',
    ):
        super(CriticNetwork, self).__init__()

        os.makedirs(chkpt_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo_optimized')

        self.critic = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.LayerNorm(fc1_dims),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fc1_dims, fc2_dims),
            nn.LayerNorm(fc2_dims),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fc2_dims, 1),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
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
    def __init__(
        self,
        input_dims,
        n_actions,
        gamma=GAMMA,
        alpha=ALPHA,
        gae_lambda=GAE_LAMBDA,
        policy_clip=POLICY_CLIP,
        batch_size=BATCH_SIZE,
        N=N,
        n_epochs=N_EPOCHS,
    ):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.actor = ActorNetwork(n_actions, input_dims, alpha)
        self.critic = CriticNetwork(input_dims, alpha)
        self.memory = PPOMemory(batch_size)
        self.positions = [0, 1, -1]

        # For now: keep entropy coefficient constant and reasonably high
        self.entropy_coef = ENTROPY_COEF_START
        self.entropy_coef_end = ENTROPY_COEF_END
        self.entropy_decay = ENTROPY_DECAY
        self.learn_step = 0

        self.actor_scheduler = optim.lr_scheduler.ExponentialLR(
            self.actor.optimizer, gamma=LR_DECAY
        )
        self.critic_scheduler = optim.lr_scheduler.ExponentialLR(
            self.critic.optimizer, gamma=LR_DECAY
        )

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

    def _compute_gae(self, rewards, dones, values, next_value):
        """
        Generalized Advantage Estimation (GAE).
        rewards, dones, values, next_value are numpy arrays/scalars.
        """
        advantages = np.zeros_like(rewards)
        last_gae = 0.0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            if dones[t]:
                next_val = 0.0
                last_gae = 0.0

            delta = rewards[t] + self.gamma * next_val - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

    def choose_action(self, observation, wlkfrwd, flag):
        """
        Choose action with anti-hold bias and limited exploration.
        Returns: action(int), log_prob(float), value(float).
        """
        state = observation

        with T.no_grad():
            action_probs = self.actor(state)
            if len(action_probs.shape) == 2:
                action_probs = action_probs.squeeze(0)

            # penalize HOLD (index 0) to encourage trading
            action_probs_modified = action_probs.clone()
            hold_index = 0
            if hold_index < len(action_probs_modified):
                action_probs_modified[hold_index] = (
                    action_probs_modified[hold_index] * HOLD_PENALTY
                )
            action_probs_modified = action_probs_modified / (
                action_probs_modified.sum() + 1e-8
            )

            dist = Categorical(action_probs_modified)

            if wlkfrwd:
                # walkforward: mostly greedy
                if np.random.random() < EXPLORATION_RATE:
                    action = dist.sample()
                else:
                    action = T.argmax(action_probs_modified)
            else:
                # optimisation: sample from modified distribution
                action = dist.sample()

            log_prob = dist.log_prob(action)
            value = self.critic(state)
            if value.dim() > 0:
                value = value.squeeze()

        # store floats (not tensors) in memory
        return (
            int(action.detach().cpu().item()),
            float(log_prob.detach().cpu().item()),
            float(value.detach().cpu().item()),
        )

    def learn(self):
        """
        PPO learn:
        - recompute critic values using current network
        - use them in GAE and targets
        - use stored log-probs as old log-probs
        """
        self.learn_step += 1

        # Optional: comment out entropy decay until stable
        # if self.entropy_coef > self.entropy_coef_end:
        #     self.entropy_coef = max(
        #         self.entropy_coef * self.entropy_decay, self.entropy_coef_end
        #     )

        state_arr, action_arr, old_probs_arr, vals_arr, reward_arr, done_arr, batches = (
            self.memory.generate_batches()
        )

        states = state_arr.to(self.actor.device)
        actions = action_arr.to(self.actor.device)
        old_log_probs = old_probs_arr.to(self.actor.device)

        # recompute current values with critic
        with T.no_grad():
            values_tensor = self.critic(states).squeeze()
        values_np = values_tensor.cpu().numpy()

        rewards_np = reward_arr.cpu().numpy()
        dones_np = done_arr.cpu().numpy()

        # GAE with current values
        advantages_np, returns_np = self._compute_gae(
            rewards_np, dones_np, values_np, values_np[-1]
        )

        advantages = T.tensor(advantages_np, dtype=T.float32).to(self.actor.device)
        returns = T.tensor(returns_np, dtype=T.float32).to(self.actor.device)

        # normalize advantages
        if advantages.std() > 0:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        batch_count = 0

        for epoch in range(self.n_epochs):
            for batch in batches:
                batch_states = states[batch]
                batch_actions = actions[batch]
                batch_old_log_probs = old_log_probs[batch]
                batch_advantages = advantages[batch]
                batch_returns = returns[batch]

                new_probs = self.actor(batch_states)
                dist = Categorical(new_probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                critic_value = self.critic(batch_states).squeeze()

                ratio = (new_log_probs - batch_old_log_probs).exp()
                surr1 = ratio * batch_advantages
                surr2 = T.clamp(
                    ratio, 1.0 - self.policy_clip, 1.0 + self.policy_clip
                ) * batch_advantages

                actor_loss = -T.min(surr1, surr2).mean() - self.entropy_coef * entropy
                critic_loss = F.mse_loss(critic_value, batch_returns)

                total_loss = actor_loss + 0.5 * critic_loss

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()

                T.nn.utils.clip_grad_norm_(self.actor.parameters(), GRAD_CLIP_NORM)
                T.nn.utils.clip_grad_norm_(self.critic.parameters(), GRAD_CLIP_NORM)

                self.actor.optimizer.step()
                self.critic.optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.item()
                batch_count += 1

            self.actor_scheduler.step()
            self.critic_scheduler.step()

        self.memory.clear_memory()

        if self.learn_step % 10 == 0 and batch_count > 0:
            print(
                f" Learn Step {self.learn_step}: "
                f"Actor Loss: {total_actor_loss / batch_count:.4f}, "
                f"Critic Loss: {total_critic_loss / batch_count:.4f}, "
                f"Entropy: {total_entropy / batch_count:.4f}, "
                f"Entropy Coef: {self.entropy_coef:.6f}"
            )
