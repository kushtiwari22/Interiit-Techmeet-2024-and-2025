import numpy as np
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import os
from parameters import FC1_DIMS, FC2_DIMS, GAMMA, ALPHA, GAE_LAMBDA, POLICY_CLIP, BATCH_SIZE, N, N_EPOCHS


class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transitions(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        start_point = np.random.choice(max_mem - batch_size) # Chooses one point
        batch = range(start_point, start_point + batch_size, 1)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, states_, actions, rewards, dones
    
class CriticNetwork(nn.Module):
    def __init__(self, input_dims, name, alpha, n_actions, fc1_dims=FC1_DIMS, fc2_dims=FC2_DIMS,
                 chkpt_dir='Net/ppo'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.chkpt_dir = chkpt_dir
        self.chkpt_file = os.path.join(self.chkpt_dir, name+'_td3')
        
        self.fc1 = nn.Linear(self.input_dims+n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q1 = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr= alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, actions):
        q1_action_value = self.fc1(T.cat([state, actions], dim= 1))
        q1_action_value = F.relu(q1_action_value)
        q1_action_value = self.fc2(q1_action_value)
        q1_action_value = F.relu(q1_action_value)

        q1 = self.q1(q1_action_value)

        return q1
    
    def save_checkpoint(self):
        print("Saving checkpoint..")
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        print("Loading checkpoint..")
        self.load_state_dict(T.load(self.chkpt_file))

class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, name, n_actions, fc1_dims=FC1_DIMS, fc2_dims=FC2_DIMS,
                 chkpt_dir='Net/ppo'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name # wil remove name later
        self.chkpt_dir = chkpt_dir
        self.chkpt_file = os.path.join(self.chkpt_dir, name+'_td3')

        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr= alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)

        mu = T.tanh(self.mu(prob))

        return mu
    
    def save_checkpoint(self):
        print("Saving checkpoint..")
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        print("Loading checkpoint..")
        self.load_state_dict(T.load(self.chkpt_file))

class Agent(nn.Module):
    def __init__(self, alpha, input_dims, tau, env, gamma= 0.99, update_actor_interval= 2, 
                 warmup= 1000, n_actions= 2, max_size= 1000000, layer1_size= 400, layer2_size= 300,
                 batch_size= 100, noise= 0.1):
        super(Agent, self).__init__()
        self.gamma = gamma
        self.tau = tau
        self.max_action = env.action_space.high # change later
        self.min_actions = env.action_space.low
        self.memory = ReplayBuffer(max_size= max_size, input_shape= input_dims, n_actions= n_actions)
        self.batch_size = batch_size
        self.learn_step_cntr = 0
        self.time_step = 0
        self.warmup = warmup
        self.n_actions = n_actions
        self.update_actor_iter = update_actor_interval

        self.actor = ActorNetwork(alpha= alpha, input_dims= input_dims, fc1_dims= layer1_size, fc2_dims= layer2_size,
                                   n_actions= n_actions, name= 'actor')
        self.critic_1 = CriticNetwork(alpha= alpha, input_dims= input_dims, fc1_dims= layer1_size, fc2_dims= layer2_size, 
                                      n_actions= n_actions, name= 'critic_1')
        self.critic_2 = CriticNetwork(alpha= alpha, input_dims= input_dims, fc1_dims= layer1_size, fc2_dims= layer2_size, 
                                      n_actions= n_actions, name= 'critic_2')
        
        self.target_actor = ActorNetwork(alpha= alpha, input_dims= input_dims, fc1_dims= layer1_size, fc2_dims= layer2_size,
                                   n_actions= n_actions, name= 'target_actor')
        self.target_critic_1 = CriticNetwork(alpha= alpha, input_dims= input_dims, fc1_dims= layer1_size, fc2_dims= layer2_size, 
                                      n_actions= n_actions, name= 'target_critic_1')
        self.target_critic_2 = CriticNetwork(alpha= alpha, input_dims= input_dims, fc1_dims= layer1_size, fc2_dims= layer2_size, 
                                      n_actions= n_actions, name= 'target_critic_2')
        
        self.noise = noise
        self.update_network_parameters(tau= 1)

    def choose_action(self, observation):
        if self.time_step < self.warmup:
            mu = T.tensor(np.random.normal(scale= self.noise, size= (self.n_actions, ))).to(self.actor.device)
        else:
            state = T.tensor(observation, dtype= T.float).to(self.actor.device)
            mu = self.actor.forward(state).to(self.actor.device)

        mu_prime = mu + T.tensor(np.random.normal(scale= self.noise), dtype= T.float).to(self.actor.device)
        mu_prime = T.clamp(mu_prime, self.min_actions[0], self.max_action[0])
        self.time_step += 1

        return mu_prime.cpu().detach().numpy()
    
    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transitions(state= state, action= action, reward= reward, state_= new_state, done= done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        state = T.tensor(state, dtype= T.float).to(self.critic_1.device) # can choose any device, but dtype is sensitive
        done = T.tensor(done).to(self.critic_1.device)
        new_state = T.tensor(new_state, dtype= T.float).to(self.critic_1.device)
        reward = T.tensor(reward, dtype= T.float).to(self.critic_1.device)
        action = T.tensor(action, dtype= T.float).to(self.critic_1.device)

        # updates
        target_actions = self.target_actor.forward(new_state)
        target_actions = target_actions + T.clamp(T.tensor(np.random.normal(scale= 0.2)), -0.5, 0.5)
        target_actions = T.clamp(target_actions, self.min_actions[0], self.max_action[0])
        q1_ = self.target_critic_1.forward(state= new_state, actions= target_actions)
        q2_ = self.target_critic_2.forward(state= new_state, actions= target_actions)

        q1 = self.critic_1.forward(state= state, actions= action)
        q2 = self.critic_2.forward(state= state, actions= action)

        q1_[done] = 0
        q2_[done] = 0

        q1_ = q1_.view(-1)
        q2_ = q1_.view(-1)

        # double ql update rule
        critic_value_ = T.min(q1_, q2_)
        target = reward + self.gamma*critic_value_
        target = target.view(self.batch_size, 1)

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        q1_loss = F.mse_loss(target, q1)
        q2_loss = F.mse_loss(target, q2)
        critic_loss = q1_loss + q2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.learn_step_cntr += 1

        if self.learn_step_cntr % self.update_actor_iter != 0:
            return
        
        self.actor.optimizer.zero_grad()
        actor_q1_loss = self.critic_1.forward(state, self.actor.forward(state= state))
        actor_loss = -T.mean(actor_q1_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau= None):
        # Corner case - write tomorrow
        if tau is None:
            tau = self.tau
        
        actor_params = self.actor.named_parameters()
        critic_1_params = self.critic_1.named_parameters()
        critic_2_params = self.critic_2.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_1_params = self.target_critic_1.named_parameters()
        target_critic_2_params = self.target_critic_2.named_parameters()

        critic_1 = dict(critic_1_params)
        critic_2 = dict(critic_2_params)
        actor = dict(actor_params)
        target_actor = dict(target_actor_params)
        target_critic_1 = dict(target_critic_1_params)
        target_critic_2 = dict(target_critic_2_params)

        for name in critic_1:
            critic_1[name] = tau*critic_1[name].clone() + (1-tau)*target_critic_1[name].clone()

        for name in critic_2:
            critic_2[name] = tau*critic_2[name].clone() + (1-tau)*target_critic_2[name].clone()
            
        for name in actor:
            actor[name] = tau*actor[name].clone() + (1-tau)*target_actor[name].clone()

        self.target_critic_1.load_state_dict(critic_1)
        self.target_critic_2.load_state_dict(critic_2)
        self.target_actor.load_state_dict(actor)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_critic_2.save_checkpoint()

    