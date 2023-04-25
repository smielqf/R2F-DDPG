import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import autograd_hacks

import random

from collections import namedtuple, deque
from utils import cons_full_grads


Partial_DDPG_Transition = namedtuple('Partial_DDPG_Transition', ['state', 'partial_state', 'action', 'reward', 'next_state', 'partial_next_state', ])
class DDPGAgentTrainer(object):
    def __init__(self, dim_input, dim_output, dim_q_input, args,  agent_index=0, device='cpu', buffer_size=1e6, partial=False):
        self.dim_input = dim_input        
        self.dim_output = dim_output
        self.dim_q_input = dim_q_input
        self.num_units = 64
        self.lr = 5e-4
        self.gamma = 0.95
        self.update_count = 0
        self.update_count_p = 0
        self.update_count_q = 0
        self.agent_index = agent_index

        self.buffer_size = buffer_size
        self.batch_size = args.batch_size
        self.max_episode_len = args.max_episode_len
        self.device = device
        

        # Define main networks
        # Local observation of the agent
        p_input_size = dim_input
        p_output_size = dim_output
        q_input_size = dim_q_input
        q_output_size = 1   # For Q-Value only

        class MLP(nn.Module):
            def __init__(self, input=69, output=5, num_units=64):
                super(MLP, self).__init__()
                self.network_stack = nn.Sequential(
                            nn.Linear(input, num_units),
                            nn.ReLU(),
                            nn.Linear(num_units, num_units),
                            nn.ReLU(),
                            nn.Linear(num_units, output)
                        )

            def forward(self, x):
                x = self.network_stack(x)

                return x

        self.p_network = MLP(input=p_input_size, output=p_output_size, num_units=self.num_units).to(self.device)
        self.q_network = MLP(input=q_input_size, output=q_output_size, num_units=self.num_units).to(self.device)
        self.target_p_network = MLP(input=p_input_size, output=p_output_size, num_units=self.num_units).to(self.device)
        self.target_q_network = MLP(input=q_input_size, output=q_output_size, num_units=self.num_units).to(self.device)
        self.target_p_network.load_state_dict(self.p_network.state_dict())
        self.target_q_network.load_state_dict(self.q_network.state_dict())

        self.p_optimizer = optim.Adam(self.p_network.parameters(), lr=self.lr)
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.full_grads = None

        # Create experience buffer
        class PartialReplayBuffer(object):
            def __init__(self, max_len):
                self.buffer = deque([], maxlen=int(max_len))
            
            def push(self, *args):
                self.buffer.append(Partial_DDPG_Transition(*args))

            def sample(self, batch_size):
                return random.sample(self.buffer, batch_size)

            def __len__(self):
                return len(self.buffer)


        self.replay_buffer = PartialReplayBuffer(self.buffer_size)
        
        # self.max_replay_buffer_len = self.batch_size * self.max_episode_len
        self.max_replay_buffer_len = 400*25

    def act(self, obs):
        """ Compute the action to be taken.
        
        Parameters
        ----------
        obs : np.array
            The local observation of the current agent, with size [observation, ]
        
        Returns
        -------
        np.array
            The action to be taken.
        """
        self.eval()
        _obs = torch.tensor(np.array(obs), dtype=torch.float32).to(self.device).view(1,-1)
        action = self._get_action(self.p_network, _obs)
        if self.device == 'cpu':
            action = action.clone().detach().numpy()[0]
        else:
            action = action.cpu().clone().detach().numpy()[0]
        return action

    def tgt_act(self, obs):
        """ Compute the action to be taken.
        
        Parameters
        ----------
        obs : np.array
            The local observation of the current agent, with size [observation, ]
        
        Returns
        -------
        np.array
            The action to be taken.
        """
        self.eval()
        _obs = torch.tensor(np.array(obs), dtype=torch.float32).to(self.device).view(1,-1)
        action = self._get_action(self.target_p_network, _obs)
        if self.device == 'cpu':
            action = action.clone().detach().numpy()[0]
        else:
            action = action.cpu().clone().detach().numpy()[0]
        return action

    def _get_action(self, p_network, obs):
        """ Return the action computed by p_network.

        Parameters
        ----------
        p_network : torch.nn.Module
            The model which is used to compute action taken by the agent.
        obs_n : list
            Each element is a tensor with size [batch_size, num_local_observation].

        Returns
        -------
        Tensor
            The tensor with  size [num_action, ].
        """
        p = p_network(obs)
        u = torch.empty(p.size())
        u = torch.nn.init.uniform_(u).to(self.device)
        act = F.softmax(p - torch.log(-torch.log(u)), dim=1)
        

        return act

    def single_update(self, obs, action, reward, new_obs):
        _obs = torch.tensor(obs, device=self.device, dtype=torch.float, requires_grad=False).view(1,-1)
        _action = torch.tensor(action, device=self.device, dtype=torch.float, requires_grad=False).view(1, -1)
        _reward = torch.tensor(reward, device=self.device, dtype=torch.float, requires_grad=False).view(1, -1)
        _new_obs = torch.tensor(new_obs, device=self.device, dtype=torch.float, requires_grad=False).view(1,-1)
        _action_next = self.target_p_network(_new_obs).detach()

        cur_q_input = torch.cat([_obs, _action], dim=1)
        next_q_input = torch.cat([_new_obs, _action_next], dim=1)
        q_value = self.q_network(cur_q_input)
        next_q = self.target_q_network(next_q_input).detach()

        q_loss = torch.mean(torch.pow(_reward + self.gamma * next_q - q_value , 2))
        self.q_optimizer.zero_grad()
        q_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.p_network.parameters(), .5)
        self.q_optimizer.step()

        ### Actor
        action = self._get_action(self.p_network, _obs)
        q_input = torch.cat([_obs, _action], dim=1)
        p_loss = -torch.mean(self.q_network(q_input))
        self.p_optimizer.zero_grad()
        p_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), .5)
        self.p_optimizer.step()

        self.make_update_exp(self.p_network, self.target_p_network)
        self.make_update_exp(self.q_network, self.target_q_network)

        return p_loss.clone().detach().cpu().numpy(), q_loss.clone().detach().cpu().numpy(), torch.mean(q_value).clone().detach().cpu().numpy()




    def batch_update_fully_p_train(self, update_gap, batch_index, partial=False):
        """ Update trainable parameters of the trainer.
        
        Parameters
        ----------
        agents : list
            It contains all the agents and each element is an instance of MADDPGAgentTrainer.
        t : int
            The amount of steps taken for training.
        """
        self.train()

        # Replay buffer is not large enough
        if len(self.replay_buffer) < self.max_replay_buffer_len:
            # return {'p_loss': np.inf, 'q_loss': np.inf, 'q-value': np.nan, 'target_q_value': np.nan}
            return .0
        if not self.update_count_p % update_gap == 0:  # Only update every 100 steps
            # return {'p_loss': np.inf, 'q_loss': np.inf, 'q-value': np.nan, 'target_q_value': np.nan}
            self.update_count_p += 1
            return .0
        self.update_count_p += 1

        # Sample data from replay buffer
        # transitions= self.replay_buffer.sample(self.batch_size)
        transitions = []
        for index in batch_index:
          transitions.append(self.replay_buffer.buffer[index])
        

        batch_data = Partial_DDPG_Transition(*zip(*transitions))
        _obs = torch.tensor(batch_data.partial_state, device=self.device, dtype=torch.float, requires_grad=False)
        obs = torch.tensor(batch_data.state, device=self.device, dtype=torch.float, requires_grad=False)
        obs = obs.clone().view(self.batch_size, -1)
        _act = torch.tensor(batch_data.action, device=self.device, dtype=torch.float, requires_grad=False)
        

        

        ### Actor
        action = self._get_action(self.p_network, _obs.view(self.batch_size, -1))
        _act[:, self.agent_index, :] = action
        # _act = action
        q_input = torch.cat([obs, _act.view(self.batch_size, -1)], dim=1)
        p_loss = -torch.mean(self.q_network(q_input))
        self.p_optimizer.zero_grad()
        p_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.p_network.parameters(), .5)
        self.p_optimizer.step()

        self.make_update_exp(self.p_network, self.target_p_network)

        return p_loss.clone().detach().cpu().numpy()


    def simple_full_q_grad(self, q_input, reward,  next_q_input):
        def copy_flat_gradients(param_dict):
            new_grads = []
            for param in param_dict:
                g = torch.clone(param_dict[param].grad).detach().view(-1)
                new_grads.append(g)
            new_grads = torch.cat(new_grads)
            return new_grads
        param_dict = {k:v for k, v in zip(self.q_network.state_dict(), self.q_network.parameters())}
        n_grads = []
        for i in range(q_input.shape[0]):
            q_value = self.q_network(q_input[i].view(1, -1))
            next_q = self.target_q_network(next_q_input).detach()
            q_loss = torch.mean(torch.pow(reward[i] + self.gamma * next_q - q_value , 2))
            self.q_optimizer.zero_grad()
            q_loss.backward()
            full_grad = copy_flat_gradients(param_dict)
            n_grads.append(full_grad)


    def batch_update_fully_q_train(self, update_gap, batch_index, partial=False, act_next=None):
        """ Update trainable parameters of the trainer.
        
        Parameters
        ----------
        agents : list
            It contains all the agents and each element is an instance of MADDPGAgentTrainer.
        t : int
            The amount of steps taken for training.
        """
        self.train()

        # Replay buffer is not large enough
        if len(self.replay_buffer) < self.max_replay_buffer_len:
            # return {'p_loss': np.inf, 'q_loss': np.inf, 'q-value': np.nan, 'target_q_value': np.nan}
            return .0, .0
        if not self.update_count_q % update_gap == 0:  # Only update every 100 steps
            # return {'p_loss': np.inf, 'q_loss': np.inf, 'q-value': np.nan, 'target_q_value': np.nan}
            self.update_count_q += 1
            return .0, .0
        self.update_count_q += 1

        # Sample data from replay buffer
        # transitions= self.replay_buffer.sample(self.batch_size)
        transitions = []
        for index in batch_index:
          transitions.append(self.replay_buffer.buffer[index])
        
        
        batch_data = Partial_DDPG_Transition(*zip(*transitions))
        _obs = torch.tensor(batch_data.state, device=self.device, dtype=torch.float, requires_grad=False)
        _obs_next = torch.tensor(batch_data.next_state, device=self.device, dtype=torch.float, requires_grad=False)
        _act = torch.tensor(batch_data.action, device=self.device, dtype=torch.float, requires_grad=False)
        # _act_next = torch.tensor(batch_data.next_action, device=self.device, dtype=torch.float, requires_grad=False)
        reward = np.array(batch_data.reward)
        obs = _obs.clone().view(self.batch_size, -1)
        act = _act.clone().view(self.batch_size, -1)
        obs_next = _obs_next.clone().view(self.batch_size, -1)

        # Critic
        cur_q_input = torch.cat([obs, act], dim=1)
        next_q_input = torch.cat([obs_next, act_next.reshape(self.batch_size, -1)], dim=1)
        reward = torch.tensor(reward, device=self.device, dtype=torch.float, requires_grad=False).view(self.batch_size, -1)
        

        # self.simple_full_q_grad(cur_q_input, reward, next_q_input)

        
        q_value = self.q_network(cur_q_input)
        next_q = self.target_q_network(next_q_input).detach()

        
        q_loss = torch.mean(torch.pow(reward + self.gamma * next_q - q_value , 2))
        
        self.q_optimizer.zero_grad()
        q_loss.backward(retain_graph=True)
        autograd_hacks.compute_grad1(self.q_network)
        self.full_grads = cons_full_grads(self.q_network.parameters(), batch_size=q_value.shape[0])

        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), .5)

        return q_loss.clone().detach().cpu().numpy(), torch.mean(q_value).clone().detach().cpu().numpy()


    def get_state_dict(self):
        """Return all the trainable parameters.
        
        Returns
        -------
        dict
            The dictionary which contains all traiable parameters.
        """
        state_dict = {
            'p_network': self.p_network.state_dict(), 'target_p_network': self.target_p_network.state_dict(),
            'q_network': self.q_network.state_dict(), 'target_q_network': self.target_q_network.state_dict(),
            'p_optimizer': self.p_optimizer.state_dict(), 'q_optimizer': self.q_optimizer.state_dict()
            }
        return state_dict
    
    def restore_state(self, ckpt):
        """Restore all the trainable parameters.
        
        Parameters
        ----------
        ckpt : dict
            Contain all information for restoring trainable parameters.
        """
        self.p_network.load_state_dict(ckpt['p_network'])
        self.q_network.load_state_dict(ckpt['q_network'])
        self.target_p_network.load_state_dict(ckpt['target_p_network'])
        self.target_q_network.load_state_dict(ckpt['target_q_network'])
        self.p_optimizer.load_state_dict(ckpt['p_optimizer'])
        self.q_optimizer.load_state_dict(ckpt['q_optimizer'])

    def eval(self):
        """ Switch all models inside the trainer into eval mode.
        """
        self.p_network.eval()
        self.q_network.eval()
        self.target_p_network.eval()
        self.target_q_network.eval()
    
    def train(self):
        """ Switch all models inside the trainer into train mode.
        """
        self.p_network.train()
        self.q_network.train()
        self.target_p_network.train()
        self.target_q_network.train()

    def make_update_exp(self, source, target, rate=1e-2):
        """ Use values of parameters from the source model to update values of parameters from the target model. Each update just change values of paramters from the target model slightly, which aims to provide relative stable evaluation. Note that structures of the two models should be the same. 
        
        Parameters
        ----------
        source : torch.nn.Module
            The model which provides updated values of parameters.
        target : torch.nn.Module
            The model which receives updated values of paramters to update itself.
        """
        polyak = rate
        for tgt, src in zip(target.named_parameters(recurse=True), source.named_parameters(recurse=True)):
            assert src[0] == tgt[0] # The identifiers should be the same
            tgt[1].data = polyak * src[1].data + (1.0 - polyak) * tgt[1].data
