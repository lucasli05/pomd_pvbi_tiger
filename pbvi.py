import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.distributions as torch_dist
seed = 0

# set plotting style
strip_size = 10
label_size = 12
mpl.rcParams["axes.labelsize"] = label_size
mpl.rcParams["xtick.labelsize"] = strip_size
mpl.rcParams["ytick.labelsize"] = strip_size
mpl.rcParams["legend.title_fontsize"] = strip_size
mpl.rcParams["axes.titlesize"] = label_size
mpl.rcParams["figure.titlesize"] = label_size

class TigerEnv():
    def __init__(self, p, q, a, b, num_grids):
        self.state_dim = 2
        self.obs_dim = 2
        self.act_dim = 3

        self.state_labels = ["tiger left", "tiger_right"]
        self.obs_labels = ["hear left", "hear right"]
        self.act_labels = ["listen", "open left", "open right"]

        self.transition_matrix = [[], [], []]
        self.transition_matrix[0] = [[1, 0], [0, 1]]
        self.transition_matrix[1] = [[p, 1-p], [1-q, q]]
        self.transition_matrix[2] = [[q, 1-q], [1-p, p]]
        self.transition_matrix = torch.Tensor(self.transition_matrix)

        self.observation_matrix = [[], [], []]
        self.observation_matrix[0] = [[a, 1-a], [1-b, b]]
        self.observation_matrix[1] = [[1, 0], [0, 1]]
        self.observation_matrix[2] = [[1, 0], [0, 1]]
        self.observation_matrix = torch.Tensor(self.observation_matrix)

        self.reward_matrix = [[-1, 10, -100], [-1, -100, 10]]
        self.reward_matrix = torch.Tensor(self.reward_matrix)
        
        eps = 1e-6
        self.num_grids = num_grids
        self.belief_grid = torch.linspace(eps, 1-eps, num_grids)
        self.belief_transition_matrix = self.compute_belief_transition_matrix(self.belief_grid)

    def compute_belief_transition_matrix(self, belief_grid):
        state_dim = len(belief_grid)
        state_id = torch.arange(state_dim)
        b = torch.stack([belief_grid, 1 - belief_grid]).T

        belief_transition_matrix = torch.zeros(self.act_dim, state_dim, state_dim)
        for a in range(self.act_dim):
            # compute observation predictive
            ## prio pr(s'|b,a) = T*b
            s_next = torch.einsum("ni, ij -> nj", b, self.transition_matrix[a])
            ## post pr(s'|a,b,z) = pr(s'|b,a)*observation_matrix  (pr(z|s',a))
            so_next = torch.einsum("nj, jo -> njo", s_next, self.observation_matrix[a])
            #belife transition
            o_next = so_next.sum(-2)  #sum s'

            for o in range(self.obs_dim):
                # compute posterior b'(s') update beilife state  normalized pr(s'|b,a,z)
                b_next = so_next[:, :, o] / o_next[:, o].unsqueeze(-1)
                
                b_next_id = self.belief2grid(b_next)
                belief_transition_matrix[a, state_id, b_next_id] += o_next[:, o]
        return belief_transition_matrix

    def belief2grid(self, beliefs):
        """ Assign a belief vector to a grid based on l2 distance """
        b = torch.stack([self.belief_grid, 1 - self.belief_grid]).T  ##the sum should be one 
        d_func = lambda x: torch.pow(b - x.view(1, -1), 2).sum(1)
        grid_id = torch.Tensor([torch.argmin(d_func(x)) for x in beliefs]).long()
        return grid_id
# init env
P = 0.8 # left
Q = 0.8 # right
A = 0.85 # left
B = 0.85 # right
num_grids = 100 # number of intervals to divide the belief space
tiger_env = TigerEnv(P, Q, A, B, num_grids)

class PBVI:
    """ Point based value iteration """
    def __init__(self, env, gamma):
        self.env = env
        self.gamma = gamma

        self.transition = env.transition_matrix
        self.observation = env.observation_matrix
        self.reward = env.reward_matrix
        self.belief_grid = env.belief_grid
    
    @staticmethod
    def compute_next_belief(belief, transition, observation):
        """ Compute the next belief point and observation probabilities 
        
        Args:
            belief (torch.tensor): belief vector. size=[n, state_dim]
            transition (torch.tensor): transition matrix. size=[act_dim, state_dim, state_dim]
            observation (torch.tensor): observation matrix. size=[act_dim, state_dim, obs_dim]

        Returns:
            b_next (torch.tensor): next belief vector for each observation and action. 
                size=[n, act_dim, obs_dim, state_dim]
            o_next (torch.tensor): next action-conditioned observation probabilities. 
                size=[n, act_dim, obs_dim]
        """
        # compute observation probabilities
        s_next = torch.einsum("ni, kij -> nkj", belief, transition)
        o_next = torch.einsum("nkj, kjo -> nko", s_next, observation)  # belife transition
        
        # compute posterior
        b_next = torch.einsum("kjo, nkj -> nkoj", observation, s_next)
        b_next = b_next / b_next.sum(-1, keepdim=True)
        return b_next, o_next

    def plan(self, num_belief, max_iter=1000):
        """ Compute alpha vectors for each horizon starting with zero alpha vectors

        Returns:
            alphas (torch.tensor): alpha vectors for each horizon. size=[H, num_belief, state_dim]
        """
        state_dim = self.transition.shape[-1]
        act_dim = self.transition.shape[0]
        
        # uniform sample belief points
        b = torch_dist.Dirichlet(torch.ones(state_dim)).sample((num_belief,)) ## sum 1
        
        # compute the next belief and action-conditioned observation probabilities starting from b
        b_next, o_next_b = self.compute_next_belief(b, self.transition, self.observation)
        
        # compute the next action-conditioned observation probabilities starting from s
        o_next_s = torch.einsum("kij, kjo -> kio", self.transition, self.observation)
        
        # compute the belief reward
        r = torch.einsum("ni, ik -> nk", b, self.reward)

        alphas = [torch.empty(0)] * (max_iter + 1)
        alphas[0] = torch.zeros(num_belief, state_dim)
        for i in range(max_iter):
            # compute next value as the maximum of alpha vectors
            v_next = torch.einsum("nkoj, mj -> nkom", b_next, alphas[i]).max(-1)[0]

            # compute the expected next value by averaging the next observation conditioned on belief
            ev_b = gamma * torch.einsum("nko, nko -> nk", o_next_b, v_next)

            # find the maximizing action
            a_max = torch.argmax(r + ev_b, dim=-1)
            
            # compute the expected next value by averaging the next observation conditioned on state
            ev_s = gamma * torch.einsum("kio, nko -> nki", o_next_s, v_next)
            
            # compute the next alpha vectors
            alphas[i+1] = self.reward.T[a_max] + ev_s[torch.arange(num_belief), a_max]
        
        alphas = torch.stack(alphas)
        self.alphas = alphas
        return alphas

    def compute_v(self, b):
        """ Compute the belief value function as the maximum alpha vectors 
        
        Args:
            b (torch.tensor): belief vector. size=[..., state_dim]

        Returns:
            v (torch.tensor): belief value. size=[T, ...]
        """
        v = torch.einsum("...i, tmi -> t...m", b, self.alphas).max(-1)[0]
        return v

    def policy(self, b):
        """ Compute the softmax belief policy by one step lookahead
        
        Args:
            b (torch.tensor): belief vector. size=[n, state_dim]

        Returns:
            pi (torch.tensor): policy vector. size=[n, act_dim]
        """
        b_next, o_next_b = self.compute_next_belief(b, self.transition, self.observation)
        #b (o',a)
        v_next = self.compute_v(b_next)[-1] #q(b,a)
        
        r = torch.einsum("ni, ik -> nk", b, self.reward)
        q = r + self.gamma * torch.einsum("nkj, nkj -> nk", o_next_b, v_next)
        pi = torch.softmax(q, dim=-1)
        return pi
    

    torch.manual_seed(seed)

gamma = 0.9 # discount factor
num_belief = 100 # number of belief points for pbvi
max_iter = 50 # value iteration steps

# testing belief points
b = tiger_env.belief_grid
b = torch.stack([b, 1-b]).T



pbvi = PBVI(tiger_env, gamma)
alphas = pbvi.plan(num_belief, max_iter)
v_pb = pbvi.compute_v(b)
pi_pb = pbvi.policy(b)

import matplotlib.pyplot as plt
cmap = mpl.cm.get_cmap("viridis")
# fig, ax = plt.subplots(1, 3, figsize=(12, 4))

for t in range(max_iter):
    # plt.figure()
    plt.plot(b[:, 0], v_pb[t], c=cmap(t/max_iter))
plt.show()
print('t')

# plt.set_xlabel("belief tiger left")
# plt.set_ylabel("value")
# plt.set_title("point based value iteration")


# plt.suptitle("value comparison")
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.show()
# print('test')