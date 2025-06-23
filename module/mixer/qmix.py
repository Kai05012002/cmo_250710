# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class QMixer(nn.Module):
    """QMixer: 混合各個 agent Q 值為一個全局 Q 值的混合網絡"""
    def __init__(self, n_agents, state_dim, mixing_embed_dim):
        super(QMixer, self).__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.embed_dim = mixing_embed_dim

        # hypernetwork for first layer weights and biases
        self.hyper_w1 = nn.Linear(state_dim, n_agents * mixing_embed_dim)
        self.hyper_b1 = nn.Linear(state_dim, mixing_embed_dim)

        # hypernetwork for second layer weights and biases
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, mixing_embed_dim)
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, 1)
        )

    def forward(self, agent_qs, states):
        """
        agent_qs: Tensor of shape [T,B,A]
        states: Tensor of shape [T,B,S]
        returns: q_tot of shape [T,B,1]
        """
        T, B, S = states.shape

        # 第一層
        w1 = torch.abs(self.hyper_w1(states))  # [T, B, n_agents * embed_dim]
        b1 = self.hyper_b1(states)             # [T, B, embed_dim]
        w1 = w1.view(T, B, self.n_agents, self.embed_dim)  # [T, B, n_agents, embed_dim]
        b1 = b1.view(T, B, 1, self.embed_dim)              # [T, B, 1, embed_dim]

        agent_qs = agent_qs.view(T, B, 1, self.n_agents)   # [T,B,1,n_agents]
        hidden = F.elu(torch.matmul(agent_qs, w1) + b1)     # [T,B,1,embed_dim]
        hidden = hidden.view(T, B, self.embed_dim)         # [T,B,embed_dim]

        # 第二層
        w2 = torch.abs(self.hyper_w2(states))            # [T,B,embed_dim]
        b2 = self.hyper_b2(states)                       # [T,B,1]
        w2 = w2.view(T, B, self.embed_dim, 1)              # [T,B,embed_dim,1]
        b2 = b2.view(T, B, 1, 1)                           # [T,B,1,1]

        hidden = hidden.view(T, B, 1, self.embed_dim)      # [T,B,1,embed_dim]
        q_tot = torch.matmul(hidden, w2) + b2               # [T,B,1,1]
        q_tot = q_tot.view(T, B, 1)                        # [T,B,1]

        return q_tot
