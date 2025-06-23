import torch.nn as nn
import torch.nn.functional as F
import torch

class Feudal_ManagerAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(Feudal_ManagerAgent, self).__init__()
        self.args = args

        # Manager network
        self.manager_fc1 = nn.Linear(input_shape, args.manager_hidden_dim)
        self.manager_rnn = nn.LSTM(args.manager_hidden_dim, args.manager_hidden_dim, batch_first=False)

        # self.manager_rnn = DilatedLSTMCell(args.manager_hidden_dim, args.manager_hidden_dim, dilation=2)
        # self.manager_fc2 = nn.Linear(args.manager_hidden_dim, args.state_dim_d)

        # 目標生成
        self.goal_network = nn.Linear(args.manager_hidden_dim, args.state_dim_d)
        
        # 狀態值估計 V_t^M
        # self.value_network = nn.Linear(args.manager_hidden_dim, 1)

    def init_hidden(self, batch_size: int = 1):
        # 初始化 Manager 隱藏與 cell 狀態: [num_layers, batch_size * n_agents, hidden_dim]
        B, A = batch_size, self.args.n_agents
        device = next(self.parameters()).device
        h = torch.zeros(1, B*A, self.args.manager_hidden_dim, device=device)
        c = torch.zeros(1, B*A, self.args.manager_hidden_dim, device=device)
        return (h, c)

    def forward(self, inputs, hidden):
        # inputs: [T, B, A, feat]
        T, B, A, feat = inputs.shape
        # fold agent dim into batch
        x = inputs.view(T, B*A, feat)
        # feature transform
        features_flat = F.relu(self.manager_fc1(x))  # [T, B*A, manager_hidden_dim]
        # RNN forward
        out, (h, c) = self.manager_rnn(features_flat, hidden)  # out: [T, B*A, manager_hidden_dim]
        # generate goals
        goal_flat = self.goal_network(out)  # [T, B*A, state_dim_d]
        # reshape outputs back to [T, B, A, ...]
        features = features_flat.view(T, B, A, self.args.manager_hidden_dim)
        goal = goal_flat.view(T, B, A, self.args.state_dim_d)
        return features, goal, (h, c)
    

class Feudal_WorkerAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(Feudal_WorkerAgent, self).__init__()
        self.args = args
        
        # Worker 網絡
        self.worker_fc1 = nn.Linear(input_shape, args.worker_hidden_dim)
        self.worker_rnn = nn.LSTM(args.worker_hidden_dim, args.worker_hidden_dim, batch_first=False)
        # self.worker_rnn = nn.GRUCell(args.worker_hidden_dim, args.worker_hidden_dim)
        
        # U_t: Action embedding matrix (n_actions x 16)
        self.U_embedding = nn.Linear(args.worker_hidden_dim, args.embedding_dim_k * args.n_actions)
        
        # w_t: 優勢方向/權重 (1x16)
        self.w_network = nn.Linear(args.state_dim_d, args.embedding_dim_k)
        
        # # 最終的 Q 值輸出
        # self.q_network = nn.Linear(args.embedding_dim, args.n_actions)

    def init_hidden(self, batch_size: int = 1):
        # 初始化 Worker 隱藏與 cell 狀態: [num_layers, batch_size * n_agents, hidden_dim]
        B, A = batch_size, self.args.n_agents
        device = next(self.parameters()).device
        h = torch.zeros(1, B*A, self.args.worker_hidden_dim, device=device)
        c = torch.zeros(1, B*A, self.args.worker_hidden_dim, device=device)
        return (h, c)
    
    def forward(self, inputs, worker_hidden, goal):
        # inputs: [T, B, A, feat]; goal: [T, B, A, state_dim_d]
        T, B, A, feat = inputs.shape
        # fold agent dim into batch
        x = inputs.view(T, B*A, feat)  # [T, B*A, feat]
        # feature transform
        x = F.relu(self.worker_fc1(x))  # [T, B*A, worker_hidden_dim]
        # RNN forward
        out, (h, c) = self.worker_rnn(x, worker_hidden)  # out: [T, B*A, worker_hidden_dim]
        # generate U embedding
        U_t = self.U_embedding(out)  # [T, B*A, n_actions * embedding_dim_k]
        U_reshaped = U_t.view(T*B*A, self.args.n_actions, self.args.embedding_dim_k)
        # flatten goal
        goal_flat = goal.view(T*B*A, self.args.state_dim_d)
        w_t = self.w_network(goal_flat)  # [T*B*A, embedding_dim_k]
        w_t_reshaped = w_t.view(T*B*A, self.args.embedding_dim_k, 1)
        # compute Q values
        q = torch.bmm(U_reshaped, w_t_reshaped).squeeze(-1)  # [T*B*A, n_actions]
        q = q.view(T, B, A, self.args.n_actions)
        return q, (h, c)
        

class FeUdalCritic(nn.Module):
    def __init__(self, input_shape, args):
        super(FeUdalCritic, self).__init__()
        self.args = args

        # 狀態值估計 V_t^M
        # print("&&&&&&&&&&&&&&&&&&&&&&", args.obs_shape)
        self.value_network = nn.Linear(input_shape, 1)

    def forward(self, inputs):
        # 估計狀態值
        value = self.value_network(inputs)
        return value
    
    

