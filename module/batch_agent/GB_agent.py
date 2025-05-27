import torch.nn as nn
import torch.nn.functional as F
import torch

class my_Agent(nn.Module):
    def __init__(self, input_shape, args):
        super(my_Agent, self).__init__()
        self.args = args
        # manager
        self.manager_fc1 = nn.Linear(input_shape, args.hidden_dim)
        
        # 定義 RNN 層
        self.manager_rnn = nn.GRU(args.hidden_dim, args.hidden_dim, batch_first=False)
        
        # 定義輸出層
        self.manager_fc2 = nn.Linear(args.hidden_dim, args.goal_dim)

        # worker
        self.worker_fc1 = nn.Linear(input_shape + args.goal_dim, args.hidden_dim)
        
        # 定義 RNN 層
        self.worker_rnn = nn.GRU(args.hidden_dim, args.hidden_dim, batch_first=False)
        
        # 定義輸出層
        self.worker_fc2 = nn.Linear(args.hidden_dim, args.n_actions)

        self.manager_rnn.flatten_parameters()
        self.worker_rnn.flatten_parameters()

    def forward(self, input, hidden):
        # Flatten RNN parameters for optimized cudnn usage
        self.manager_rnn.flatten_parameters()
        self.worker_rnn.flatten_parameters()
        # input: [T, B, A, feat]
        T, B, A, feat = input.shape

        # assert A == self.args.n_agents and feat == self.args.input_size

        # 把 agent 维度 fold 进 batch
        x = input.view(T, B * A, feat)                # [T, B*A, feat]
        # manager
        m1 = F.relu(self.manager_fc1(x))              # [T, B*A, hidden]
        m_out, mh = self.manager_rnn(m1, hidden[0])   # m_out: [T, B*A, hidden]
        goal = self.manager_fc2(m_out)                # [T, B*A, goal_dim]

        # worker: concat feat + goal
        y = torch.cat([x, goal], dim=2)               # [T, B*A, feat+goal_dim]
        w1 = F.relu(self.worker_fc1(y))               # [T, B*A, hidden]
        w_out, wh = self.worker_rnn(w1, hidden[1])    # w_out: [T, B*A, hidden]
        action_flat = self.worker_fc2(w_out)          # [T, B*A, n_actions]

        # 把 batch*A 再拆回 agent 维度
        action = action_flat.view(T, B, A, -1)        # [T, B, A, n_actions]
        return action, (mh, wh)

    def init_hidden(self, batch_size: int = 1):
        B, A = batch_size, self.args.n_agents
        h_manager = torch.zeros(1,                      # num_layers
                                B * A,
                                self.args.hidden_dim,
                                device=next(self.parameters()).device)
        h_worker  = torch.zeros(1,
                                B * A,
                                self.args.hidden_dim,
                                device=next(self.parameters()).device)
        return h_manager, h_worker

