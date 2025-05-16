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

    def forward(self, input, hidden):
        # manager
        # input: [T, B=1, feat]
        x = F.relu(self.manager_fc1(input))
        m_out, mh = self.manager_rnn(x, hidden[0])  # mh: [T,1,hidden]
        goal = self.manager_fc2(m_out)                          # [T,1,goal_dim]
        
        # worker
        # 將 x 和 goal 串接起來，而不是相加
        y = torch.cat([input, goal], dim=2)  # 在特徵維度上串接
        y = F.relu(self.worker_fc1(y))
        
        w_out, wh = self.worker_rnn(y, hidden[1])  # wh: [T,1,hidden]
        action = self.worker_fc2(w_out)                        # [T,1,n_actions]
        return action, (mh, wh)

    def init_hidden(self, batch_size: int = 1):
        h_manager = torch.zeros(1,                      # num_layers
                                batch_size,
                                self.args.hidden_dim,
                                device=next(self.parameters()).device)
        h_worker  = torch.zeros(1,
                                batch_size,
                                self.args.hidden_dim,
                                device=next(self.parameters()).device)
        return h_manager, h_worker

