import torch.nn as nn
import torch.nn.functional as F
import torch


class RNN_Agent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNN_Agent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        
        # 定義 RNN 層
        self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        
        # 定義輸出層
        self.fc2 = nn.Linear(args.hidden_dim, args.n_actions)

    def forward(self, x, hidden):
        # RNN 前向傳播
        x = F.relu(self.fc1(x))
        h_in = hidden.reshape(-1, self.args.hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h

    def init_hidden(self):
        # 初始化隱藏狀態
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()

