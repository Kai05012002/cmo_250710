import torch.nn as nn
import torch.nn.functional as F
import torch
import time


class DQN_Agent(nn.Module):
    def __init__(self, input_shape, args):
        super(DQN_Agent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        
        # 定義 DQN 層
        self.dqn = nn.Linear(args.hidden_dim, args.hidden_dim)
        
        # 定義輸出層
        self.fc2 = nn.Linear(args.hidden_dim, args.n_actions)

    def forward(self, input):
        # start_time = time.perf_counter()
        
        # input: [T, B, A, feat]
        T, B, A, feat = input.shape
        # shape_time = time.perf_counter() - start_time
        # print(f"Shape extraction time: {shape_time:.6f} seconds")
        
        # fold agent dim into batch
        # start_op = time.perf_counter()
        x = input.view(T, B*A, feat)  # [T, B*A, feat]
        # fold_time = time.perf_counter() - start_op
        # print(f"Fold operation time: {fold_time:.6f} seconds")
        
        # feature transform
        # start_op = time.perf_counter()
        x = F.relu(self.fc1(x))  # [T, B*A, hidden_dim]
        # transform_time = time.perf_counter() - start_op
        # print(f"Feature transform time: {transform_time:.6f} seconds")
        
        # DQN forward
        # start_op = time.perf_counter()
        out = self.dqn(x)  # out: [T, B*A, hidden_dim]
        # dqn_time = time.perf_counter() - start_op
        # print(f"DQN forward time: {dqn_time:.6f} seconds")
        
        # output q-values
        # start_op = time.perf_counter()
        q = self.fc2(out)  # [T, B*A, n_actions]
        # fc2_time = time.perf_counter() - start_op
        # print(f"FC2 forward time: {fc2_time:.6f} seconds")
        
        # unfold batch*agent back to agent dim
        # start_op = time.perf_counter()
        q = q.view(T, B, A, self.args.n_actions)
        # unfold_time = time.perf_counter() - start_op
        # print(f"Unfold operation time: {unfold_time:.6f} seconds")
        
        # total_time = time.perf_counter() - start_time
        # print(f"Total forward pass time: {total_time:.6f} seconds\n")
        
        return q

