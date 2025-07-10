"""
Center GAT + Shared DQN Head
================================
This file implements the graph‑based Q‑network described in your diagram.
* X_l  : (n_nodes, feat_dim)
* E_l  : (n_nodes, n_nodes, edge_dim)
Outputs:
* Q    : (n_nodes, n_actions)
The network stacks several **Gated GAT** layers followed by a shared MLP head
that maps each node embedding to its action‑value vector.
"""
from __future__ import annotations

from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

def _repeat(src: torch.Tensor, dim: int, n_repeat: int) -> torch.Tensor:
    """Utility: repeat along *new* axis for broadcasting (unsqueeze‑expand)."""
    return src.unsqueeze(dim).expand(*([-1] * dim), n_repeat, -1)


class GatedGATLayer(nn.Module):
    """One layer matching the architecture in the provided diagram.

    Args:
        in_dim:  input feature size  (H)
        out_dim: output feature size (H) – we keep in==out for residual ease.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.out_dim = out_dim

        # W3 : node -> edge key/value transform
        self.W3 = nn.Linear(in_dim, out_dim, bias=False)
        # W4 : edge transform
        self.W4 = nn.Linear(in_dim, out_dim, bias=False)
        # W5, W6 : node transforms for aggregation / residual update
        self.W5 = nn.Linear(in_dim, out_dim, bias=False)
        self.W6 = nn.Linear(in_dim, out_dim, bias=False)

        # layernorm for stability
        self.norm_nodes = nn.LayerNorm(out_dim)
        self.norm_edges = nn.LayerNorm(out_dim)

    def forward(
        self, X: torch.Tensor, E: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass

        Args:
            X: (B, n, H) or (n, H) – node features
            E: (B, n, n, H_e) or (n, n, H_e) – edge features (H_e == H)
        Returns:
            X_next, E_next with same shapes as inputs
        """
        # ensure 4‑D/3‑D shapes consistent
        batched = X.dim() == 3  # B,n,H
        if not batched:
            X = X.unsqueeze(0)
            E = E.unsqueeze(0)

        B, n, _ = X.size()
        H = self.out_dim

        # 1. Node -> edge projections
        Xi = self.W3(X)               # (B, n, H)
        Xj = Xi                       # identical transform for simplicity
        # unsqueeze for broadcasting
        Xi_exp = Xi.unsqueeze(2)      # (B, n,1,H)
        Xj_exp = Xj.unsqueeze(1)      # (B,1,n,H)

        # 2. Edge transform
        E_hat = self.W4(E)            # (B,n,n,H)

        # 3. Compute attention logits (Hadamard + sum)
        att_logits = (Xi_exp + Xj_exp + E_hat).sum(-1)   # (B,n,n)
        att_coeff  = F.softmax(att_logits, dim=-1).unsqueeze(-1)  # (B,n,n,1)

        # 4. Edge gating (Hadamard product)
        E_next = E_hat * att_coeff    # (B,n,n,H)

        # 5. Node aggregation
        #   Σ_j  (E_ij ⊙ W5(X_j))
        Xj_msg   = self.W5(X).unsqueeze(1).expand(B, n, n, H)  # (B,n,n,H)
        agg_msg  = (E_next * Xj_msg).sum(dim=2)                # (B,n,H)

        # 6. Residual node update
        X_res    = self.W6(X)
        X_next   = self.norm_nodes(F.relu(agg_msg + X_res)) + X  # residual

        # 7. Normalise edges
        E_next   = self.norm_edges(E_next)

        if not batched:
            X_next = X_next.squeeze(0)
            E_next = E_next.squeeze(0)
        return X_next, E_next


class CenterGAT_QNet(nn.Module):
    """Stacked Gated‑GAT layers followed by shared DQN head."""
    def __init__(self,
                 in_dim: int,
                 hid_dim: int,
                 n_layers: int,
                 n_actions: int,
                 edge_dim: int | None = None,   # ← 新增
                 dropout: float = 0.0):
        super().__init__()
        self.n_actions = n_actions

        # ──節點投影──
        self.in_proj = nn.Identity() if in_dim == hid_dim else nn.Linear(in_dim, hid_dim)

        # ──邊投影──
        edge_dim = edge_dim or in_dim          # 若沒給就跟 in_dim 一樣
        self.edge_proj = nn.Identity() if edge_dim == hid_dim else nn.Linear(edge_dim, hid_dim)
        #                ↑33→128 時就會是真的 Linear

        # GAT stack
        self.layers = nn.ModuleList(
            [GatedGATLayer(hid_dim, hid_dim) for _ in range(n_layers)]
        )
        self.dropout = nn.Dropout(dropout)

        # Shared head
        self.head = nn.Sequential(
            nn.Linear(hid_dim, hid_dim), nn.ReLU(),
            nn.Linear(hid_dim, n_actions),
        )

    def forward(self, X: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
        """Args:
            X: (n,H) or (B,n,H)
            E: (n,n,H) or (B,n,n,H)
        Returns:
            Q: (n,n_actions)  or (B,n,n_actions)
        """
        X = self.in_proj(X)     # node  33→128
        E = self.edge_proj(E)   # edge  33→128   ← 關鍵
        for layer in self.layers:
            X, E = layer(X, E)
            X = self.dropout(X)
        return self.head(X)



if __name__ == "__main__":
    # simple sanity check
    n, H = 3, 31
    X = torch.randn(n, H)
    E = torch.randn(n, n, H)
    net = CenterGAT_QNet(in_dim=H, hid_dim=H, n_layers=3, n_actions=5)
    Q = net(X, E)
    print(Q.shape)  # (n,5)
