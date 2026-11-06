"""
ex002/model.py
論文 Ninomiya (NOLTA 2017) の再現実験2 (Example 1 におけるモーメンタム係数 mu の感度解析・
Table II および 終了条件なし収束曲線 Fig. 3 の検証) 向けモデル定義．
"""

import torch
import torch.nn as nn


class FeedforwardMLP(nn.Module):
    """1隠れ層（シグモイド活性化）・線形出力のフィードフォワードニューラルネットワーク．"""

    def __init__(self, n_in: int = 1, n_hid: int = 7, n_out: int = 1, dtype: torch.dtype = torch.float64):
        super().__init__()
        self.n_in = n_in
        self.n_hid = n_hid
        self.n_out = n_out
        self.fc1 = nn.Linear(n_in, n_hid, bias=True, dtype=dtype)
        self.fc2 = nn.Linear(n_hid, n_out, bias=True, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.sigmoid(self.fc1(x)))


class BatchedMLP:
    """
    複数条件・複数試行 (例えば 19種類の mu × 30試行 = 570試行) を単一バッチ次元 R として束ね，
    GPU上で一括順伝播・解析的逆伝播を実行する高速モデルクラス．
    """

    def __init__(self, n_in: int = 1, n_hid: int = 7, n_out: int = 1):
        self.n_in = n_in
        self.n_hid = n_hid
        self.n_out = n_out
        self.i1 = n_in * n_hid
        self.i2 = self.i1 + n_hid
        self.i3 = self.i2 + n_hid * n_out
        self.num_params = self.i3 + n_out

    def loss_only(self, w: torch.Tensor, X: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
        """誤差 E(w) (R,) を一括計算する．Flattened GEMM により大バッチ R でも高速に動作する．"""
        R = w.shape[0]
        N = X.shape[0]
        W1 = w[:, :self.i1].reshape(R, self.n_in, self.n_hid).permute(1, 0, 2).reshape(self.n_in, R * self.n_hid)
        b1 = w[:, self.i1:self.i2].reshape(1, R * self.n_hid)
        W2 = w[:, self.i2:self.i3].reshape(R, self.n_hid)  # n_out=1
        b2 = w[:, self.i3:].reshape(1, R)

        h = torch.sigmoid(torch.mm(X, W1) + b1).view(N, R, self.n_hid)
        o = torch.sum(h * W2.unsqueeze(0), dim=-1) + b2    # (N, R)
        diff = o - D                                       # D: (N, 1) -> (N, R)
        return 0.5 * torch.mean(diff * diff, dim=0)

    def loss_and_grad(self, w: torch.Tensor, X: torch.Tensor, XT: torch.Tensor, D: torch.Tensor):
        """誤差 E(w) (R,) と勾配ベクトル grad E(w) (R, D_w) をメモリコピー最小のFlattened GEMMで計算する．"""
        R = w.shape[0]
        N = X.shape[0]
        W1 = w[:, :self.i1].reshape(R, self.n_in, self.n_hid).permute(1, 0, 2).reshape(self.n_in, R * self.n_hid)
        b1 = w[:, self.i1:self.i2].reshape(1, R * self.n_hid)
        W2 = w[:, self.i2:self.i3].reshape(R, self.n_hid)
        b2 = w[:, self.i3:].reshape(1, R)

        h = torch.sigmoid(torch.mm(X, W1) + b1).view(N, R, self.n_hid)
        o = torch.sum(h * W2.unsqueeze(0), dim=-1) + b2
        diff = o - D
        loss = 0.5 * torch.mean(diff * diff, dim=0)

        d_o = diff * (1.0 / float(N))                      # (N, R)
        g_W2 = torch.sum(h * d_o.unsqueeze(-1), dim=0)     # (R, n_hid)
        g_b2 = torch.sum(d_o, dim=0).unsqueeze(-1)         # (R, 1)

        d_z1 = (d_o.unsqueeze(-1) * W2.unsqueeze(0)) * (h * (1.0 - h))  # (N, R, n_hid)
        g_b1 = torch.sum(d_z1, dim=0)                      # (R, n_hid)
        g_W1_raw = torch.mm(XT, d_z1.view(N, R * self.n_hid))
        g_W1 = g_W1_raw.view(self.n_in, R, self.n_hid).permute(1, 0, 2).reshape(R, self.i1)

        grad = torch.cat([g_W1, g_b1, g_W2, g_b2], dim=1)
        return loss, grad
