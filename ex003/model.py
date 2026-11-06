"""
ex003/model.py
論文 Ninomiya (NOLTA 2017) の再現実験3 (4.1節 Example 2: 5次元多峰性 Levy 関数近似問題，
隠れ層150ニューロン，パラメータ次元 dim(w) = 1,051) 向け高速モデル定義．

GPUのメモリ帯域ボトルネックを回避するため，TorchScript (@torch.jit.script) による要素演算融合と
ゼロ転置コピーの Flattened GEMM カーネルを実装している．
"""

import torch
import torch.nn as nn


class FeedforwardMLP5D(nn.Module):
    """Example 2 向け標準 PyTorch 3層ネットワーク (5 - 150 - 1, dim(w) = 1,051)．"""

    def __init__(self, n_in: int = 5, n_hid: int = 150, n_out: int = 1, dtype: torch.dtype = torch.float64):
        super().__init__()
        self.n_in = n_in
        self.n_hid = n_hid
        self.n_out = n_out
        self.fc1 = nn.Linear(n_in, n_hid, bias=True, dtype=dtype)
        self.fc2 = nn.Linear(n_hid, n_out, bias=True, dtype=dtype)

    @property
    def num_params(self) -> int:
        return self.n_in * self.n_hid + self.n_hid + self.n_hid * self.n_out + self.n_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.sigmoid(self.fc1(x)))


@torch.jit.script
def _jit_loss_only(
    X: torch.Tensor,
    D_col: torch.Tensor,
    W1: torch.Tensor,
    b1: torch.Tensor,
    W2: torch.Tensor,
    b2: torch.Tensor,
    R: int,
    H: int,
) -> torch.Tensor:
    N = X.shape[0]
    h = torch.sigmoid(torch.mm(X, W1) + b1).view(N, R, H)
    o = torch.sum(h * W2.unsqueeze(0), dim=-1) + b2
    diff = o - D_col
    return 0.5 * torch.mean(diff * diff, dim=0)


@torch.jit.script
def _jit_loss_and_grad(
    X: torch.Tensor,
    XT: torch.Tensor,
    D_col: torch.Tensor,
    W1: torch.Tensor,
    b1: torch.Tensor,
    W2: torch.Tensor,
    b2: torch.Tensor,
    R: int,
    H: int,
):
    N = X.shape[0]
    h = torch.sigmoid(torch.mm(X, W1) + b1).view(N, R, H)
    o = torch.sum(h * W2.unsqueeze(0), dim=-1) + b2
    diff = o - D_col
    loss = 0.5 * torch.mean(diff * diff, dim=0)

    d_o = diff * (1.0 / float(N))
    g_W2 = torch.sum(h * d_o.unsqueeze(-1), dim=0)
    g_b2 = torch.sum(d_o, dim=0).unsqueeze(-1)

    d_z1 = (d_o.unsqueeze(-1) * W2.unsqueeze(0)) * (h * (1.0 - h))
    g_b1 = torch.sum(d_z1, dim=0)
    g_W1_raw = torch.mm(XT, d_z1.view(N, R * H))
    g_W1 = g_W1_raw.view(-1, R, H).permute(1, 0, 2).reshape(R, -1)

    return loss, torch.cat([g_W1, g_b1, g_W2, g_b2], dim=1)


class BatchedMLP5D:
    """
    5入力・隠れ層150素子・1出力 (dim(w)=1,051) の R 試行バッチ並列モデル．
    訓練サンプル数 N=10,000 の大規模行列演算を TorchScript 融合カーネルで高速処理する．
    """

    def __init__(self, n_in: int = 5, n_hid: int = 150, n_out: int = 1):
        assert n_out == 1
        self.n_in = n_in
        self.n_hid = n_hid
        self.n_out = n_out
        self.i1 = n_in * n_hid      # 750
        self.i2 = self.i1 + n_hid   # 900
        self.i3 = self.i2 + n_hid   # 1050
        self.num_params = self.i3 + 1  # 1051

    def loss_only(self, w: torch.Tensor, X: torch.Tensor, D_col: torch.Tensor) -> torch.Tensor:
        R = w.shape[0]
        W1 = w[:, :self.i1].reshape(R, self.n_in, self.n_hid).permute(1, 0, 2).reshape(self.n_in, R * self.n_hid)
        b1 = w[:, self.i1:self.i2].reshape(1, R * self.n_hid)
        W2 = w[:, self.i2:self.i3].reshape(R, self.n_hid)
        b2 = w[:, self.i3:].reshape(1, R)
        return _jit_loss_only(X, D_col, W1, b1, W2, b2, R, self.n_hid)

    def loss_and_grad(self, w: torch.Tensor, X: torch.Tensor, XT: torch.Tensor, D_col: torch.Tensor):
        R = w.shape[0]
        W1 = w[:, :self.i1].reshape(R, self.n_in, self.n_hid).permute(1, 0, 2).reshape(self.n_in, R * self.n_hid)
        b1 = w[:, self.i1:self.i2].reshape(1, R * self.n_hid)
        W2 = w[:, self.i2:self.i3].reshape(R, self.n_hid)
        b2 = w[:, self.i3:].reshape(1, R)
        return _jit_loss_and_grad(X, XT, D_col, W1, b1, W2, b2, R, self.n_hid)
