"""
ex001/model.py
論文 "A novel quasi-Newton-based optimization for neural network training incorporating
Nesterov's accelerated gradient" (Ninomiya, NOLTA 2017) の Example 1 向けモデル定義．

1. FeedforwardMLP (torch.nn.Module):
   標準的なPyTorchモジュールとしての3層（隠れ層1層）フィードフォワードニューラルネットワーク．
   カスタム torch.optim.Optimizer (NAQ等) と組み合わせて使用する．
2. BatchedMLP:
   独立した R 回の試行を単一テンソル (R, D_w) としてGPU上で一括並列計算するための高速モデルクラス．
"""

import torch
import torch.nn as nn


class FeedforwardMLP(nn.Module):
    """隠れ層1層（シグモイド活性化）・線形出力層を持つフィードフォワードニューラルネットワーク．"""

    def __init__(self, n_in: int = 1, n_hid: int = 7, n_out: int = 1, dtype: torch.dtype = torch.float64):
        super().__init__()
        self.n_in = n_in
        self.n_hid = n_hid
        self.n_out = n_out
        self.dtype = dtype

        # 隠れ層と出力層の定義（全ニューロンにバイアス項を付与）
        self.fc1 = nn.Linear(n_in, n_hid, bias=True, dtype=dtype)
        self.fc2 = nn.Linear(n_hid, n_out, bias=True, dtype=dtype)

    @property
    def num_params(self) -> int:
        """全パラメータ数 dim(w) を返す．"""
        return self.n_in * self.n_hid + self.n_hid + self.n_hid * self.n_out + self.n_out

    def reset_parameters_uniform(self, low: float = -0.5, high: float = 0.5) -> None:
        """論文4節に従い，全重み・バイアスを一様乱数 [low, high] で初期化する．"""
        with torch.no_grad():
            for p in self.parameters():
                p.uniform_(low, high)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """順伝播計算: h = sig(W1 x + b1), o = W2 h + b2."""
        h = torch.sigmoid(self.fc1(x))
        o = self.fc2(h)
        return o

    def get_flat_params(self) -> torch.Tensor:
        """BatchedMLPと同一の並び順 [W1(n_in, n_hid), b1(n_hid), W2(n_hid, n_out), b2(n_out)] で1次元ベクトルを取得する．"""
        with torch.no_grad():
            w1 = self.fc1.weight.t().contiguous().view(-1)
            b1 = self.fc1.bias.view(-1)
            w2 = self.fc2.weight.t().contiguous().view(-1)
            b2 = self.fc2.bias.view(-1)
            return torch.cat([w1, b1, w2, b2], dim=0)

    def set_flat_params(self, flat_w: torch.Tensor) -> None:
        """1次元パラメータベクトル flat_w を各層の重み・バイアスへ書き戻す．"""
        i1 = self.n_in * self.n_hid
        i2 = i1 + self.n_hid
        i3 = i2 + self.n_hid * self.n_out
        with torch.no_grad():
            self.fc1.weight.copy_(flat_w[:i1].view(self.n_in, self.n_hid).t())
            self.fc1.bias.copy_(flat_w[i1:i2])
            self.fc2.weight.copy_(flat_w[i2:i3].view(self.n_hid, self.n_out).t())
            self.fc2.bias.copy_(flat_w[i3:])


def half_mse_loss(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    論文 式(1) の誤差関数 E(w) = (1 / |Tr|) * sum_{p in Tr} (1/2) * ||d_p - o_p||^2 を計算する．
    """
    diff = targets - outputs
    return 0.5 * torch.mean(torch.sum(diff * diff, dim=-1))


class BatchedMLP:
    """
    R 個の独立試行を GPU 上で一括並列評価・勾配計算するクラス．
    重み行列 w の形状は (R, D_w) であり，各行が1つのニューラルネットワークの全パラメータを表す．
    """

    def __init__(self, n_in: int = 1, n_hid: int = 7, n_out: int = 1):
        self.n_in = n_in
        self.n_hid = n_hid
        self.n_out = n_out
        self.i1 = n_in * n_hid
        self.i2 = self.i1 + n_hid
        self.i3 = self.i2 + n_hid * n_out
        self.num_params = self.i3 + n_out

    def predict(self, w: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """順伝播出力 o (R, N, n_out) を返す．"""
        R = w.shape[0]
        W1 = w[:, :self.i1].view(R, self.n_in, self.n_hid)
        b1 = w[:, self.i1:self.i2].view(R, 1, self.n_hid)
        W2 = w[:, self.i2:self.i3].view(R, self.n_hid, self.n_out)
        b2 = w[:, self.i3:].view(R, 1, self.n_out)

        h = torch.sigmoid(torch.matmul(X.unsqueeze(0), W1) + b1)
        o = torch.bmm(h, W2) + b2
        return o

    def loss_only(self, w: torch.Tensor, X: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
        """誤差関数 E(w) のみ (R,) を高速計算する（Armijo直線探索の候補評価用）．"""
        o = self.predict(w, X)
        diff = o - D.unsqueeze(0)
        return 0.5 * torch.mean(torch.sum(diff * diff, dim=-1), dim=-1)

    def loss_and_grad(self, w: torch.Tensor, X: torch.Tensor, D: torch.Tensor):
        """
        誤差関数 E(w) (R,) と勾配ベクトル grad E(w) (R, D_w) を解析的誤差逆伝播により一括計算する．
        autograd のグラフ構築オーバーヘッドを排除し，小規模モデルの反復計算を大幅に高速化する．
        """
        R = w.shape[0]
        N = X.shape[0]
        W1 = w[:, :self.i1].view(R, self.n_in, self.n_hid)
        b1 = w[:, self.i1:self.i2].view(R, 1, self.n_hid)
        W2 = w[:, self.i2:self.i3].view(R, self.n_hid, self.n_out)
        b2 = w[:, self.i3:].view(R, 1, self.n_out)

        h = torch.sigmoid(torch.matmul(X.unsqueeze(0), W1) + b1)  # (R, N, n_hid)
        o = torch.bmm(h, W2) + b2                                  # (R, N, n_out)
        diff = o - D.unsqueeze(0)                                  # o_p - d_p
        loss = 0.5 * torch.mean(torch.sum(diff * diff, dim=-1), dim=-1)

        # 逆伝播: dE/do = (o - d) / N
        d_o = diff / float(N)                                      # (R, N, n_out)
        g_W2 = torch.bmm(h.transpose(1, 2), d_o).reshape(R, self.n_hid * self.n_out)
        g_b2 = torch.sum(d_o, dim=1).reshape(R, self.n_out)

        d_h = torch.bmm(d_o, W2.transpose(1, 2))                   # (R, N, n_hid)
        d_z1 = d_h * h * (1.0 - h)                                 # sig'(z) = sig(z)(1 - sig(z))
        g_W1 = torch.matmul(X.t().unsqueeze(0), d_z1).reshape(R, self.n_in * self.n_hid)
        g_b1 = torch.sum(d_z1, dim=1).reshape(R, self.n_hid)

        grad = torch.cat([g_W1, g_b1, g_W2, g_b2], dim=1)
        return loss, grad
