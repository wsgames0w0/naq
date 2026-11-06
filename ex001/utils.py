"""
ex001/utils.py
論文 Ninomiya (NOLTA 2017) の再現実験1 (Example 1) に使用する各種関数・Optimizer定義．

含まれる機能:
1. データセット生成・読み込み・[-1, 1]正規化関数 (load_or_create_ex1_dataset)
2. PyTorch標準の torch.optim.Optimizer を継承した汎用 NAQ / QN クラス (NAQOptimizer)
3. 30試行一括GPU並列学習エンジン: SG, NAG, Qprop, QN, NAQ (train_batched_ex1)
4. 実験結果集計・可視化関数
"""

import os
import time
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.optim import Optimizer
from tqdm import tqdm

from model import BatchedMLP, FeedforwardMLP, half_mse_loss


def setup_japanese_matplotlib() -> None:
    """Matplotlibで日本語フォントとマイナス記号が正しく描画されるように設定する．"""
    plt.rcParams["font.family"] = ["Meiryo", "Yu Gothic", "MS Gothic", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False


def target_function_f1(x: np.ndarray) -> np.ndarray:
    """論文 式(18): f1(x) = 1 + (x + 2x^2) sin(-x^2), |x| <= 4."""
    return 1.0 + (x + 2.0 * (x ** 2)) * np.sin(-(x ** 2))


def load_or_create_ex1_dataset(
    dataset_dir: str = "../datasets",
    seed: int = 2017,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
) -> Dict[str, object]:
    """
    Example 1 の訓練データ(400点: 0.02刻み)とテストデータ(10,000点: 一様乱数)を生成または読み込み，
    論文4節に従い入出力の各要素を [-1.0, 1.0] に正規化して返す．
    """
    os.makedirs(dataset_dir, exist_ok=True)
    npz_path = os.path.join(dataset_dir, "ex1_dataset.npz")

    if os.path.exists(npz_path):
        data = np.load(npz_path)
        x_tr_raw = data["x_tr_raw"]
        y_tr_raw = data["y_tr_raw"]
        x_te_raw = data["x_te_raw"]
        y_te_raw = data["y_te_raw"]
    else:
        rng = np.random.default_rng(seed)
        # 訓練データ Tr: [-4, 4) を 0.02 間隔でサンプリングした 400 点
        x_tr_raw = np.arange(-4.0, 4.0, 0.02, dtype=np.float64)
        y_tr_raw = target_function_f1(x_tr_raw)

        # テストデータ Te: [-4, 4) から独立に一様サンプリングした 10,000 点
        x_te_raw = rng.uniform(-4.0, 4.0, size=10000).astype(np.float64)
        x_te_raw.sort()
        y_te_raw = target_function_f1(x_te_raw)

        np.savez(
            npz_path,
            x_tr_raw=x_tr_raw,
            y_tr_raw=y_tr_raw,
            x_te_raw=x_te_raw,
            y_te_raw=y_te_raw,
        )

    # [-1.0, 1.0] への線形正規化
    x_min, x_max = -4.0, 4.0
    y_min = min(float(y_tr_raw.min()), float(y_te_raw.min()))
    y_max = max(float(y_tr_raw.max()), float(y_te_raw.max()))

    x_tr_norm = 2.0 * (x_tr_raw - x_min) / (x_max - x_min) - 1.0
    y_tr_norm = 2.0 * (y_tr_raw - y_min) / (y_max - y_min) - 1.0
    x_te_norm = 2.0 * (x_te_raw - x_min) / (x_max - x_min) - 1.0
    y_te_norm = 2.0 * (y_te_raw - y_min) / (y_max - y_min) - 1.0

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return {
        "X_train": torch.tensor(x_tr_norm[:, None], device=device, dtype=dtype),
        "D_train": torch.tensor(y_tr_norm[:, None], device=device, dtype=dtype),
        "X_test": torch.tensor(x_te_norm[:, None], device=device, dtype=dtype),
        "D_test": torch.tensor(y_te_norm[:, None], device=device, dtype=dtype),
        "x_tr_raw": x_tr_raw,
        "y_tr_raw": y_tr_raw,
        "x_te_raw": x_te_raw,
        "y_te_raw": y_te_raw,
        "norm_stats": {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max},
    }


class NAQOptimizer(Optimizer):
    """
    論文 Algorithm 2: Nesterov's Accelerated Quasi-Newton (NAQ) 法の汎用 PyTorch Optimizer 実装．
    mu = 0.0 と設定した場合は従来の準ニュートン法 (Algorithm 1: QN / BFGS) と完全に一致する．

    Parameters
    ----------
    params : iterable
        最適化対象の torch.nn.Parameter イテラブル
    mu : float, default=0.9
        モーメンタム係数 mu in [0, 1)
    xi : float, default=1e-3
        Armijo条件の係数 xi (論文4節: 1e-3)
    tau : float, default=0.5
        Armijoバックトラッキングのステップ縮小率
    max_step_norm : float, default=1.0
        1回あたりの探索方向ベクトルの最大ノルム上限（シグモイド飽和防止のための初期試行幅調整）
    curvature_eps : float, default=1e-14
        曲率条件 p_k^T q_k > 0 の数値的判定閾値
    """

    def __init__(
        self,
        params,
        mu: float = 0.9,
        xi: float = 1e-3,
        tau: float = 0.5,
        max_step_norm: float = 1.0,
        curvature_eps: float = 1e-14,
    ):
        if not 0.0 <= mu < 1.0:
            raise ValueError(f"Invalid momentum parameter mu: {mu}")
        defaults = dict(
            mu=mu,
            xi=xi,
            tau=tau,
            max_step_norm=max_step_norm,
            curvature_eps=curvature_eps,
        )
        super().__init__(params, defaults)
        self._params_list: List[torch.nn.Parameter] = []
        for group in self.param_groups:
            self._params_list.extend(group["params"])

    def _gather_flat_params(self) -> torch.Tensor:
        return torch.cat([p.data.view(-1) for p in self._params_list], dim=0)

    def _set_flat_params(self, flat_w: torch.Tensor) -> None:
        offset = 0
        for p in self._params_list:
            numel = p.numel()
            p.data.copy_(flat_w[offset : offset + numel].view_as(p.data))
            offset += numel

    def _gather_flat_grad(self) -> torch.Tensor:
        grads = []
        for p in self._params_list:
            if p.grad is None:
                grads.append(torch.zeros_like(p.data).view(-1))
            else:
                grads.append(p.grad.data.view(-1))
        return torch.cat(grads, dim=0)

    @torch.no_grad()
    def step(self, closure: Callable[[bool], torch.Tensor]) -> Tuple[float, float, float]:
        """
        NAQ の1反復 (Algorithm 2 の 5〜10行目) を実行する．

        Parameters
        ----------
        closure : Callable[[bool], torch.Tensor]
            closure(backward=True) のとき順伝播＋逆伝播を行い損失テンソルを返し，
            closure(backward=False) のとき順伝播のみを行い損失テンソルを返す関数．

        Returns
        -------
        loss_next : float
            更新後の誤差 E(w_{k+1})
        grad_norm_next : float
            更新後の勾配ノルム ||grad E(w_{k+1})||
        alpha : float
            Armijo直線探索で採択されたステップ幅 alpha_k
        """
        group = self.param_groups[0]
        mu: float = group["mu"]
        xi: float = group["xi"]
        tau: float = group["tau"]
        max_step_norm: float = group["max_step_norm"]
        curv_eps: float = group["curvature_eps"]

        w_k = self._gather_flat_params()
        dim_w = w_k.numel()
        state = self.state["global_naq_state"]

        # 初回ステップの初期化: H_1 = I, v_1 = 0
        if len(state) == 0:
            state["step"] = 0
            state["v"] = torch.zeros_like(w_k)
            state["H"] = torch.eye(dim_w, device=w_k.device, dtype=w_k.dtype)
            with torch.enable_grad():
                loss_init = closure(True)
            state["loss_k"] = float(loss_init.item())
            state["grad_k"] = self._gather_flat_grad().clone()

        v_k: torch.Tensor = state["v"]
        H_k: torch.Tensor = state["H"]

        # Step 5: Nesterov先読み点 w_k + mu * v_k における勾配計算 (1回目の勾配評価)
        if mu == 0.0:
            w_nes = w_k
            loss_nes = state["loss_k"]
            grad_nes = state["grad_k"]
        else:
            w_nes = w_k + mu * v_k
            self._set_flat_params(w_nes)
            with torch.enable_grad():
                loss_nes_tensor = closure(True)
            loss_nes = float(loss_nes_tensor.item())
            grad_nes = self._gather_flat_grad().clone()

        # 探索方向 g_k = -H_k * grad E(w_k + mu * v_k) の計算
        g_raw = -torch.mv(H_k, grad_nes)
        gn = float(torch.norm(g_raw).item())
        scale = max(1.0, gn / max_step_norm)
        g_k = g_raw / scale

        # 降下方向条件 grad_nes^T g_k < 0 の確認（非降下時は H_k = I にリセット）
        slope = float(torch.dot(grad_nes, g_k).item())
        if slope >= 0.0:
            H_k.copy_(torch.eye(dim_w, device=w_k.device, dtype=w_k.dtype))
            g_raw = -grad_nes
            scale = max(1.0, float(torch.norm(g_raw).item()) / max_step_norm)
            g_k = g_raw / scale
            slope = float(torch.dot(grad_nes, g_k).item())

        # Step 6: Armijo条件 式(17) によるバックトラッキング直線探索
        alpha = 0.0
        for m in range(25):
            cand_alpha = tau ** m
            self._set_flat_params(w_nes + cand_alpha * g_k)
            loss_try = float(closure(False).item())
            if loss_try <= loss_nes + xi * cand_alpha * slope:
                alpha = cand_alpha
                break

        # Step 7 & 8: 更新ベクトル v_{k+1} と重み w_{k+1} の更新 (式7, 式2)
        v_next = mu * v_k + alpha * g_k
        w_next = w_k + v_next
        self._set_flat_params(w_next)

        # Step 9: 更新後の重み w_{k+1} における通常勾配計算 (2回目の勾配評価)
        with torch.enable_grad():
            loss_next_tensor = closure(True)
        loss_next = float(loss_next_tensor.item())
        grad_next = self._gather_flat_grad().clone()

        # Step 10: セカントベクトル p_k, q_k と 逆ヘッセ近似行列 H_{k+1} の更新 (式8)
        p_k = alpha * g_k  # = w_{k+1} - (w_k + mu * v_k)
        q_k = grad_next - grad_nes
        pq = float(torch.dot(p_k, q_k).item())

        if pq > curv_eps:
            Hq = torch.mv(H_k, q_k)
            qHq = float(torch.dot(q_k, Hq).item())
            term1 = (torch.outer(Hq, p_k) + torch.outer(p_k, Hq)) / pq
            term2 = (1.0 + qHq / pq) * torch.outer(p_k, p_k) / pq
            H_k.add_(-term1 + term2)
        else:
            H_k.copy_(torch.eye(dim_w, device=w_k.device, dtype=w_k.dtype))

        state["step"] += 1
        state["v"] = v_next
        state["loss_k"] = loss_next
        state["grad_k"] = grad_next

        return loss_next, float(torch.norm(grad_next).item()), alpha


def generate_initial_weights(
    num_runs: int, num_params: int, seed: int = 42, device: Optional[torch.device] = None
) -> torch.Tensor:
    """R 試行分の初期重み行列 (R, D_w) を一様乱数 [-0.5, 0.5] で生成する．"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    w_cpu = torch.rand((num_runs, num_params), generator=gen, dtype=torch.float64) - 0.5
    return w_cpu.to(device)


def train_batched_ex1(
    algo: str,
    stepsize_mode: str,
    w_init: torch.Tensor,
    X_train: torch.Tensor,
    D_train: torch.Tensor,
    X_test: torch.Tensor,
    D_test: torch.Tensor,
    n_in: int = 1,
    n_hid: int = 7,
    n_out: int = 1,
    mu: float = 0.0,
    fixed_lr: float = 0.01,
    xi: float = 1e-3,
    tau: float = 0.5,
    max_step_norm: float = 1.0,
    eps: float = 1e-6,
    max_epochs: int = 100000,
    show_pbar: bool = True,
    desc: Optional[str] = None,
) -> Dict[str, object]:
    """
    Example 1 向けに R 回の独立試行を GPU 上で一括並列学習する汎用ソルバ．

    Parameters
    ----------
    algo : {'SG', 'NAG', 'Qprop', 'QN', 'NAQ'}
    stepsize_mode : {'fixed', 'Armijo'}
    """
    device = X_train.device
    dtype = X_train.dtype
    model = BatchedMLP(n_in=n_in, n_hid=n_hid, n_out=n_out)
    R, Dw = w_init.shape

    w_final = w_init.clone()
    loss_final = torch.zeros(R, device=device, dtype=dtype)
    epochs_final = torch.zeros(R, device=device, dtype=torch.long)

    orig_idx = torch.arange(R, device=device)
    w = w_init.clone()
    v = torch.zeros_like(w)

    is_qn = algo in ("QN", "NAQ")
    is_nesterov = algo in ("NAG", "NAQ")
    use_mu = mu if is_nesterov else 0.0

    I_d = torch.eye(Dw, device=device, dtype=dtype)
    H = I_d.unsqueeze(0).expand(R, Dw, Dw).clone() if is_qn else None
    # Qprop用の前回差分 s_{k-1}, y_{k-1}
    s_prev = torch.zeros_like(w) if algo == "Qprop" else None
    y_prev = torch.zeros_like(w) if algo == "Qprop" else None

    K_armijo = 25
    alphas_grid = tau ** torch.arange(K_armijo, device=device, dtype=dtype)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    loss_k, grad_k = model.loss_and_grad(w, X_train, D_train)

    label = desc or f"{algo} ({stepsize_mode}, mu={mu if is_nesterov else '-'})"
    pbar = tqdm(total=max_epochs, desc=label, disable=not show_pbar, mininterval=1.0, leave=True)

    check_interval = 250
    last_pbar_k = 0

    for k in range(1, max_epochs + 1):
        # 一定間隔ごとに収束済み試行をバッチから圧縮除外してGPU演算量を削減
        if k % check_interval == 1:
            alive = torch.norm(grad_k, dim=1) > eps
            done = ~alive
            if done.any():
                d_idx = orig_idx[done]
                w_final[d_idx] = w[done]
                loss_final[d_idx] = loss_k[done]
                epochs_final[d_idx] = k - 1
                if not alive.any():
                    pbar.update(k - 1 - last_pbar_k)
                    last_pbar_k = k - 1
                    break
                orig_idx = orig_idx[alive]
                w = w[alive]
                v = v[alive]
                loss_k = loss_k[alive]
                grad_k = grad_k[alive]
                if H is not None:
                    H = H[alive]
                if s_prev is not None:
                    s_prev = s_prev[alive]
                    y_prev = y_prev[alive]

            if show_pbar and (k - last_pbar_k >= 2500):
                pbar.update(k - last_pbar_k)
                pbar.set_postfix(active=int(w.shape[0]), E_ave=f"{loss_k.mean().item()*1e3:.3f}")
                last_pbar_k = k

        Ra = w.shape[0]
        still_active = torch.norm(grad_k, dim=1) > eps

        # 1. 勾配評価位置の決定 (通常点 w_k または Nesterov先読み点 w_k + mu * v_k)
        if is_nesterov and use_mu > 0.0:
            w_base = w + use_mu * v
            loss_base, grad_base = model.loss_and_grad(w_base, X_train, D_train)
        else:
            w_base, loss_base, grad_base = w, loss_k, grad_k

        # 2. 各アルゴリズムの探索方向ベクトル g_k の計算
        if algo in ("SG", "NAG"):
            g_raw = -grad_base
        elif algo == "Qprop":
            if k == 1:
                lam = torch.ones_like(grad_base)
            else:
                # 対角近似 lambda_{ii,k} = s_{i,k-1} / y_{i,k-1} (Vrahatis et al., 2000 の正定値化)
                sy_pos = (s_prev * y_prev) > 1e-16
                ratio = torch.where(sy_pos, s_prev / torch.where(sy_pos, y_prev, torch.ones_like(y_prev)), torch.ones_like(s_prev))
                lam = torch.clamp(ratio, min=1e-4, max=1e4)
            g_raw = -lam * grad_base
        elif is_qn:
            I_a = I_d.unsqueeze(0).expand(Ra, Dw, Dw)
            g_raw = -torch.bmm(H, grad_base.unsqueeze(-1)).squeeze(-1)

        # ステップノルム上限によるスケーリング（Armijo初期試行幅調整と数学的に等価）
        if stepsize_mode == "Armijo":
            gn = torch.norm(g_raw, dim=1, keepdim=True)
            scale = torch.clamp(gn / max_step_norm, min=1.0)
            g_k = g_raw / scale
            slope = torch.sum(grad_base * g_k, dim=1)

            if is_qn:
                bad = slope >= 0.0
                H = torch.where(bad[:, None, None], I_a, H)
                g_fb = -grad_base / torch.clamp(torch.norm(grad_base, dim=1, keepdim=True) / max_step_norm, min=1.0)
                g_k = torch.where(bad[:, None], g_fb, g_k)
                slope = torch.sum(grad_base * g_k, dim=1)

            # 並列Armijo直線探索: K_armijo 個の候補 alpha を一括評価
            w_cand = w_base.unsqueeze(1) + alphas_grid[None, :, None] * g_k.unsqueeze(1)
            losses_cand = model.loss_only(w_cand.reshape(Ra * K_armijo, Dw), X_train, D_train).view(Ra, K_armijo)
            rhs = loss_base.unsqueeze(1) + xi * alphas_grid.unsqueeze(0) * slope.unsqueeze(1)
            ok = losses_cand <= rhs
            any_ok = ok.any(dim=1)
            first_ok = torch.argmax(ok.to(torch.int64), dim=1)
            alpha = torch.where(any_ok, alphas_grid[first_ok], torch.zeros_like(loss_base))
        else:
            g_k = g_raw
            alpha = torch.full((Ra,), fixed_lr, device=device, dtype=dtype)

        # 3. パラメータ更新
        if is_nesterov:
            v_next = use_mu * v + alpha[:, None] * g_k
            w_next = w + v_next
        else:
            v_next = alpha[:, None] * g_k
            w_next = w + v_next

        loss_next, grad_next = model.loss_and_grad(w_next, X_train, D_train)

        # 4. 準ニュートン行列 / Qprop履歴の更新
        if is_qn:
            p_k = alpha[:, None] * g_k  # w_{k+1} - w_base
            q_k = grad_next - grad_base
            pq = torch.sum(p_k * q_k, dim=1)
            Hq = torch.bmm(H, q_k.unsqueeze(-1)).squeeze(-1)
            qHq = torch.sum(q_k * Hq, dim=1)
            pq_safe = torch.where(pq > 1e-14, pq, torch.ones_like(pq))
            term1 = (Hq.unsqueeze(2) * p_k.unsqueeze(1) + p_k.unsqueeze(2) * Hq.unsqueeze(1)) / pq_safe[:, None, None]
            term2 = (1.0 + qHq / pq_safe)[:, None, None] * (p_k.unsqueeze(2) * p_k.unsqueeze(1)) / pq_safe[:, None, None]
            H_new = torch.where((pq > 1e-14)[:, None, None], H - term1 + term2, I_a)
            H = torch.where(still_active[:, None, None], H_new, H)
        elif algo == "Qprop":
            s_new = w_next - w
            y_new = grad_next - grad_k
            s_prev = torch.where(still_active[:, None], s_new, s_prev)
            y_prev = torch.where(still_active[:, None], y_new, y_prev)

        w = torch.where(still_active[:, None], w_next, w)
        v = torch.where(still_active[:, None], v_next, v)
        loss_k = torch.where(still_active, loss_next, loss_k)
        grad_k = torch.where(still_active[:, None], grad_next, grad_k)

    # 未収束で max_epochs に到達した残りの試行を格納
    if orig_idx.numel() > 0:
        still = torch.norm(grad_k, dim=1) > eps
        ep_rem = torch.where(still, torch.full_like(orig_idx, max_epochs), torch.full_like(orig_idx, k))
        w_final[orig_idx] = w
        loss_final[orig_idx] = loss_k
        epochs_final[orig_idx] = ep_rem
        if show_pbar and last_pbar_k < max_epochs:
            pbar.update(max_epochs - last_pbar_k)

    pbar.close()
    if device.type == "cuda":
        torch.cuda.synchronize()
    total_time = time.perf_counter() - t_start

    # テスト誤差 E_test(w) の一括計算
    with torch.no_grad():
        loss_test = model.loss_only(w_final, X_test, D_test)

    e_tr = (loss_final * 1000.0).cpu().numpy()
    e_te = (loss_test * 1000.0).cpu().numpy()
    ep_np = epochs_final.cpu().numpy()

    return {
        "algo": algo,
        "stepsize": stepsize_mode,
        "mu": "-" if not is_nesterov else f"{mu:.2f}",
        "train_ave": float(e_tr.mean()),
        "train_best": float(e_tr.min()),
        "train_worst": float(e_tr.max()),
        "test_ave": float(e_te.mean()),
        "test_best": float(e_te.min()),
        "test_worst": float(e_te.max()),
        "ave_epochs": float(ep_np.mean()),
        "ave_time_sec": float(total_time / R),
        "batch_time_sec": float(total_time),
        "train_errors_all": e_tr,
        "test_errors_all": e_te,
        "epochs_all": ep_np,
        "best_weights": w_final[int(np.argmin(e_tr))].detach().clone(),
    }


def format_summary_dataframe(results: List[Dict[str, object]]) -> pd.DataFrame:
    """論文 Table I と同じ形式の pandas.DataFrame に整形する．"""
    rows = []
    for r in results:
        rows.append({
            "Algorithm": r["algo"],
            "Stepsize": r["stepsize"],
            "mu": r["mu"],
            "E_train (x10^3) Ave/Best/Worst": f"{r['train_ave']:.3f} / {r['train_best']:.3f} / {r['train_worst']:.3f}",
            "Ave. T(sec)/run": round(r["ave_time_sec"], 4),
            "Batch GPU T(sec)": round(r["batch_time_sec"], 2),
            "Ave. epochs": int(round(r["ave_epochs"])),
            "E_test (x10^3) Ave/Best/Worst": f"{r['test_ave']:.3f} / {r['test_best']:.3f} / {r['test_worst']:.3f}",
        })
    return pd.DataFrame(rows)
