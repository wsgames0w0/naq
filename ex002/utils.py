"""
ex002/utils.py
論文 Ninomiya (NOLTA 2017) の再現実験2 (Table II: モーメンタム係数 mu の感度解析，
および Fig. 3: 終了条件なし収束曲線比較) に使用する関数群．
"""

import os
import time
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from model import BatchedMLP


def setup_japanese_matplotlib() -> None:
    """Matplotlibの日本語表示設定を行う．"""
    plt.rcParams["font.family"] = ["Meiryo", "Yu Gothic", "MS Gothic", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False


def load_ex1_dataset(
    dataset_dir: str = "../datasets",
    seed: int = 2017,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
) -> Dict[str, torch.Tensor]:
    """datasets/ex1_dataset.npz から Example 1 の正規化済みデータを読み込む（未生成時は自動生成）．"""
    os.makedirs(dataset_dir, exist_ok=True)
    npz_path = os.path.join(dataset_dir, "ex1_dataset.npz")
    if not os.path.exists(npz_path):
        rng = np.random.default_rng(seed)
        x_tr_raw = np.arange(-4.0, 4.0, 0.02, dtype=np.float64)
        y_tr_raw = 1.0 + (x_tr_raw + 2.0 * x_tr_raw ** 2) * np.sin(-(x_tr_raw ** 2))
        x_te_raw = np.sort(rng.uniform(-4.0, 4.0, size=10000).astype(np.float64))
        y_te_raw = 1.0 + (x_te_raw + 2.0 * x_te_raw ** 2) * np.sin(-(x_te_raw ** 2))
        np.savez(npz_path, x_tr_raw=x_tr_raw, y_tr_raw=y_tr_raw, x_te_raw=x_te_raw, y_te_raw=y_te_raw)
    else:
        d = np.load(npz_path)
        x_tr_raw, y_tr_raw = d["x_tr_raw"], d["y_tr_raw"]
        x_te_raw, y_te_raw = d["x_te_raw"], d["y_te_raw"]

    x_min, x_max = -4.0, 4.0
    y_min = min(float(y_tr_raw.min()), float(y_te_raw.min()))
    y_max = max(float(y_tr_raw.max()), float(y_te_raw.max()))

    x_tr_n = 2.0 * (x_tr_raw - x_min) / (x_max - x_min) - 1.0
    y_tr_n = 2.0 * (y_tr_raw - y_min) / (y_max - y_min) - 1.0
    x_te_n = 2.0 * (x_te_raw - x_min) / (x_max - x_min) - 1.0
    y_te_n = 2.0 * (y_te_raw - y_min) / (y_max - y_min) - 1.0

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_tr = torch.tensor(x_tr_n[:, None], device=device, dtype=dtype)
    return {
        "X_train": X_tr,
        "XT_train": X_tr.t().contiguous(),
        "D_train": torch.tensor(y_tr_n[:, None], device=device, dtype=dtype),
        "X_test": torch.tensor(x_te_n[:, None], device=device, dtype=dtype),
        "D_test": torch.tensor(y_te_n[:, None], device=device, dtype=dtype),
    }


def generate_initial_weights(num_runs: int, num_params: int = 22, seed: int = 2017, device: Optional[torch.device] = None) -> torch.Tensor:
    """[-0.5, 0.5] の一様乱数で R 試行の初期重みを生成する．"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    return (torch.rand((num_runs, num_params), generator=gen, dtype=torch.float64) - 0.5).to(device)


def staged_parallel_armijo(
    model: BatchedMLP,
    w_base: torch.Tensor,
    g_k: torch.Tensor,
    loss_base: torch.Tensor,
    slope: torch.Tensor,
    X: torch.Tensor,
    D: torch.Tensor,
    xi: float = 1e-3,
    tau: float = 0.5,
) -> torch.Tensor:
    """
    2段階並列Armijo直線探索:
    まず alpha=1.0 を全試行で評価し，棄却された試行のみ残りの候補 (tau^1 .. tau^24) を並列評価する．
    これによりVRAM消費量と演算量を大幅に節約する．
    """
    Ra, Dw = w_base.shape
    alpha = torch.ones(Ra, device=w_base.device, dtype=w_base.dtype)
    l0 = model.loss_only(w_base + g_k, X, D)
    ok0 = l0 <= loss_base + xi * slope
    if ok0.all():
        return alpha

    rem = torch.nonzero(~ok0).squeeze(-1)
    cands = tau ** torch.arange(1, 25, device=w_base.device, dtype=w_base.dtype)
    K = cands.numel()
    wc = w_base[rem].unsqueeze(1) + cands[None, :, None] * g_k[rem].unsqueeze(1)
    lc = model.loss_only(wc.reshape(rem.numel() * K, Dw), X, D).view(rem.numel(), K)
    ok_c = lc <= loss_base[rem].unsqueeze(1) + xi * cands.unsqueeze(0) * slope[rem].unsqueeze(1)
    any_c = ok_c.any(dim=1)
    first_idx = torch.argmax(ok_c.to(torch.int64), dim=1)
    alpha[rem] = torch.where(any_c, cands[first_idx], torch.zeros_like(loss_base[rem]))
    return alpha


def run_mu_sweep_batched(
    algo: str,
    mu_list: List[float],
    w_init_single_set: torch.Tensor,
    X_train: torch.Tensor,
    XT_train: torch.Tensor,
    D_train: torch.Tensor,
    X_test: torch.Tensor,
    D_test: torch.Tensor,
    eps: float = 1e-6,
    max_epochs: int = 100000,
    max_step_norm: float = 1.0,
) -> List[Dict[str, object]]:
    """
    論文 Table II 用: mu_list (例: 19通り) × R (30試行) の全試行を1つのバッチに統合して
    GPU上で一括並列実行し，各 mu ごとの統計量を返す．
    """
    assert algo in ("NAQ", "NAG")
    is_qn = algo == "NAQ"
    device = X_train.device
    dtype = X_train.dtype
    model = BatchedMLP(1, 7, 1)

    M = len(mu_list)
    R0, Dw = w_init_single_set.shape
    total_R = M * R0

    # (M, R0, Dw) -> (total_R, Dw)
    w_all_init = w_init_single_set.unsqueeze(0).expand(M, R0, Dw).reshape(total_R, Dw).clone()
    mu_per_run = torch.tensor(mu_list, device=device, dtype=dtype).repeat_interleave(R0)  # (total_R,)

    w_final = w_all_init.clone()
    loss_final = torch.zeros(total_R, device=device, dtype=dtype)
    epochs_final = torch.zeros(total_R, device=device, dtype=torch.long)

    orig_idx = torch.arange(total_R, device=device)
    w = w_all_init.clone()
    v = torch.zeros_like(w)
    mu_vec = mu_per_run.clone()

    I_d = torch.eye(Dw, device=device, dtype=dtype)
    H = I_d.unsqueeze(0).expand(total_R, Dw, Dw).clone() if is_qn else None

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    loss_k, grad_k = model.loss_and_grad(w, X_train, XT_train, D_train)
    pbar = tqdm(total=max_epochs, desc=f"{algo} mu-sweep ({M}x{R0}={total_R} runs)", mininterval=1.0)
    last_k = 0
    total_run_steps_executed = 0

    for k in range(1, max_epochs + 1):
        if k % 250 == 1:
            alive = torch.norm(grad_k, dim=1) > eps
            done = ~alive
            if done.any():
                d_idx = orig_idx[done]
                w_final[d_idx] = w[done]
                loss_final[d_idx] = loss_k[done]
                epochs_final[d_idx] = k - 1
                if not alive.any():
                    pbar.update(k - 1 - last_k)
                    last_k = k - 1
                    break
                orig_idx = orig_idx[alive]
                w = w[alive]
                v = v[alive]
                mu_vec = mu_vec[alive]
                loss_k = loss_k[alive]
                grad_k = grad_k[alive]
                if H is not None:
                    H = H[alive]
            if k - last_k >= 2500:
                pbar.update(k - last_k)
                pbar.set_postfix(active=int(w.shape[0]), E_ave=f"{loss_k.mean().item()*1e3:.3f}")
                last_k = k

        Ra = w.shape[0]
        total_run_steps_executed += Ra
        still = torch.norm(grad_k, dim=1) > eps

        w_nes = w + mu_vec[:, None] * v
        loss_nes, grad_nes = model.loss_and_grad(w_nes, X_train, XT_train, D_train)

        if is_qn:
            I_a = I_d.unsqueeze(0).expand(Ra, Dw, Dw)
            g_raw = -torch.bmm(H, grad_nes.unsqueeze(-1)).squeeze(-1)
        else:
            g_raw = -grad_nes

        scale = torch.clamp(torch.norm(g_raw, dim=1, keepdim=True) / max_step_norm, min=1.0)
        g_k = g_raw / scale
        slope = torch.sum(grad_nes * g_k, dim=1)

        if is_qn:
            bad = slope >= 0.0
            H = torch.where(bad[:, None, None], I_a, H)
            g_fb = -grad_nes / torch.clamp(torch.norm(grad_nes, dim=1, keepdim=True) / max_step_norm, min=1.0)
            g_k = torch.where(bad[:, None], g_fb, g_k)
            slope = torch.sum(grad_nes * g_k, dim=1)

        alpha = staged_parallel_armijo(model, w_nes, g_k, loss_nes, slope, X_train, D_train)
        v_next = mu_vec[:, None] * v + alpha[:, None] * g_k
        w_next = w + v_next
        loss_next, grad_next = model.loss_and_grad(w_next, X_train, XT_train, D_train)

        if is_qn:
            p_k = alpha[:, None] * g_k
            q_k = grad_next - grad_nes
            pq = torch.sum(p_k * q_k, dim=1)
            Hq = torch.bmm(H, q_k.unsqueeze(-1)).squeeze(-1)
            qHq = torch.sum(q_k * Hq, dim=1)
            pq_s = torch.where(pq > 1e-14, pq, torch.ones_like(pq))
            term1 = (Hq.unsqueeze(2) * p_k.unsqueeze(1) + p_k.unsqueeze(2) * Hq.unsqueeze(1)) / pq_s[:, None, None]
            term2 = (1.0 + qHq / pq_s)[:, None, None] * (p_k.unsqueeze(2) * p_k.unsqueeze(1)) / pq_s[:, None, None]
            H_new = torch.where((pq > 1e-14)[:, None, None], H - term1 + term2, I_a)
            H = torch.where(still[:, None, None], H_new, H)

        w = torch.where(still[:, None], w_next, w)
        v = torch.where(still[:, None], v_next, v)
        loss_k = torch.where(still, loss_next, loss_k)
        grad_k = torch.where(still[:, None], grad_next, grad_k)

    if orig_idx.numel() > 0:
        still = torch.norm(grad_k, dim=1) > eps
        ep_rem = torch.where(still, torch.full_like(orig_idx, max_epochs), torch.full_like(orig_idx, k))
        w_final[orig_idx] = w
        loss_final[orig_idx] = loss_k
        epochs_final[orig_idx] = ep_rem
        if last_k < max_epochs:
            pbar.update(max_epochs - last_k)

    pbar.close()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    sec_per_run_step = elapsed / max(1, total_run_steps_executed)

    with torch.no_grad():
        loss_test = model.loss_only(w_final, X_test, D_test)

    e_tr = (loss_final * 1000.0).view(M, R0).cpu().numpy()
    e_te = (loss_test * 1000.0).view(M, R0).cpu().numpy()
    ep_mat = epochs_final.view(M, R0).cpu().numpy()

    out = []
    for m_idx, mu_val in enumerate(mu_list):
        ave_ep = float(ep_mat[m_idx].mean())
        out.append({
            "algo": algo,
            "mu": mu_val,
            "train_ave": float(e_tr[m_idx].mean()),
            "train_best": float(e_tr[m_idx].min()),
            "train_worst": float(e_tr[m_idx].max()),
            "test_ave": float(e_te[m_idx].mean()),
            "test_best": float(e_te[m_idx].min()),
            "test_worst": float(e_te[m_idx].max()),
            "ave_epochs": ave_ep,
            "est_time_sec": float(ave_ep * sec_per_run_step),
            "total_batch_sec": elapsed,
        })
    return out


def run_convergence_curves_no_stop(
    mu_configs: List[float],
    w_init_single_set: torch.Tensor,
    X_train: torch.Tensor,
    XT_train: torch.Tensor,
    D_train: torch.Tensor,
    max_epochs: int = 100000,
    record_stride: int = 100,
    max_step_norm: float = 1.0,
) -> Dict[str, np.ndarray]:
    """
    論文 Fig. 3 用: 終了条件を無効化し，QN (mu=0.0) および NAQ (mu=0.80, 0.85, 0.90, 0.95) を
    max_epochs (=100,000) まで全30試行学習させ，各エポックにおける平均訓練誤差 E_train(w) を記録する．
    """
    device = X_train.device
    dtype = X_train.dtype
    model = BatchedMLP(1, 7, 1)

    M = len(mu_configs)
    R0, Dw = w_init_single_set.shape
    total_R = M * R0

    w = w_init_single_set.unsqueeze(0).expand(M, R0, Dw).reshape(total_R, Dw).clone()
    v = torch.zeros_like(w)
    mu_vec = torch.tensor(mu_configs, device=device, dtype=dtype).repeat_interleave(R0)[:, None]
    is_nes_mask = (mu_vec.squeeze(-1) > 0.0)

    I_d = torch.eye(Dw, device=device, dtype=dtype)
    I_all = I_d.unsqueeze(0).expand(total_R, Dw, Dw)
    H = I_all.clone()

    loss_k, grad_k = model.loss_and_grad(w, X_train, XT_train, D_train)

    # 記録するエポック番号: 0..1000 は10刻み，以降は record_stride 刻み
    recorded_epochs = [0]
    init_means = loss_k.view(M, R0).mean(dim=1).cpu().numpy()
    curves = [init_means]

    pbar = tqdm(total=max_epochs, desc=f"Fig.3 Convergence ({M}x{R0}={total_R} runs, no early-stop)", mininterval=1.0)
    last_k = 0

    for k in range(1, max_epochs + 1):
        w_nes = torch.where(is_nes_mask[:, None], w + mu_vec * v, w)
        loss_nes, grad_nes = model.loss_and_grad(w_nes, X_train, XT_train, D_train)

        g_raw = -torch.bmm(H, grad_nes.unsqueeze(-1)).squeeze(-1)
        scale = torch.clamp(torch.norm(g_raw, dim=1, keepdim=True) / max_step_norm, min=1.0)
        g_k = g_raw / scale
        slope = torch.sum(grad_nes * g_k, dim=1)

        bad = slope >= 0.0
        H = torch.where(bad[:, None, None], I_all, H)
        g_fb = -grad_nes / torch.clamp(torch.norm(grad_nes, dim=1, keepdim=True) / max_step_norm, min=1.0)
        g_k = torch.where(bad[:, None], g_fb, g_k)
        slope = torch.sum(grad_nes * g_k, dim=1)

        alpha = staged_parallel_armijo(model, w_nes, g_k, loss_nes, slope, X_train, D_train)
        v = mu_vec * v + alpha[:, None] * g_k
        w = w + v
        loss_next, grad_next = model.loss_and_grad(w, X_train, XT_train, D_train)

        p_k = alpha[:, None] * g_k
        q_k = grad_next - grad_nes
        pq = torch.sum(p_k * q_k, dim=1)
        Hq = torch.bmm(H, q_k.unsqueeze(-1)).squeeze(-1)
        qHq = torch.sum(q_k * Hq, dim=1)
        pq_s = torch.where(pq > 1e-14, pq, torch.ones_like(pq))
        term1 = (Hq.unsqueeze(2) * p_k.unsqueeze(1) + p_k.unsqueeze(2) * Hq.unsqueeze(1)) / pq_s[:, None, None]
        term2 = (1.0 + qHq / pq_s)[:, None, None] * (p_k.unsqueeze(2) * p_k.unsqueeze(1)) / pq_s[:, None, None]
        H = torch.where((pq > 1e-14)[:, None, None], H - term1 + term2, I_all)

        loss_k, grad_k = loss_next, grad_next

        if (k <= 1000 and k % 10 == 0) or (k % record_stride == 0):
            recorded_epochs.append(k)
            curves.append(loss_k.view(M, R0).mean(dim=1).cpu().numpy())

        if k - last_k >= 2500:
            pbar.update(k - last_k)
            last_k = k

    if last_k < max_epochs:
        pbar.update(max_epochs - last_k)
    pbar.close()

    curves_arr = np.stack(curves, axis=0)  # (T_steps, M)
    res_dict: Dict[str, np.ndarray] = {"epochs": np.array(recorded_epochs, dtype=np.int64)}
    for m_idx, mu_val in enumerate(mu_configs):
        key = "QN" if mu_val == 0.0 else f"NAQ({mu_val:.2f})"
        res_dict[key] = curves_arr[:, m_idx]
    return res_dict
