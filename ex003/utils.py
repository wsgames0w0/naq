"""
ex003/utils.py
論文 Ninomiya (NOLTA 2017) の再現実験3 (4.1節 Example 2: 5次元 Levy 関数近似問題，Table III)
に使用するデータセット生成・高速GPUバッチ学習・集計用関数群．
"""

import os
import time
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from model import BatchedMLP5D


def setup_japanese_matplotlib() -> None:
    """Matplotlibの日本語フォント設定．"""
    plt.rcParams["font.family"] = ["Meiryo", "Yu Gothic", "MS Gothic", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False


def levy_function_5d(X_raw: np.ndarray) -> np.ndarray:
    """
    論文 式(19) および文献[14]に基づく標準5次元 Levy 関数 f2(x1, ..., x5), x_i in [-4, 4].
    標準Levy変換 w_i = 1 + (x_i - 1) / 4 に対して
    f2 = (pi / n) * { sum_{i=1}^{n-1} (w_i - 1)^2 [1 + 10 sin^2(pi w_{i+1})] + 10 sin^2(pi w_1) + (w_n - 1)^2 }
    を計算する（正規化後の定数予測誤差 0.5*Var(d)*10^3 ≈ 52.5 となり論文 Table III の初期プラトー値と一致）．
    """
    n = X_raw.shape[1]
    w = 1.0 + (X_raw - 1.0) / 4.0
    term1 = np.sum(((w[:, :-1] - 1.0) ** 2) * (1.0 + 10.0 * (np.sin(np.pi * w[:, 1:]) ** 2)), axis=1)
    term2 = 10.0 * (np.sin(np.pi * w[:, 0]) ** 2)
    term3 = (w[:, -1] - 1.0) ** 2
    return (np.pi / float(n)) * (term1 + term2 + term3)


def load_or_create_ex2_dataset(
    dataset_dir: str = "../datasets",
    seed: int = 2,
    device: Optional[torch.device] = None,
) -> Dict[str, object]:
    """
    Example 2 のデータセット (全15,000点: 訓練10,000点・テスト5,000点，x_i in [-4, 4] 一様乱数) を
    生成または読み込み，[-1.0, 1.0] に正規化して返す．
    """
    os.makedirs(dataset_dir, exist_ok=True)
    npz_path = os.path.join(dataset_dir, "ex2_dataset.npz")

    if os.path.exists(npz_path):
        d = np.load(npz_path)
        X_all = d["X_all"]
        y_all = d["y_all"]
    else:
        np.random.seed(seed)
        X_all = np.random.uniform(-4.0, 4.0, size=(15000, 5)).astype(np.float64)
        y_all = levy_function_5d(X_all).astype(np.float64)
        np.savez(npz_path, X_all=X_all, y_all=y_all)

    X_min, X_max = X_all.min(axis=0), X_all.max(axis=0)
    y_min, y_max = float(y_all.min()), float(y_all.max())

    X_norm = 2.0 * (X_all - X_min) / (X_max - X_min) - 1.0
    y_norm = 2.0 * (y_all - y_min) / (y_max - y_min) - 1.0

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_tr_64 = torch.tensor(X_norm[:10000], device=device, dtype=torch.float64)
    D_tr_64 = torch.tensor(y_norm[:10000, None], device=device, dtype=torch.float64)
    X_te_64 = torch.tensor(X_norm[10000:], device=device, dtype=torch.float64)
    D_te_64 = torch.tensor(y_norm[10000:, None], device=device, dtype=torch.float64)

    return {
        "X_train_64": X_tr_64,
        "XT_train_64": X_tr_64.t().contiguous(),
        "D_train_64": D_tr_64,
        "X_test_64": X_te_64,
        "D_test_64": D_te_64,
        "X_train_32": X_tr_64.to(torch.float32),
        "XT_train_32": X_tr_64.t().contiguous().to(torch.float32),
        "D_train_32": D_tr_64.to(torch.float32),
        "X_test_32": X_te_64.to(torch.float32),
        "D_test_32": D_te_64.to(torch.float32),
    }


def generate_initial_weights_ex2(
    num_runs: int = 10, num_params: int = 1051, seed: int = 2017, device: Optional[torch.device] = None
) -> torch.Tensor:
    """10試行分の初期重み (10, 1051) を [-0.5, 0.5] 一様乱数で生成する．"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    return (torch.rand((num_runs, num_params), generator=gen, dtype=torch.float64) - 0.5).to(device)


def staged_armijo_ex2(
    model: BatchedMLP5D,
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
    大規模モデル向け省メモリ多段階Armijo直線探索:
    Stage 0 で alpha=1.0 を検証し，棄却された試行のみ8候補ずつチャンク評価することでVRAM肥大化を防ぐ．
    """
    Ra, Dw = w_base.shape
    alpha = torch.ones(Ra, device=w_base.device, dtype=w_base.dtype)
    l0 = model.loss_only(w_base + g_k, X, D)
    ok0 = l0 <= loss_base + xi * slope
    if ok0.all():
        return alpha

    rem = torch.nonzero(~ok0).squeeze(-1)
    for p_start, p_end in [(1, 9), (9, 17), (17, 26)]:
        K = p_end - p_start
        cands = tau ** torch.arange(p_start, p_end, device=w_base.device, dtype=w_base.dtype)
        wc = w_base[rem].unsqueeze(1) + cands[None, :, None] * g_k[rem].unsqueeze(1)
        lc = model.loss_only(wc.reshape(rem.numel() * K, Dw), X, D).view(rem.numel(), K)
        ok_c = lc <= loss_base[rem].unsqueeze(1) + xi * cands.unsqueeze(0) * slope[rem].unsqueeze(1)
        any_c = ok_c.any(dim=1)
        if any_c.any():
            hit = rem[any_c]
            idx_first = torch.argmax(ok_c[any_c].to(torch.int64), dim=1)
            alpha[hit] = cands[idx_first]
        rem = rem[~any_c]
        if rem.numel() == 0:
            break
    if rem.numel() > 0:
        alpha[rem] = 0.0
    return alpha


def run_first_order_group_ex2(
    configs: List[Tuple[str, str, float]],
    w_init_10_64: torch.Tensor,
    ds: Dict[str, torch.Tensor],
    max_epochs: int = 100000,
    eps: float = 1e-8,
    fixed_lr: float = 0.01,
    max_step_norm: float = 2.0,
    record_stride: int = 1000,
) -> Tuple[List[Dict[str, object]], Dict[str, np.ndarray]]:
    """
    Example 2 における1次手法 (SG, NAG, Qprop) を統合バッチとしてGPU上で高速実行する．
    コンシューマGPU (RTX 4080) の単精度/TF32演算器を活用して計算時間を数分に短縮し，
    最終評価は倍精度で厳密に算出する．
    """
    torch.backends.cuda.matmul.allow_tf32 = True
    X_tr, XT_tr, D_tr = ds["X_train_32"], ds["XT_train_32"], ds["D_train_32"]
    X_te_64, D_te_64 = ds["X_test_64"], ds["D_test_64"]
    X_tr_64, D_tr_64 = ds["X_train_64"], ds["D_train_64"]
    device = X_tr.device
    model = BatchedMLP5D(5, 150, 1)

    M = len(configs)
    R0, Dw = w_init_10_64.shape
    total_R = M * R0

    # 全設定が fixed か Armijo かを統一して呼ぶ前提
    smode = configs[0][1]
    assert all(c[1] == smode for c in configs)

    w = w_init_10_64.to(torch.float32).unsqueeze(0).expand(M, R0, Dw).reshape(total_R, Dw).clone()
    v = torch.zeros_like(w)

    mu_list = [c[2] if c[0] == "NAG" else 0.0 for c in configs]
    is_qprop_list = [c[0] == "Qprop" for c in configs]
    mu_vec = torch.tensor(mu_list, device=device, dtype=torch.float32).repeat_interleave(R0)[:, None]
    nes_mask = (mu_vec.squeeze(-1) > 0.0)
    qprop_mask = torch.tensor(is_qprop_list, device=device, dtype=torch.bool).repeat_interleave(R0)
    has_any_nes = bool(nes_mask.any().item())
    has_any_qprop = bool(qprop_mask.any().item())

    s_prev = torch.zeros_like(w) if has_any_qprop else None
    y_prev = torch.zeros_like(w) if has_any_qprop else None

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    loss_k, grad_k = model.loss_and_grad(w, X_tr, XT_tr, D_tr)
    rec_epochs = [0]
    rec_curves = [loss_k.view(M, R0).mean(dim=1).cpu().numpy()]

    desc_str = f"Ex2 1st-Order ({smode}, {M}x{R0}={total_R} runs)"
    pbar = tqdm(total=max_epochs, desc=desc_str, mininterval=1.5)
    last_k = 0

    for k in range(1, max_epochs + 1):
        if has_any_nes:
            w_base = torch.where(nes_mask[:, None], w + mu_vec * v, w)
            loss_base, grad_base = model.loss_and_grad(w_base, X_tr, XT_tr, D_tr)
        else:
            w_base, loss_base, grad_base = w, loss_k, grad_k

        g_raw = -grad_base
        if has_any_qprop and k > 1:
            sy_pos = (s_prev * y_prev) > 1e-12
            ratio = torch.where(sy_pos, s_prev / torch.where(sy_pos, y_prev, torch.ones_like(y_prev)), torch.ones_like(s_prev))
            lam = torch.clamp(ratio, min=1e-3, max=1e3)
            g_qp = -lam * grad_base
            g_raw = torch.where(qprop_mask[:, None], g_qp, g_raw)

        if smode == "Armijo":
            scale = torch.clamp(torch.norm(g_raw, dim=1, keepdim=True) / max_step_norm, min=1.0)
            g_k = g_raw / scale
            slope = torch.sum(grad_base * g_k, dim=1)
            alpha = staged_armijo_ex2(model, w_base, g_k, loss_base, slope, X_tr, D_tr)
        else:
            g_k = g_raw
            alpha = torch.full((total_R,), fixed_lr, device=device, dtype=torch.float32)

        v_next = mu_vec * v + alpha[:, None] * g_k
        w_next = w + v_next
        loss_next, grad_next = model.loss_and_grad(w_next, X_tr, XT_tr, D_tr)

        if has_any_qprop:
            s_prev = w_next - w
            y_prev = grad_next - grad_k

        w, v, loss_k, grad_k = w_next, v_next, loss_next, grad_next

        if k % record_stride == 0:
            rec_epochs.append(k)
            rec_curves.append(loss_k.view(M, R0).mean(dim=1).cpu().numpy())

        if k - last_k >= 2500:
            pbar.update(k - last_k)
            last_k = k

    if last_k < max_epochs:
        pbar.update(max_epochs - last_k)
    pbar.close()

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    # 最終重みを倍精度 (float64) に変換して厳密な訓練・テスト誤差を評価
    w_64 = w.to(torch.float64)
    with torch.no_grad():
        loss_tr_64 = model.loss_only(w_64, X_tr_64, D_tr_64).view(M, R0).cpu().numpy() * 1000.0
        loss_te_64 = model.loss_only(w_64, X_te_64, D_te_64).view(M, R0).cpu().numpy() * 1000.0

    results = []
    curves_arr = np.stack(rec_curves, axis=0)
    curves_dict: Dict[str, np.ndarray] = {"epochs": np.array(rec_epochs, dtype=np.int64)}
    for m_idx, (algo, sm, mu_val) in enumerate(configs):
        tag = f"{algo}({sm},{mu_val:.2f})" if algo == "NAG" else f"{algo}({sm})"
        curves_dict[tag] = curves_arr[:, m_idx]
        results.append({
            "algo": algo,
            "stepsize": sm,
            "mu": f"{mu_val:.2f}" if algo == "NAG" else "-",
            "train_ave": float(loss_tr_64[m_idx].mean()),
            "train_best": float(loss_tr_64[m_idx].min()),
            "train_worst": float(loss_tr_64[m_idx].max()),
            "test_ave": float(loss_te_64[m_idx].mean()),
            "test_best": float(loss_te_64[m_idx].min()),
            "test_worst": float(loss_te_64[m_idx].max()),
            "ave_epochs": float(max_epochs),
            "ave_time_sec": float(elapsed / total_R),
            "batch_time_sec": float(elapsed),
        })
    return results, curves_dict


def run_second_order_ex2(
    algo: str,
    mu: float,
    w_init_10_64: torch.Tensor,
    ds: Dict[str, torch.Tensor],
    max_epochs: int = 100000,
    eps: float = 1e-8,
    max_step_norm: float = 2.0,
    record_stride: int = 250,
) -> Tuple[Dict[str, object], np.ndarray, np.ndarray]:
    """
    Example 2 における2次手法 (QN および NAQ) を完全倍精度 (IEEE 754 float64) で実行する．
    dim(w) = 1,051 の逆ヘッセ近似行列 H_k (10, 1051, 1051) を保持し，
    収束済み試行の動的バッチ圧縮および過大オーバーシュート時のモーメンタムリスタートを備える．
    """
    X_tr, XT_tr, D_tr = ds["X_train_64"], ds["XT_train_64"], ds["D_train_64"]
    X_te, D_te = ds["X_test_64"], ds["D_test_64"]
    device = X_tr.device
    dtype = torch.float64
    model = BatchedMLP5D(5, 150, 1)

    R0, Dw = w_init_10_64.shape
    is_nes = (algo == "NAQ") and (mu > 0.0)

    w_final = w_init_10_64.clone()
    loss_final = torch.zeros(R0, device=device, dtype=dtype)
    epochs_final = torch.zeros(R0, device=device, dtype=torch.long)

    orig_idx = torch.arange(R0, device=device)
    w = w_init_10_64.clone()
    v = torch.zeros_like(w)
    I_d = torch.eye(Dw, device=device, dtype=dtype)
    H = I_d.unsqueeze(0).expand(R0, Dw, Dw).clone()

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    loss_k, grad_k = model.loss_and_grad(w, X_tr, XT_tr, D_tr)
    # 収束曲線追跡用に全10試行の現在損失を保持
    live_losses_all = loss_k.clone()
    rec_epochs = [0]
    rec_means = [float(live_losses_all.mean().item())]

    label = f"{algo} (Armijo, mu={mu:.2f})" if algo == "NAQ" else "QN (Armijo)"
    pbar = tqdm(total=max_epochs, desc=label, mininterval=1.5)
    last_k = 0

    for k in range(1, max_epochs + 1):
        if k % 100 == 1:
            alive = torch.norm(grad_k, dim=1) > eps
            done = ~alive
            if done.any():
                d_idx = orig_idx[done]
                w_final[d_idx] = w[done]
                loss_final[d_idx] = loss_k[done]
                live_losses_all[d_idx] = loss_k[done]
                epochs_final[d_idx] = k - 1
                if not alive.any():
                    pbar.update(k - 1 - last_k)
                    last_k = k - 1
                    break
                orig_idx = orig_idx[alive]
                w = w[alive]
                v = v[alive]
                H = H[alive]
                loss_k = loss_k[alive]
                grad_k = grad_k[alive]

            if k - last_k >= 1000:
                pbar.update(k - last_k)
                pbar.set_postfix(
                    active=int(w.shape[0]),
                    E_ave=f"{live_losses_all.mean().item()*1e3:.5f}",
                    gnorm=f"{torch.norm(grad_k, dim=1).min().item():.2e}",
                )
                last_k = k

        Ra = w.shape[0]
        I_a = I_d.unsqueeze(0).expand(Ra, Dw, Dw)
        still = torch.norm(grad_k, dim=1) > eps

        if is_nes:
            w_nes = w + mu * v
            loss_nes, grad_nes = model.loss_and_grad(w_nes, X_tr, XT_tr, D_tr)
            # 高次元谷壁へのオーバーシュート検知時のみモーメンタムをリセットし単調減少性を保護
            over = loss_nes > loss_k
            if over.any():
                v = torch.where(over[:, None], torch.zeros_like(v), v)
                w_nes = torch.where(over[:, None], w, w_nes)
                loss_nes = torch.where(over, loss_k, loss_nes)
                grad_nes = torch.where(over[:, None], grad_k, grad_nes)
        else:
            w_nes, loss_nes, grad_nes = w, loss_k, grad_k

        g_raw = -torch.bmm(H, grad_nes.unsqueeze(-1)).squeeze(-1)
        scale = torch.clamp(torch.norm(g_raw, dim=1, keepdim=True) / max_step_norm, min=1.0)
        g_k = g_raw / scale
        slope = torch.sum(grad_nes * g_k, dim=1)

        bad = slope >= 0.0
        if bad.any():
            H = torch.where(bad[:, None, None], I_a, H)
            g_fb = -grad_nes / torch.clamp(torch.norm(grad_nes, dim=1, keepdim=True) / max_step_norm, min=1.0)
            g_k = torch.where(bad[:, None], g_fb, g_k)
            slope = torch.sum(grad_nes * g_k, dim=1)

        alpha = staged_armijo_ex2(model, w_nes, g_k, loss_nes, slope, X_tr, D_tr)
        v_next = (mu * v + alpha[:, None] * g_k) if is_nes else (alpha[:, None] * g_k)
        w_next = w + v_next
        loss_next, grad_next = model.loss_and_grad(w_next, X_tr, XT_tr, D_tr)

        p_k = alpha[:, None] * g_k
        q_k = grad_next - grad_nes
        pq = torch.sum(p_k * q_k, dim=1)
        Hq = torch.bmm(H, q_k.unsqueeze(-1)).squeeze(-1)
        qHq = torch.sum(q_k * Hq, dim=1)
        pq_s = torch.where(pq > 1e-14, pq, torch.ones_like(pq))
        term1 = (Hq.unsqueeze(2) * p_k.unsqueeze(1) + p_k.unsqueeze(2) * Hq.unsqueeze(1)) / pq_s[:, None, None]
        term2 = (1.0 + qHq / pq_s)[:, None, None] * (p_k.unsqueeze(2) * p_k.unsqueeze(1)) / pq_s[:, None, None]
        H_new = torch.where((pq > 1e-14)[:, None, None], H - term1 + term2, I_a)

        w = torch.where(still[:, None], w_next, w)
        v = torch.where(still[:, None], v_next, v)
        H = torch.where(still[:, None, None], H_new, H)
        loss_k = torch.where(still, loss_next, loss_k)
        grad_k = torch.where(still[:, None], grad_next, grad_k)

        if k % record_stride == 0:
            live_losses_all[orig_idx] = loss_k
            rec_epochs.append(k)
            rec_means.append(float(live_losses_all.mean().item()))

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

    with torch.no_grad():
        loss_te = model.loss_only(w_final, X_te, D_te)

    e_tr = (loss_final * 1000.0).cpu().numpy()
    e_te = (loss_te * 1000.0).cpu().numpy()
    ep_np = epochs_final.cpu().numpy()

    res = {
        "algo": algo,
        "stepsize": "Armijo",
        "mu": f"{mu:.2f}" if algo == "NAQ" else "-",
        "train_ave": float(e_tr.mean()),
        "train_best": float(e_tr.min()),
        "train_worst": float(e_tr.max()),
        "test_ave": float(e_te.mean()),
        "test_best": float(e_te.min()),
        "test_worst": float(e_te.max()),
        "ave_epochs": float(ep_np.mean()),
        "ave_time_sec": float(elapsed / R0),
        "batch_time_sec": float(elapsed),
    }
    return res, np.array(rec_epochs, dtype=np.int64), np.array(rec_means, dtype=np.float64)


def format_ex2_dataframe(results: List[Dict[str, object]]) -> pd.DataFrame:
    """論文 Table III 形式の集計表を作成する．"""
    rows = []
    for r in results:
        rows.append({
            "Algorithm": r["algo"],
            "Stepsize": r["stepsize"],
            "mu": r["mu"],
            "E_train (x10^3) Ave/Best/Worst": f"{r['train_ave']:.5f} / {r['train_best']:.5f} / {r['train_worst']:.5f}",
            "Ave. T(sec)/run": round(r["ave_time_sec"], 2),
            "Batch GPU T(sec)": round(r["batch_time_sec"], 1),
            "Ave. epochs": int(round(r["ave_epochs"])),
            "E_test (x10^3) Ave/Best/Worst": f"{r['test_ave']:.5f} / {r['test_best']:.5f} / {r['test_worst']:.5f}",
        })
    return pd.DataFrame(rows)
