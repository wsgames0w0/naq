# NAQ (Nesterov's Accelerated Quasi-Newton) PyTorch 再現実験

本リポジトリは，以下の論文で提案されたニューラルネットワーク向け2次最適化手法 **NAQ (Nesterov's Accelerated Quasi-Newton)** を Python / PyTorch で実装し，数値実験の再現を行ったプロジェクトである．

> Hiroshi Ninomiya, "A novel quasi-Newton-based optimization for neural network training incorporating Nesterov's accelerated gradient," *Nonlinear Theory and Its Applications, IEICE*, vol. 8, no. 4, pp. 1–13, Oct. 2017. DOI: 10.1587/nolta.8.1

---

## 1. プロジェクト概要

NAQ は，準ニュートン法（BFGS）が持つ局所的な曲率情報（逆ヘッセ近似行列 $\hat{\mathbf{H}}\_k$）と，Nesterov の加速勾配法（NAG）の先読み慣性ベクトル $\mu \mathbf{v}\_k$ を数理的に統合した最適化アルゴリズムである．

本プロジェクトでは，PyTorch 標準の `torch.optim.Optimizer` に準拠した汎用オプティマイザ [`NAQOptimizer`](ex001/utils.py) を実装するとともに，論文のモンテカルロ試行（多数の独立試行・ハイパーパラメータ探索）を高速化する GPU バッチ並列計算エンジンを構築した．論文 4.1 節に示されたすべての数学的ベンチマーク（Table I，Table II，Table III，Fig. 3）について忠実な再現実験を行い，詳細な検証レポートを作成した．

---

## 2. ディレクトリ構成

```text
naq/
├── ex001/                  # 実験1: 1次元高非線形関数近似 (Table I)
│   ├── model.py            # ネットワークモデル定義 (FeedforwardMLP, BatchedMLP)
│   ├── utils.py            # NAQOptimizer 実装 & バッチ学習エンジン
│   └── ex001.ipynb         # 実験1 実行ノートブック
├── ex002/                  # 実験2: モーメンタム感度解析 & 収束曲線 (Table II, Fig. 3)
│   ├── model.py            # Flattened GEMM 高速モデル定義
│   ├── utils.py            # 570試行一括並列エンジン
│   └── ex002.ipynb         # 実験2 実行ノートブック
├── ex003/                  # 実験3: 5次元多峰性 Levy 関数近似 (Table III)
│   ├── model.py            # TorchScript 融合カーネル付きモデル (1,051パラメータ)
│   ├── utils.py            # 完全倍精度 BFGS バッチ学習エンジン
│   └── ex003.ipynb         # 実験3 実行ノートブック
├── report/                 # 詳細実験レポート
│   ├── img/                # 生成図・グラフ (PNG)
│   ├── ex001.md            # 実験1 レポート
│   ├── ex002.md            # 実験2 レポート
│   ├── ex003.md            # 実験3 レポート
│   └── report.md           # 総合レポート (数理導出・Python実装詳細・統合考察)
├── datasets/               # データセット (自動生成)
├── requirements.txt        # 依存ライブラリ一覧
├── LICENSE                 # MIT License
├── .gitignore              # Git 除外設定
└── README.md               # 本ファイル
```

---

## 3. 実験環境と構築手順

### 3.1 動作確認環境

- **OS**: Windows 11 (64-bit)
- **Python**: 3.10.11
- **GPU**: NVIDIA GeForce RTX 4080 (16 GB VRAM)
- **CUDA**: 11.8
- **主要パッケージ**: `torch==2.7.1+cu118`, `torchvision==0.22.1+cu118`, `numpy==2.2.6`, `pandas==2.3.3`, `matplotlib==3.10.9`, `tqdm==4.70.1`

### 3.2 仮想環境の起動と依存関係の導入

```powershell
# 仮想環境の有効化
.\venv\Scripts\activate

# 依存パッケージの導入 (新規環境構築時)
pip install -r requirements.txt
```

---

## 4. 各実験の概要と実行方法

各実験の Jupyter Notebook（`.ipynb`）には，データセットの生成，モデル構築，学習，集計，グラフ描画の全パイプラインが含まれている．

### 実験1 (`ex001/`): 1次元高非線形関数近似 (Table I)

- **対象関数**: $f_1(x) = 1 + (x + 2x^2)\mathrm{sin}(-x^2)$（$|x| \le 4$）
- **モデル**: 1 − 7 − 1 MLP（$\dim(\mathbf{w}) = 22$）
- **検証項目**: 
  - 標準 `NAQOptimizer`（`autograd`）と GPU バッチ並列実装の数値的等価性（$k=1$ での最大差 $2.78 \times 10^{-17}$）
  - 全16条件（SG, NAG, Qprop, QN, NAQ）の比較
- **詳細レポート**: [report/ex001.md](report/ex001.md)

### 実験2 (`ex002/`): モーメンタム係数 $\mu$ の感度解析と収束特性 (Table II, Fig. 3)

- **検証項目**:
  - $\mu \in [0.05, 0.95]$（19通り）における NAQ と NAG の感度解析（570試行の一括並列学習）
  - 終了条件を無効化した $100,000$ エポックの長期収束曲線比較（QN vs NAQ 4条件）
- **詳細レポート**: [report/ex002.md](report/ex002.md)

### 実験3 (`ex003/`): 5次元多峰性 Levy 関数近似 (Table III)

- **対象関数**: 5次元 Levy 関数 No. 8
- **モデル**: 5 − 150 − 1 MLP（$\dim(\mathbf{w}) = 1,051$）
- **検証項目**:
  - 1,051次元の大規模パラメータ空間における全16条件の性能評価
  - TorchScript 融合カーネルと多段階 Armijo 直線探索による高精度倍精度計算
- **詳細レポート**: [report/ex003.md](report/ex003.md)

---

## 5. 主要な再現成果

| 項目 | 成果の要点 |
| :--- | :--- |
| **収束加速** | Example 1 において，NAQ ($\mu = 0.80 \sim 0.90$) は QN と比較して平均エポック数を約 1/11 〜 1/17，計算時間を 6〜7 倍短縮した（論文 Table I の結論を忠実に再現）． |
| **感度の頑健性** | NAQ は $\mu \in [0.05, 0.95]$ の全領域で安定して収束し，1次手法（NAG）に比べてハイパーパラメータへの過敏性が著しく低いことを確認した（Table II）．推奨値は $\mu = 0.80 \sim 0.85$． |
| **高次元・多峰性での精度** | 1,051次元の Example 2 において，1次手法（誤差 $10^0 \sim 10^1$）を数万倍凌駕する $10^{-4} \times 10^{-3}$（$10^{-7}$ レベル）の極小誤差へ到達した（Table III）． |
| **PyTorch 実装の提示** | `torch.optim.Optimizer` 準拠の `NAQOptimizer` を作成し，先読み点勾配と通常勾配の2段階評価，セカント条件に基づく逆ヘッセ更新，Armijo 直線探索，シグモイド飽和を防ぐ初期ステップ制限などの実装方法を [report/report.md](report/report.md) に完全解説した． |

---

## 6. レポート一覧

本プロジェクトの数理的背景，実装の詳細解説，実験データおよび考察は，以下のレポートに記載されている．

1. [総合レポート (report/report.md)](report/report.md): NAQ の数理導出，PyTorch 実装の詳細，3実験の統合考察
2. [実験1 レポート (report/ex001.md)](report/ex001.md): 1次元高非線形関数近似 (Table I)
3. [実験2 レポート (report/ex002.md)](report/ex002.md): $\mu$ 感度解析と収束曲線 (Table II, Fig. 3)
4. [実験3 レポート (report/ex003.md)](report/ex003.md): 5次元 Levy 関数近似 (Table III)

---

## 7. ライセンス

本リポジトリのソースコードおよびレポートは [MIT License](LICENSE) の下で公開されている．
