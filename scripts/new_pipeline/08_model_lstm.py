#!/usr/bin/env python3
"""
08_model_lstm.py — 全局 LSTM + 车系 embedding 月度预测（无舆情基线）

数据来源（统一）：06_make_splits.py 的 train / val / test。
  * 模型权重只在 train (2022-01..2025-06) 上学习；
  * 测试预测以已观测的 val (2025-07..12) 作为历史，预测 test
    (2026-01..06)，绝不读取 test 真实销量。

Run:
  python scripts/new_pipeline/08_model_lstm.py
"""
import os
import warnings
import random
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import _font_setup
import torch
import torch.nn as nn

import _model_utils as mu
import _subset

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG = os.path.join(BASE, "figures_new")
PROC = os.path.join(BASE, "data", "processed_new", "stage3")
os.makedirs(PROC, exist_ok=True)

WIN = 12
SEED = 42
EPOCHS = 40
BATCH = 256
TEST_STEPS = 6   # test 窗口长度（月）

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

TRAIN_STYLE = "#4C78A8"
TEST_STYLE = "#F58518"
FC_STYLE = "#72B7B2"


def msin(m):
    return np.sin(2 * np.pi * m / 12.0)


def mcos(m):
    return np.cos(2 * np.pi * m / 12.0)


class LSTMModel(nn.Module):
    def __init__(self, n_series, emb=10, hidden=40):
        super().__init__()
        self.emb = nn.Embedding(n_series, emb)
        self.lstm = nn.LSTM(3, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + emb + 2, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, xseq, sidx, xmeta):
        e = self.emb(sidx)
        out, _ = self.lstm(xseq)
        h = out[:, -1, :]
        return self.head(torch.cat([h, e, xmeta], dim=1)).squeeze(-1)


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[LSTM] device: {device}")

    tr, va, te = mu.load_splits()
    subset = _subset.load_subset()
    print(f"[LSTM] 评估子集 {len(subset)} 系 | 训练用 train ({len(tr)} 行)")

    # 全部 in-pop 车系用于 embedding（train 中存在的车系）
    names = sorted(tr["series_name"].astype(str).unique())
    name2idx = {n: i for i, n in enumerate(names)}

    # 每车系 train 序列及其真实日历月份。不能把每个车系的第一行
    # 都当作一月：train.csv 为了 lag 完整性会从不同月份开始。
    tr_by_name = {}
    tr_dates_by_name = {}
    for name in names:
        g = tr[tr["series_name"].astype(str) == name].sort_values("date")
        tr_by_name[name] = g["monthly_sales"].astype(float).values
        tr_dates_by_name[name] = pd.to_datetime(g["date"]).values
    norm = {}
    for name in names:
        vals = np.log1p(tr_by_name[name])
        mu_, sd_ = float(vals.mean()), float(vals.std()) + 1e-6
        norm[name] = (mu_, sd_, vals)

    # 训练样本（全 train 窗口）
    Xseq, Xmeta, yv, sidxv = [], [], [], []
    for name in names:
        mu_, sd_, vn = norm[name]
        T = len(vn)
        months = pd.DatetimeIndex(tr_dates_by_name[name]).month.to_numpy()
        for i in range(WIN, T):
            seq = np.stack([vn[i - WIN:i], msin(months[i - WIN:i]), mcos(months[i - WIN:i])], axis=1)
            Xseq.append(seq)
            Xmeta.append([msin(months[i]), mcos(months[i])])
            yv.append(vn[i])
            sidxv.append(name2idx[name])
    Xseq = torch.tensor(np.array(Xseq), dtype=torch.float32, device=device)
    Xmeta = torch.tensor(np.array(Xmeta), dtype=torch.float32, device=device)
    yv = torch.tensor(np.array(yv), dtype=torch.float32, device=device)
    sidxv = torch.tensor(np.array(sidxv), dtype=torch.long, device=device)
    print(f"[LSTM] train samples: {len(yv)}")

    model = LSTMModel(len(names)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.005)
    lossf = nn.MSELoss()
    N = len(yv)
    for ep in range(EPOCHS):
        perm = torch.randperm(N, device=device)
        tot = 0.0
        model.train()
        for b in range(0, N, BATCH):
            ix = perm[b:b + BATCH]
            pred = model(Xseq[ix], sidxv[ix], Xmeta[ix])
            loss = lossf(pred, yv[ix])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            tot += loss.item() * len(ix)
        if ep % 10 == 0:
            print(f"[LSTM] epoch {ep:02d}  loss {tot / N:.4f}")

    # The realised validation period is available at the test origin and is
    # therefore valid history.  Test actuals are used only for scoring.
    va_by_name, va_dates_by_name = {}, {}
    te_by_name = {}
    te_dates_by_name = {}
    for name in subset:
        vg = va[va["series_name"].astype(str) == name].sort_values("date")
        tg = te[te["series_name"].astype(str) == name].sort_values("date")
        va_by_name[name] = vg["monthly_sales"].astype(float).values
        va_dates_by_name[name] = pd.to_datetime(vg["date"]).values
        te_by_name[name] = tg["monthly_sales"].astype(float).values
        te_dates_by_name[name] = pd.to_datetime(tg["date"]).values

    model.eval()
    rows, preds_rows, examples = [], [], []
    with torch.no_grad():
        for name in subset:
            if name not in norm:
                rows.append({"series_name": name, "status": "no_train"})
                continue
            mu_, sd_, vn = norm[name]
            val_actuals = va_by_name.get(name, np.array([]))
            actuals = te_by_name.get(name, np.array([]))
            test_dates = te_dates_by_name.get(name, np.array([]))
            if len(vn) <= WIN or len(val_actuals) != TEST_STEPS or len(actuals) != TEST_STEPS:
                rows.append({"series_name": name, "status": "too_short"})
                continue
            vn_min, vn_max = float(np.min(vn)), float(np.max(vn))
            hist = list(vn) + list(np.log1p(val_actuals))
            hist_months = list(pd.DatetimeIndex(tr_dates_by_name[name]).month) + \
                          list(pd.DatetimeIndex(va_dates_by_name[name]).month)
            preds = []
            for step in range(TEST_STEPS):
                target_month = int(pd.Timestamp(test_dates[step]).month)
                wm = np.asarray(hist_months[-WIN:])
                seq = torch.tensor(np.stack([np.array(hist[-WIN:], dtype=float),
                                             msin(wm), mcos(wm)], axis=1),
                                   dtype=torch.float32, device=device).unsqueeze(0)
                meta = torch.tensor([[msin(target_month), mcos(target_month)]],
                                    dtype=torch.float32, device=device)
                si = torch.tensor([name2idx[name]], dtype=torch.long, device=device)
                pn = float(model(seq, si, meta)[0].cpu())
                pn = min(max(pn, vn_min - 0.5), vn_max + 0.5)  # 防递归发散(log1p 空间裁剪)
                hist.append(pn)
                hist_months.append(target_month)
                preds.append(max(float(np.expm1(pn)), 0.0))  # pn 已是 log1p 空间
            met = mu.metrics(actuals, np.array(preds))
            met.update({"series_name": name, "status": "ok"})
            rows.append(met)
            for j in range(TEST_STEPS):
                preds_rows.append({"series_name": name,
                                   "date": pd.Timestamp(te[te["series_name"].astype(str) == name]
                                                        .sort_values("date")["date"].values[j])
                                   .strftime("%Y-%m-%d"),
                                   "actual": float(actuals[j]), "pred": float(preds[j])})
            if len(examples) < 9:
                examples.append((name, pd.Series(np.r_[tr_by_name[name], val_actuals]),
                                 pd.Series(actuals), pd.Series(preds)))

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(PROC, "lstm_results.csv"), index=False)
    if preds_rows:
        pd.DataFrame(preds_rows).to_csv(os.path.join(PROC, "lstm_preds.csv"), index=False)
    ok = res[res["status"] == "ok"] if "status" in res.columns else pd.DataFrame()
    print(f"\n[LSTM] test 评估 ok: {len(ok)}/{len(subset)}")
    if len(ok):
        a = pd.DataFrame(preds_rows)["actual"].values.astype(float)
        p = pd.DataFrame(preds_rows)["pred"].values.astype(float)
        print(f"  WMAPE(全局volume-weighted) = {mu.wmape_vol(a, p):.1f}%")
        print(f"  WMAPE(per-series mean)     = {ok['WMAPE'].mean():.1f}%  "
              f"(median {ok['WMAPE'].median():.1f}%)")
        print(f"  MAPE={ok['MAPE'].mean():.1f}%  RMSE={ok['RMSE'].mean():.1f}  MAE={ok['MAE'].mean():.1f}")

    n_ex = min(9, len(examples))
    if n_ex:
        cols, rows_n = 3, (n_ex + 2) // 3
        fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 4, rows_n * 2.6), constrained_layout=True)
        axes = np.array(axes).reshape(-1)
        for i, (name, trs, tes, fcs) in enumerate(examples):
            ax = axes[i]
            ax.plot(range(len(trs)), trs.values, color=TRAIN_STYLE, lw=1.2, label="train")
            ax.plot(range(len(trs), len(trs) + len(tes)), tes.values, color=TEST_STYLE, lw=1.6, marker="o", ms=3, label="actual")
            ax.plot(range(len(trs), len(trs) + len(fcs)), fcs.values, color=FC_STYLE, lw=1.6, marker="s", ms=3, label="forecast")
            ax.set_title(name, fontsize=9); ax.tick_params(labelsize=7)
            if i == 0:
                ax.legend(fontsize=7, loc="upper left")
        for j in range(n_ex, len(axes)):
            axes[j].axis("off")
        fig.suptitle("Global LSTM + series embedding — 6-month forecast on temporal TEST", fontsize=11)
        fig.savefig(os.path.join(FIG, "lstm_forecast.png"), dpi=130)
        print("[LSTM] figure saved -> figures_new/lstm_forecast.png")
    print("[LSTM] done.")


if __name__ == "__main__":
    main()
