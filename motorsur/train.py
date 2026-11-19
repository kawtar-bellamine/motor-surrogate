"""Train and compare surrogate models, measure the learning curve, and use the best surrogate to optimise a design."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from .dataset_fem import COMBOS, FEATURES, RANGES, TARGETS
from .fem import Machine, evaluate

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/designs_fem.csv"


def featurize(df: pd.DataFrame) -> np.ndarray:
    """Raw inputs plus the dimensionless groups an electrical engineer would use; (Q, p) one-hot encoded."""
    X = df[FEATURES].astype(float).copy()
    X["q"] = df["Q"] / (6 * df["p"])
    X["geff"] = df["g"] + df["hm"] / 1.05
    X["Bgap"] = df["Br"] * df["hm"] / (1.05 * df["g"] + df["hm"])          # open-circuit gap field, slotless
    pitch = 2 * np.pi * (df["R"] + df["g"]) / df["Q"]
    X["b0_over_g"] = df["so"] * pitch / X["geff"]
    X["slot_area"] = (1 - df["tw"]) * pitch * 0.9 * 0.45 * df["R"]
    X["elec_load"] = df["J"] * 1e6 * X["slot_area"] * df["Q"] / (2 * np.pi * df["R"])   # A/m
    X["tooth_flux"] = X["Bgap"] / df["tw"]                                # tooth field if nothing saturates
    X["R2"] = df["R"] ** 2
    X["cos_psi"] = np.cos(np.deg2rad(df["psi"]))
    # sizing equation (linear, no saturation): T ~ pi R^2 L * B1 * A1 * cos(psi); its logarithm makes the power
    # laws of electrical machines linear for the regression models
    B1 = 4 / np.pi * X["Bgap"] * np.sin(np.pi * df["alpha"] / 2)
    X["T_sizing"] = np.pi * df["R"] ** 2 * 0.1 * B1 * X["elec_load"] * np.sqrt(2) * X["cos_psi"]
    # cogging: its period count (LCM of slots and poles), its scale (gap field squared times the slot opening) and
    # where the magnet edges fall relative to the slots (pole arc in slot pitches, modulo one)
    X["lcm"] = np.lcm(df["Q"].astype(int), 2 * df["p"].astype(int))
    X["cog_scale"] = X["Bgap"] ** 2 * df["R"] ** 2 * df["so"] * pitch / df["g"]
    frac = (df["alpha"] * df["Q"] / (2 * df["p"])) % 1
    X["edge_sin"], X["edge_cos"] = np.sin(2 * np.pi * frac), np.cos(2 * np.pi * frac)
    for k in ("R", "g", "hm", "J", "Bgap", "elec_load", "T_sizing", "b0_over_g", "tooth_flux", "lcm", "cog_scale"):
        X[f"log_{k}"] = np.log(df[k] if k in df else X[k])
    for Q, p in COMBOS:
        X[f"combo_{Q}_{p}"] = ((df["Q"] == Q) & (df["p"] == p)).astype(float)
    return X.values


def y_of(df):
    return np.column_stack([np.log(df[t].clip(lower=1e-3)) for t in TARGETS])


def y_back(Y):
    return np.exp(Y)


class MLP:
    def __init__(self, hidden=96, epochs=600, lr=3e-3, seed=0):
        self.hidden, self.epochs, self.lr, self.seed = hidden, epochs, lr, seed

    def fit(self, X, Y):
        torch.manual_seed(self.seed)
        self.sx, self.sy = StandardScaler().fit(X), StandardScaler().fit(Y)
        Xt = torch.tensor(self.sx.transform(X), dtype=torch.float32)
        Yt = torch.tensor(self.sy.transform(Y), dtype=torch.float32)
        h = self.hidden
        self.net = torch.nn.Sequential(torch.nn.Linear(Xt.shape[1], h), torch.nn.SiLU(), torch.nn.Linear(h, h),
                                       torch.nn.SiLU(), torch.nn.Linear(h, h), torch.nn.SiLU(),
                                       torch.nn.Linear(h, Yt.shape[1]))
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs)
        g = torch.Generator().manual_seed(self.seed)
        for _ in range(self.epochs):
            perm = torch.randperm(len(Xt), generator=g)
            for i in range(0, len(Xt), 32):
                b = perm[i:i + 32]
                loss = torch.nn.functional.mse_loss(self.net(Xt[b]), Yt[b])
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()
        return self

    def predict(self, X):
        with torch.no_grad():
            out = self.net(torch.tensor(self.sx.transform(X), dtype=torch.float32)).numpy()
        return self.sy.inverse_transform(out)


class PerTarget:
    def __init__(self, make):
        self.make = make

    def fit(self, X, Y):
        self.sx = StandardScaler().fit(X)
        self.m = [self.make().fit(self.sx.transform(X), Y[:, j]) for j in range(Y.shape[1])]
        return self

    def predict(self, X):
        Xs = self.sx.transform(X)
        return np.column_stack([m.predict(Xs) for m in self.m])


MODELS = {
    "ridge (linear on log)": lambda: PerTarget(lambda: Ridge(alpha=1.0)),
    "k nearest neighbours": lambda: PerTarget(lambda: KNeighborsRegressor(n_neighbors=7, weights="distance")),
    "gradient boosting": lambda: PerTarget(lambda: HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05,
                                                                                 random_state=0)),
    "neural network (MLP)": lambda: MLP(),
}


def metrics(Y_true_log, Y_pred_log) -> dict:
    yt, yp = y_back(Y_true_log), y_back(Y_pred_log)
    out = {}
    for j, t in enumerate(TARGETS):
        err = np.abs(yp[:, j] - yt[:, j]) / np.abs(yt[:, j])
        ss = ((yt[:, j] - yp[:, j]) ** 2).sum() / ((yt[:, j] - yt[:, j].mean()) ** 2).sum()
        out[t] = dict(r2=round(float(1 - ss), 4), median_err_pct=round(float(100 * np.median(err)), 2),
                      p90_err_pct=round(float(100 * np.quantile(err, 0.9)), 2))
    return out


class Ensemble:
    """Mean of several networks trained from different seeds: steadier on a few hundred designs."""
    def __init__(self, n=5):
        self.n = n

    def fit(self, X, Y):
        self.ms = [MLP(seed=k).fit(X, Y) for k in range(self.n)]
        return self

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.ms], axis=0)




def cross_validate(df, X, Y, folds=5, seed=0):
    idx = np.random.default_rng(seed).permutation(len(df))
    parts = np.array_split(idx, folds)
    out = {}
    preds = {}
    for name, make in MODELS.items():
        P = np.zeros_like(Y); t_fit = 0.0
        for k in range(folds):
            te = parts[k]; tr = np.concatenate([parts[j] for j in range(folds) if j != k])
            t0 = time.perf_counter(); m = make().fit(X[tr], Y[tr]); t_fit += time.perf_counter() - t0
            P[te] = m.predict(X[te])
        out[name] = dict(metrics(Y, P), fit_seconds=round(t_fit / folds, 2))
        preds[name] = P
        print(name, {t: out[name][t]["median_err_pct"] for t in TARGETS}, flush=True)
    return out, preds


def evaluate_row(r) -> dict:
    m = Machine(**{k: (int(r[k]) if k in ("Q", "p") else float(r[k])) for k in ["Q", "p", "R", "g", "hm", "alpha",
                                                                             "so", "tw", "Br", "psi"]})
    m.NI = float(r["J"]) * 1e6 * m.slot_area
    out = {k: v for k, v in evaluate(m).items() if not k.startswith("_")}
    out.update({k: r[k] for k in FEATURES})
    return out


def sample_in(lim, n, rng):
    rows = []
    for _ in range(n):
        Q, p = COMBOS[rng.integers(len(COMBOS))]
        rows.append(dict(Q=Q, p=p, **{k: float(rng.uniform(lo, hi)) for k, (lo, hi) in lim.items()}))
    return pd.DataFrame(rows)


SPEC = dict(R_max=0.045, J_max=6.0, torque_min=25.0, cogging_max=1.0)


def feasible(s):
    return s["torque_mean"] >= SPEC["torque_min"] and s["cogging_pp"] <= SPEC["cogging_max"]


def optimise(model, n_cand=100_000, budget=8, seed=1):
    """A 25 N.m machine (0.1 m stack) with rotor radius <= 45 mm and current density <= 6 A/mm2 (natural cooling):
    minimise torque ripple with cogging <= 1 N.m. The surrogate screens candidates; FEM checks the best."""
    from multiprocessing import Pool
    rng = np.random.default_rng(seed)
    lim = dict(RANGES, R=(0.030, SPEC["R_max"]), J=(2.0, SPEC["J_max"]))
    designs = sample_in(lim, n_cand, rng)
    X = featurize(designs)
    t0 = time.perf_counter(); P = y_back(model.predict(X)); t_screen = time.perf_counter() - t0
    ok = (P[:, 0] >= SPEC["torque_min"] * 1.05) & (P[:, 2] <= SPEC["cogging_max"] * 0.8)
    idx = np.where(ok)[0][np.argsort(P[ok, 1])][:budget]
    with Pool(2) as pool:
        sims = pool.map(evaluate_row, [designs.iloc[i] for i in idx])
        base = {}
        for n in (budget, 5 * budget):
            rs = pool.map(evaluate_row, [r for _, r in sample_in(lim, n, np.random.default_rng(seed + n)).iterrows()])
            f = [s for s in rs if feasible(s)]
            base[f"random_{n}_fem_runs"] = dict(feasible=len(f), best_ripple=round(min(s["torque_ripple_pct"]
                                                                                    for s in f), 2) if f else None)
    feas = [s for s in sims if feasible(s)]
    best = min(feas, key=lambda s: s["torque_ripple_pct"]) if feas else None
    pred_vs_fem = [dict(pred_torque=round(float(P[i, 0]), 2), fem_torque=round(s["torque_mean"], 2),
                        pred_ripple=round(float(P[i, 1]), 2), fem_ripple=round(s["torque_ripple_pct"], 2),
                        pred_cogging=round(float(P[i, 2]), 3), fem_cogging=round(s["cogging_pp"], 3))
                   for i, s in zip(idx, sims)]
    return dict(spec=SPEC, candidates=n_cand, screen_seconds=round(t_screen, 2), predicted_feasible=int(ok.sum()),
                fem_runs=len(idx), feasible_after_fem=len(feas),
                best={k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in best.items()}
                if best else None, baselines=base, pred_vs_fem=pred_vs_fem)


def main():
    df = pd.read_csv(DATA)
    df = df[df["error"].isna() | (df["error"] == "")].reset_index(drop=True)
    X, Y = featurize(df), y_of(df)
    res = {"n_designs": len(df), "protocol": "5-fold cross-validation, every design predicted once",
           "fem_seconds_per_design": round(float(df["fem_seconds"].mean()), 1)}
    res["models"], preds = cross_validate(df, X, Y)
    best_name = min(res["models"], key=lambda k: np.mean([res["models"][k][t]["median_err_pct"] for t in TARGETS]))
    res["best_model"] = best_name
    # learning curve on a fixed test fold
    idx = np.random.default_rng(1).permutation(len(df)); te, pool_ = idx[:len(df) // 5], idx[len(df) // 5:]
    lc = {}
    for n in (40, 80, 160, len(pool_)):
        for name in ("gradient boosting", best_name):
            m = MODELS[name]().fit(X[pool_[:n]], Y[pool_[:n]])
            met = metrics(Y[te], m.predict(X[te]))
            lc.setdefault(name, {})[int(n)] = {t: met[t]["median_err_pct"] for t in TARGETS}
        print("learning curve", n, flush=True)
    res["learning_curve"] = lc
    final = MODELS[best_name]().fit(X, Y)
    t0 = time.perf_counter(); final.predict(np.repeat(X, 50, axis=0)); inf = (time.perf_counter() - t0) / (50 * len(X))
    res["inference_ms_per_design"] = round(1000 * inf, 5)
    res["speedup_vs_fem"] = int(res["fem_seconds_per_design"] / max(inf, 1e-12))
    pd.DataFrame(np.column_stack([y_back(Y), y_back(preds[best_name])]),
                 columns=[f"true_{t}" for t in TARGETS] + [f"pred_{t}" for t in TARGETS]).to_csv(
        ROOT / "results/test_predictions.csv", index=False)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/metrics.json").write_text(json.dumps(res, indent=1))
    res["optimisation"] = optimise(final)
    (ROOT / "results/metrics.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res["optimisation"], indent=1, default=float))


if __name__ == "__main__":
    main()
