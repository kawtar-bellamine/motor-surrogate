"""Figures for the README, from results/metrics.json, results/validation.json and a few FEM solves."""
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .analytic import gap_field  # noqa: E402
from .fem import Machine, q_axis_phase, solve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/figures"
BLUE, ORANGE, GREY, RED = "#1f5fa8", "#e69138", "#9aa7b5", "#c0392b"
NAMES = {"torque_mean": "mean torque (N·m)", "torque_ripple_pct": "torque ripple (%)",
         "cogging_pp": "cogging torque pk-pk (N·m)", "B_tooth_p95": "tooth flux density, 95th pct (T)"}


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def field():
    """Flux lines and |B| of a 36-slot, 6-pole machine at 8 A/mm2, saturating iron."""
    m = Machine(Q=36, p=3, R=0.04, g=0.0008, hm=0.004, alpha=0.8, so=0.25, tw=0.45)
    m.NI = 8e6 * m.slot_area
    g0, s = q_axis_phase(m.Q, m.p)
    r = solve(m, 0.0, gamma=g0)
    P, T, A = r["_P"], r["_T"], r["_A"]
    tri = mtri.Triangulation(P[0] * 1e3, P[1] * 1e3, T.T)
    Bmag = np.hypot(r["_Brad"], r["_Bth"])
    fig, ax = plt.subplots(figsize=(6.4, 6.0), dpi=160)
    tp = ax.tripcolor(tri, facecolors=Bmag, cmap="magma", vmin=0, vmax=2.2, rasterized=True)
    ax.tricontour(tri, A, levels=36, colors="w", linewidths=0.45)
    ax.set_aspect("equal"); ax.axis("off")
    cb = fig.colorbar(tp, ax=ax, shrink=0.75, pad=0.02); cb.set_label("|B| (T)")
    ax.set_title(f"FEM, 36 slots / 6 poles, 8 A/mm², saturating iron\ntorque {r['torque']:.1f} N·m on a 0.1 m stack, "
                 f"{r['elements']:,} elements, {r['iterations']} Newton iterations", fontsize=9.5)
    fig.tight_layout(); fig.savefig(OUT / "fig0_field.png"); plt.close(fig)


def validation(v):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.7), dpi=140)
    m = Machine(Q=24, p=2, R=0.04, g=0.001, hm=0.004, alpha=0.8, slotless=True, iron_mur=1e5)
    r = solve(m, 0.0, h_gap=m.g / 6)
    band = r["_reg"] == "band"; cx, cy = r["_c"]
    th = np.arctan2(cy, cx)[band]; o = np.argsort(th)
    rr = (m.R + m.Rs) / 2
    tt = np.linspace(-math.pi, math.pi, 1000)
    Ba, _ = gap_field(np.full_like(tt, rr), tt, m.Rr, m.R, m.Rs, m.p, m.alpha, m.Br, m.mur)
    axes[0].plot(np.degrees(th[o]), r["_Brad"][band][o], ".", ms=1.2, color=BLUE, label="FEM (band elements)")
    axes[0].plot(np.degrees(tt), Ba, color=RED, lw=1, label="exact (Zhu et al., 1993)")
    sl = [x for x in v["slotless_vs_exact"] if x["p"] == 2][-1]
    axes[0].set_title(f"Slotless machine: FEM vs exact field, error {sl['err_Br_pct']:.2f}%", fontsize=10)
    axes[0].set_xlabel("angle (degrees)"); axes[0].set_ylabel("radial B in the air gap (T)")
    axes[0].legend(frameon=False, fontsize=8.5, loc="lower right"); clean(axes[0])
    s = v["saturation"]
    J = np.array(s["J_A_per_mm2"])
    axes[1].plot(J, s["torque_linear_iron"], "o--", color=GREY, label="linear iron (µr 2000)")
    axes[1].plot(J, s["torque_saturating"], "o-", color=BLUE, label="saturating iron (Brauer law)")
    axes[1].set_xlabel("current density (A/mm², rms)"); axes[1].set_ylabel("mean torque (N·m)")
    gap = 100 * (1 - s["torque_saturating"][-1] / s["torque_linear_iron"][-1])
    axes[1].set_title(f"Saturation: {gap:.0f}% less torque at 12 A/mm² than a linear model", fontsize=10)
    axes[1].legend(frameon=False, fontsize=8.5); clean(axes[1])
    fig.tight_layout(); fig.savefig(OUT / "fig_validation.png"); plt.close(fig)


def parity(r):
    df = pd.read_csv(ROOT / "results/test_predictions.csv")
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), dpi=130)
    m = r["models"][r["best_model"]]
    for ax, t in zip(axes, NAMES):
        x, y = df[f"true_{t}"], df[f"pred_{t}"]
        ax.scatter(x, y, s=8, alpha=.55, color=BLUE)
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], color="k", lw=.8)
        if t != "B_tooth_p95":
            ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"{NAMES[t]}\nR² {m[t]['r2']:.3f}, median error {m[t]['median_err_pct']:.1f}%", fontsize=9.5)
        ax.set_xlabel("finite elements", fontsize=9); clean(ax)
    axes[0].set_ylabel(r["best_model"])
    fig.suptitle(f"Surrogate vs FEM on {r['n_designs']} designs, each predicted by a model that never saw it "
                 "(5-fold cross-validation)", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "fig1_parity.png"); plt.close(fig)


def learning(r):
    lc = r["learning_curve"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), dpi=130)
    for ax, t in zip(axes, ["torque_mean", "torque_ripple_pct", "cogging_pp"]):
        for name, col in zip(lc, (ORANGE, BLUE)):
            ns = sorted(int(n) for n in lc[name])
            ax.plot(ns, [lc[name][str(n)][t] for n in ns], "o-", color=col, label=name)
        ax.set_xscale("log"); ax.set_xlabel("FEM designs used for training"); ax.set_ylabel("median error (%)")
        ax.set_title(NAMES[t], fontsize=10); clean(ax)
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "fig2_learning_curve.png"); plt.close(fig)


def optimisation(r):
    o = r["optimisation"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), dpi=130)
    pv = pd.DataFrame(o["pred_vs_fem"])
    axes[0].scatter(pv.pred_ripple, pv.fem_ripple, color=BLUE)
    hi = max(pv.pred_ripple.max(), pv.fem_ripple.max()) * 1.1
    axes[0].plot([0, hi], [0, hi], "k", lw=.8)
    axes[0].set_xlabel("ripple predicted by the surrogate (%)"); axes[0].set_ylabel("ripple computed by FEM (%)")
    axes[0].set_title("Shortlist: prediction against FEM", fontsize=10); clean(axes[0])
    b = o["baselines"]; ks = list(b)
    labels = [f"surrogate screening\n+ {o['fem_runs']} FEM runs"] + [k.replace("random_", "random search\n")
                                                                     .replace("_fem_runs", " FEM runs") for k in ks]
    vals = [o["best"]["torque_ripple_pct"] if o["best"] else None] + [b[k]["best_ripple"] for k in ks]
    bars = axes[1].bar(labels, [v or 0 for v in vals], color=[BLUE] + [GREY] * len(ks))
    axes[1].bar_label(bars, labels=[f"{v:.1f}%" if v else "no feasible\ndesign" for v in vals], fontsize=9)
    sp = o["spec"]
    axes[1].set_title(f"Lowest ripple, torque ≥ {sp['torque_min']:.0f} N·m, cogging ≤ {sp['cogging_max']} N·m, "
                      f"J ≤ {sp['J_max']:.0f} A/mm²", fontsize=9.5)
    axes[1].set_ylabel("best torque ripple (%)"); clean(axes[1])
    fig.tight_layout(); fig.savefig(OUT / "fig3_optimisation.png"); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    r = json.loads((ROOT / "results/metrics.json").read_text())
    v = json.loads((ROOT / "results/validation.json").read_text())
    field(); validation(v); parity(r); learning(r)
    if "optimisation" in r:
        optimisation(r)


if __name__ == "__main__":
    main()
