"""How far can the finite element model be trusted? Four checks, written to results/validation.json.

1. Slotless machine, linear iron: FEM field in the air gap against the exact solution of Zhu et al. (1993).
2. Mesh convergence on slotted machines: the same designs with air-gap elements twice smaller.
3. Cogging sampling: 8 rotor positions per cogging period against 24.
4. Saturation: torque against current density with saturating and with linear iron, and the semi-analytical model
   the first version of this project used (linear, slot openings as permeance), against FEM on the same designs.
"""
from __future__ import annotations

import json
import math
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from .analytic import gap_field
from .fem import Machine, dataclass_replace, evaluate, q_axis_phase, solve

ROOT = Path(__file__).resolve().parents[1]


def check_slotless():
    rows = []
    for p, hm, g in ((2, 0.004, 0.001), (4, 0.003, 0.0007)):
        m = Machine(Q=24, p=p, R=0.04, g=g, hm=hm, alpha=0.8, slotless=True, iron_mur=1e5)
        for h in (g / 3, g / 6):
            r = solve(m, 0.0, h_gap=h)
            band = r["_reg"] == "band"; cx, cy = r["_c"]
            rr, th = np.hypot(cx, cy)[band], np.arctan2(cy, cx)[band]
            Ba, Bt = gap_field(rr, th, m.Rr, m.R, m.Rs, m.p, m.alpha, m.Br, m.mur)
            rows.append(dict(p=p, hm_mm=hm * 1e3, gap_mm=g * 1e3, h_gap_mm=round(h * 1e3, 3), elements=r["elements"],
                             err_Br_pct=round(100 * np.linalg.norm(r["_Brad"][band] - Ba) / np.linalg.norm(Ba), 3),
                             err_Btheta_pct=round(100 * np.linalg.norm(r["_Bth"][band] - Bt) / np.linalg.norm(Ba), 3),
                             Br_peak_fem=round(float(np.abs(r["_Brad"][band]).max()), 4),
                             Br_peak_exact=round(float(np.abs(Ba).max()), 4),
                             torque_should_be_0=round(r["torque"], 6)))
    return rows


def _eval(args):
    m, h, n_cog = args
    r = evaluate(m, n_cog=n_cog, h_gap=h)
    return {k: v for k, v in r.items() if not k.startswith("_")}


def designs(n, seed):
    from .dataset_fem import sample
    return sample(n, seed)


def check_mesh(pool):
    ms = designs(4, 123)
    coarse = pool.map(_eval, [(m, None, 8) for m in ms])
    fine = pool.map(_eval, [(m, m.g / 6, 8) for m in ms])
    out = []
    for m, c, f in zip(ms, coarse, fine):
        out.append(dict(Q=m.Q, p=m.p, elements=(c["elements"], f["elements"]),
                        **{k: (round(c[k], 4), round(f[k], 4), round(100 * (c[k] - f[k]) / abs(f[k]), 2))
                           for k in ("torque_mean", "torque_ripple_pct", "cogging_pp")}))
    return out


def check_cogging_sampling(pool):
    ms = [dataclass_replace(m, NI=0.0) for m in designs(3, 7)]
    a = pool.map(_eval, [(m, None, 8) for m in ms]); b = pool.map(_eval, [(m, None, 24) for m in ms])
    return [dict(Q=m.Q, p=m.p, cogging_8=round(x["cogging_pp"], 4), cogging_24=round(y["cogging_pp"], 4),
                 diff_pct=round(100 * (x["cogging_pp"] - y["cogging_pp"]) / y["cogging_pp"], 2))
            for m, x, y in zip(ms, a, b)]


def _torque_at(args):
    m, = args
    g0, s = q_axis_phase(m.Q, m.p)
    return float(np.mean([solve(m, pos, gamma=g0 + s * m.p * pos)["torque"]
                          for pos in np.linspace(0, 2 * math.pi / (6 * m.p), 3, endpoint=False)]))


def check_saturation(pool):
    base = Machine(Q=36, p=3, R=0.04, g=0.0008, hm=0.004, alpha=0.8, so=0.25, tw=0.45)
    Js = [1, 2, 4, 6, 8, 10, 12]
    sat = pool.map(_torque_at, [(dataclass_replace(base, NI=J * 1e6 * base.slot_area),) for J in Js])
    lin = pool.map(_torque_at, [(dataclass_replace(base, NI=J * 1e6 * base.slot_area, iron_mur=2000.0),) for J in Js])
    return dict(design="36 slots, 6 poles, R 40 mm, gap 0.8 mm, stack 0.1 m", J_A_per_mm2=Js,
                torque_saturating=[round(x, 3) for x in sat], torque_linear_iron=[round(x, 3) for x in lin])


def check_old_model():
    """The first version's semi-analytical simulator against FEM on dataset designs it can represent."""
    from .sim import COMBOS as OLD, Design, simulate
    df = pd.read_csv(ROOT / "data/designs_fem.csv")
    df = df[(df["error"].isna()) & df.apply(lambda r: (int(r.Q), int(r.p)) in OLD, axis=1)].head(60)
    rows = []
    for _, r in df.iterrows():
        d = Design(Q=int(r.Q), p=int(r.p), R=r.R, g=r.g, hm=r.hm, alpha=r.alpha, so=r.so, Br=r.Br, L=0.1,
                   I=r.NI / 20, psi=r.psi, Ns=20)
        s = simulate(d)
        rows.append(dict(J=r.J, fem_torque=r.torque_mean, old_torque=s["torque_mean"], fem_cog=r.cogging_pp,
                         old_cog=s["cogging_pp"], fem_ripple=r.torque_ripple_pct, old_ripple=s["torque_ripple_pct"]))
    d = pd.DataFrame(rows)
    err = lambda a, b: float(100 * np.median(np.abs(d[a] - d[b]) / np.abs(d[b])))
    hi, lo = d[d.J >= 8], d[d.J <= 4]
    return dict(n=len(d), median_err_torque_pct=round(err("old_torque", "fem_torque"), 1),
                median_err_cogging_pct=round(err("old_cog", "fem_cog"), 1),
                median_err_ripple_pct=round(err("old_ripple", "fem_ripple"), 1),
                torque_ratio_old_over_fem_J_le_4=round(float((lo.old_torque / lo.fem_torque).median()), 3),
                torque_ratio_old_over_fem_J_ge_8=round(float((hi.old_torque / hi.fem_torque).median()), 3))


def main():
    res = {"slotless_vs_exact": check_slotless()}
    print(json.dumps(res, indent=1), flush=True)
    with Pool(2) as pool:
        res["mesh_convergence"] = check_mesh(pool); print(res["mesh_convergence"], flush=True)
        res["cogging_sampling"] = check_cogging_sampling(pool); print(res["cogging_sampling"], flush=True)
        res["saturation"] = check_saturation(pool); print(res["saturation"], flush=True)
    res["old_semi_analytical_vs_fem"] = check_old_model(); print(res["old_semi_analytical_vs_fem"], flush=True)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/validation.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
