"""Sample machine designs and run the finite element model on each one (the expensive step the surrogate replaces).
Rows are appended to data/designs_fem.csv as they finish, so an interrupted run resumes where it stopped."""
from __future__ import annotations

import csv
import sys
import time
from dataclasses import asdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .fem import Machine, evaluate

COMBOS = [(12, 2), (24, 2), (36, 2), (18, 3), (36, 3), (24, 4), (48, 4), (48, 8)]    # integer slots per pole/phase
RANGES = dict(R=(0.030, 0.060), g=(0.0005, 0.0015), hm=(0.002, 0.006), alpha=(0.60, 0.95), so=(0.10, 0.50),
              tw=(0.35, 0.65), Br=(1.10, 1.30), J=(2.0, 12.0), psi=(-30.0, 30.0))   # J: A/mm2 rms in the slot
FEATURES = ["Q", "p", *RANGES]
TARGETS = ["torque_mean", "torque_ripple_pct", "cogging_pp", "B_tooth_p95"]
OUT = Path(__file__).resolve().parents[1] / "data/designs_fem.csv"


def sample(n: int, seed: int) -> list[Machine]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        Q, p = COMBOS[rng.integers(len(COMBOS))]
        kw = {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in RANGES.items()}
        J = kw.pop("J")
        m = Machine(Q=Q, p=p, **kw)
        m.NI = J * 1e6 * m.slot_area
        out.append(m)
    return out


def run_one(args):
    i, m = args
    t = time.perf_counter()
    try:
        r = evaluate(m)
        r["J"] = m.NI / (1e6 * m.slot_area)
    except Exception as e:  # noqa: BLE001  (a failed mesh is recorded, not hidden)
        return dict(id=i, **asdict(m), J=m.NI / (1e6 * m.slot_area), error=str(e)[:200])
    row = dict(id=i, **asdict(m), **{k: v for k, v in r.items() if not k.startswith("_")},
               fem_seconds=round(time.perf_counter() - t, 2), error="")
    row["torque_curve"] = " ".join(f"{x:.4f}" for x in r["_load"])
    row["cogging_curve"] = " ".join(f"{x:.4f}" for x in r["_cog"])
    return row


def build(n: int, seed: int = 0, workers: int = 2, out: Path = OUT):
    out.parent.mkdir(exist_ok=True)
    done = set()
    if out.exists():
        with open(out) as f:
            done = {int(r["id"]) for r in csv.DictReader(f)}
    todo = [(i, m) for i, m in enumerate(sample(n, seed)) if i not in done]
    fields = None
    with Pool(workers) as pool, open(out, "a", newline="") as f:
        for row in pool.imap_unordered(run_one, todo):
            if fields is None:
                fields = ["id", *asdict(Machine()), "J", "torque_mean", "torque_ripple_pct", "cogging_pp",
                          "B_tooth_p95", "B_iron_max", "elements", "fem_seconds", "error", "torque_curve", "cogging_curve"]
                w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                if not done:
                    w.writeheader()
            w.writerow(row); f.flush()
            print(row["id"], row.get("torque_mean"), row.get("fem_seconds"), row.get("error"), flush=True)


if __name__ == "__main__":
    build(int(sys.argv[1]) if len(sys.argv) > 1 else 400)
