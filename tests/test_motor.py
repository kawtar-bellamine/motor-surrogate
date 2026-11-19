"""Tests of the finite element model and of the surrogate pipeline (a few FEM solves: about 1 minute)."""
import math

import numpy as np
import pandas as pd
import pytest

from motorsur.analytic import coefficients, gap_field, zhu_closed_form_peak_coefficient
from motorsur.dataset_fem import FEATURES, sample
from motorsur.fem import Machine, dataclass_replace, q_axis_phase, solve
from motorsur.train import MLP, featurize, metrics


def test_subdomain_solution_matches_zhu_closed_form():
    Rr, Rm, Rs = 0.036, 0.040, 0.041
    r = (Rm + Rs) / 2
    for k in (1, 3, 5):
        n = 2 * k
        KB = zhu_closed_form_peak_coefficient(Rr, Rm, Rs, 2, 0.8, 1.2, 1.05, k)
        zhu = KB * ((r / Rs) ** (n - 1) * (Rm / Rs) ** (n + 1) + (Rm / r) ** (n + 1))
        _, a, b, d, e = [c for c in coefficients(Rr, Rm, Rs, 2, 0.8, 1.2, 1.05, 2 * k) if c[0] == n][0]
        assert abs(n * (d * r ** n + e * r ** -n) / r - zhu) < 1e-9


def test_fem_matches_exact_slotless_field():
    m = Machine(Q=24, p=2, R=0.04, g=0.001, hm=0.004, slotless=True, iron_mur=1e5)
    s = solve(m, 0.0)
    band = s["_reg"] == "band"; cx, cy = s["_c"]
    Ba, _ = gap_field(np.hypot(cx, cy)[band], np.arctan2(cy, cx)[band], m.Rr, m.R, m.Rs, 2, m.alpha, m.Br, m.mur)
    assert np.linalg.norm(s["_Brad"][band] - Ba) / np.linalg.norm(Ba) < 0.015
    assert abs(s["torque"]) < 1e-3                                   # no slots, no current: no torque


@pytest.fixture(scope="module")
def machine():
    m = Machine(Q=24, p=2, R=0.04, g=0.001, hm=0.004, so=0.3, tw=0.5)
    m.NI = 6e6 * m.slot_area
    return m


def test_maximum_torque_on_the_q_axis(machine):
    g0, s = q_axis_phase(24, 2)
    t = [solve(machine, 0.0, gamma=g0 + s * math.radians(a))["torque"] for a in (-25, 0, 25)]
    assert t[1] > t[0] > 0 and t[1] > t[2] > 0


def test_saturation_makes_torque_sublinear(machine):
    g0, s = q_axis_phase(24, 2)
    t1 = solve(dataclass_replace(machine, NI=machine.NI / 3), 0.0, gamma=g0)["torque"]
    t3 = solve(machine, 0.0, gamma=g0)["torque"]
    assert 1.2 < t3 / t1 < 3.0                                      # less than proportional to the current


def test_sampling_and_features():
    df = pd.DataFrame([dict(m.__dict__, J=m.NI / (1e6 * m.slot_area)) for m in sample(20, 0)])
    assert set(FEATURES) <= set(df.columns) and featurize(df).shape[0] == 20


def test_mlp_fits_a_smooth_function():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (300, 5)); Y = np.column_stack([np.log(2 + np.sin(X[:, 0]) + X[:, 1] ** 2)] * 4)
    m = MLP(hidden=64, epochs=300).fit(X[:240], Y[:240])
    assert metrics(Y[240:], m.predict(X[240:]))["torque_mean"]["median_err_pct"] < 5
