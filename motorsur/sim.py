"""Semi-analytical model of a surface-mounted permanent magnet synchronous machine (SPMSM).

Air-gap field = permeance x MMF (magnets + three-phase distributed winding), with the relative permeance of the
slot openings from Zhu and Howe (1993). From the field:
  - electromagnetic torque by the co-energy method at constant currents, over one electrical period,
  - cogging torque (same computation with zero current),
  - radial magnetic pressure B^2 / (2 mu0) and its spatial harmonics (force waves, the source of magnetic noise).
Linear magnetic materials, no saturation, no end effects: a fast proxy of a finite element simulation, used here
to generate the training data of the surrogate model.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

MU0 = 4e-7 * np.pi
NTH = 2048          # points around the air gap
NPOS = 120          # rotor positions over one electrical period
NHARM = 41          # magnet MMF harmonics kept (odd)
COMBOS = [(12, 2), (24, 2), (36, 2), (24, 4), (36, 3), (48, 4), (36, 6), (48, 8)]   # (slots, pole pairs), q integer


@dataclass
class Design:
    Q: int = 24              # stator slots
    p: int = 2               # pole pairs
    R: float = 0.060         # rotor outer radius (magnet surface), m
    g: float = 0.001         # mechanical air gap, m
    hm: float = 0.004        # magnet thickness, m
    alpha: float = 0.8       # magnet pole-arc ratio
    so: float = 0.3          # slot opening / slot pitch
    Br: float = 1.2          # magnet remanence, T
    L: float = 0.1           # stack length, m
    I: float = 20.0          # phase current, A rms
    psi: float = 0.0         # current angle from the q axis, electrical degrees
    Ns: int = 20             # conductors per slot
    mur: float = 1.05


def _winding(Q: int, p: int, th: np.ndarray) -> np.ndarray:
    """Winding functions of phases A, B, C (60 degree phase belts, single layer equivalent), zero mean."""
    slot = 2 * np.pi * (np.arange(Q) + 0.5) / Q
    e = (p * slot) % (2 * np.pi)
    belt = np.floor(e / (np.pi / 3)).astype(int)            # 0..5 : A+, C-, B+, A-, C+, B-
    sign = {0: ("A", 1), 1: ("C", -1), 2: ("B", 1), 3: ("A", -1), 4: ("C", 1), 5: ("B", -1)}
    W = np.zeros((3, th.size))
    for k in range(Q):
        ph, s = sign[belt[k]]
        i = "ABC".index(ph)
        W[i] += s * (th >= slot[k])                         # cumulative conductor count
    W -= W.mean(axis=1, keepdims=True)
    return W


def _relative_permeance(d: Design, th: np.ndarray, g_eff: float) -> np.ndarray:
    Rs = d.R + d.g
    pitch = 2 * np.pi / d.Q
    b0 = d.so * pitch * Rs                                  # slot opening width, m
    th0 = b0 / Rs
    beta = 0.5 * (1 - 1 / np.sqrt(1 + (b0 / (2 * g_eff)) ** 2))
    lam = np.ones_like(th)
    for k in range(d.Q):
        c = 2 * np.pi * k / d.Q
        x = np.angle(np.exp(1j * (th - c)))                 # distance to the slot centre, wrapped
        m = np.abs(x) <= 0.8 * th0
        lam[m] = 1 - beta - beta * np.cos(np.pi * x[m] / (0.8 * th0))
    return lam


def _magnet_mmf_coeffs(d: Design, F0: float):
    n = np.arange(1, NHARM + 1, 2)
    # square wave of width alpha*pi per pole, amplitude F0 (Fourier coefficients of the electrical waveform)
    a = 4 * F0 / (n * np.pi) * np.sin(n * d.alpha * np.pi / 2)
    return n, a


def simulate(d: Design, with_currents: bool = True) -> dict:
    th = np.linspace(0, 2 * np.pi, NTH, endpoint=False)
    g_eff = d.g + d.hm / d.mur
    F0 = d.Br * d.hm / (MU0 * d.mur)
    lam = _relative_permeance(d, th, g_eff)
    Lam = MU0 * lam / g_eff
    W = _winding(d.Q, d.p, th) * d.Ns            # MMF jump at a slot = conductors x current
    n, a = _magnet_mmf_coeffs(d, F0)
    pos = np.linspace(0, 2 * np.pi / d.p, NPOS, endpoint=False)          # mechanical rotor angle
    Rg = d.R + d.g / 2
    # magnet MMF and its angular derivative at every rotor position (spectral)
    E = d.p * (th[None, :] - pos[:, None])                                # electrical angle seen by the magnets
    Fm = np.einsum("k,kij->ij", a, np.cos(n[:, None, None] * E[None]))
    dFm = -np.einsum("k,kij->ij", a * n * d.p, np.sin(n[:, None, None] * E[None]))   # d/dtheta
    out = {}
    for tag, cur in (("load", with_currents), ("cog", False)):
        if cur:
            gamma = d.p * pos + np.pi / 2 + np.deg2rad(d.psi) + PHASE_OFFSET[(d.Q, d.p)]
            ia = np.sqrt(2) * d.I * np.cos(gamma)
            ib = np.sqrt(2) * d.I * np.cos(gamma - 2 * np.pi / 3)
            ic = np.sqrt(2) * d.I * np.cos(gamma + 2 * np.pi / 3)
            Fs = ia[:, None] * W[0] + ib[:, None] * W[1] + ic[:, None] * W[2]
        else:
            Fs = 0.0
        F = Fm + Fs
        # co-energy method at constant current: T = -L Rg  int Lam F dFm/dtheta dtheta
        T = -d.L * Rg * np.sum(Lam[None] * F * dFm, axis=1) * (2 * np.pi / NTH)
        out[tag] = (T, F)
    T, F = out["load"]
    B = Lam[None] * F                                                    # radial flux density, T
    P = B ** 2 / (2 * MU0)                                               # radial pressure, Pa
    S = np.abs(np.fft.rfft(P, axis=1)) / NTH * 2                          # spatial harmonics at each position
    amp = S.mean(axis=0)
    orders = np.arange(amp.size)
    cand = (orders >= 1) & (orders <= 4 * d.p) & (orders != 2 * d.p)
    r_dom = int(orders[cand][np.argmax(amp[cand])])
    Tcog = out["cog"][0]
    Tm = float(T.mean())
    return dict(
        torque_mean=Tm,
        torque_ripple_pct=float(100 * (T.max() - T.min()) / max(abs(Tm), 1e-9)),
        cogging_pp=float(Tcog.max() - Tcog.min()),
        B_gap_peak=float(np.abs(B).max()),
        force_wave_kpa=float(amp[r_dom] / 1e3),
        force_wave_order=r_dom,
        _T=T, _Tcog=Tcog,
    )


def _calibrate() -> dict:
    """Current phase offset that puts psi = 0 on the q axis (maximum torque per ampere), per slot/pole combination."""
    out = {}
    global PHASE_OFFSET
    for Q, p in COMBOS:
        best = (-np.inf, 0.0)
        for off in np.deg2rad(np.arange(0, 360, 5)):
            PHASE_OFFSET = {(Q, p): off}
            t = simulate(Design(Q=Q, p=p, so=0.01, I=20))["torque_mean"]
            best = max(best, (t, off))
        out[(Q, p)] = best[1]
    return out


PHASE_OFFSET: dict = {}
PHASE_OFFSET = _calibrate()


def design_from_row(row: dict) -> Design:
    return Design(**{k: row[k] for k in asdict(Design()) if k in row})
