"""Exact 2D air-gap field of a slotless surface-mounted PM machine with radially magnetised magnets and infinitely
permeable iron: the subdomain (harmonic) solution of Zhu, Howe, Bolte and Ackermann, "Instantaneous magnetic field
distribution in brushless permanent magnet DC motors, Part I: open-circuit field", IEEE Trans. Magn. 29(1), 1993.

Each harmonic n of the magnetisation solves A = (a r^n + b r^-n + c_p r) sin(n theta) in the magnet and
A = (d r^n + e r^-n) sin(n theta) in the air gap, with H_theta = 0 on the iron surfaces and continuity of B_r and
H_theta at the magnet surface. The 4 x 4 system is solved per harmonic, which is the closed form of the paper written
as a linear system (and checked against it in the tests).
"""
from __future__ import annotations

import math

import numpy as np

MU0 = 4e-7 * math.pi


def coefficients(Rr, Rm, Rs, p, alpha, Br, mur, nmax=99, scale=1.0):
    """Per harmonic: (n, a, b, d, e). Radii may be scaled (divided by `scale`): the flux density only depends on
    their ratios, and scaling keeps r^(+-n) finite for high harmonics."""
    Rr, Rm, Rs = Rr / scale, Rm / scale, Rs / scale
    out = []
    for k in range(1, nmax + 1, 2):
        n = k * p
        M = 2 * (Br / MU0) * alpha * math.sin(k * math.pi * alpha / 2) / (k * math.pi * alpha / 2)   # M_n
        cp = MU0 * n * M / (n * n - 1)                       # particular solution in the magnet, A_p = cp r
        # unknowns x = [a, b, d, e]; A' = n a r^(n-1) - n b r^(-n-1) + cp
        Mx = np.zeros((4, 4)); y = np.zeros(4)
        Mx[0] = [n * Rr ** (n - 1), -n * Rr ** (-n - 1), 0, 0]; y[0] = -cp            # H_theta = 0 at Rr
        Mx[1] = [0, 0, n * Rs ** (n - 1), -n * Rs ** (-n - 1)]                          # H_theta = 0 at Rs
        Mx[2] = [Rm ** n, Rm ** -n, -Rm ** n, -Rm ** -n]; y[2] = -cp * Rm               # A continuous (B_r)
        # H_theta continuous: (B_theta - mu0 M_theta)/(mu0 mur) = B_theta/mu0, M_theta = 0 -> A'_m = mur A'_g
        Mx[3] = [n * Rm ** (n - 1), -n * Rm ** (-n - 1), -mur * n * Rm ** (n - 1), mur * n * Rm ** (-n - 1)]
        y[3] = -cp
        out.append((n, *np.linalg.solve(Mx, y)))
    return out


def gap_field(r, theta, Rr, Rm, Rs, p, alpha, Br, mur, nmax=99):
    """Radial and tangential flux density in the air gap at radius r, angles theta (north pole centred on 0)."""
    Brad = np.zeros_like(theta); Bth = np.zeros_like(theta)
    r = np.asarray(r) / Rs
    for n, a, b, d, e in coefficients(Rr, Rm, Rs, p, alpha, Br, mur, nmax, scale=Rs):
        # A = f(r) sin(n theta): B_r = (1/r) dA/dtheta = n f / r cos, B_theta = -dA/dr = -f' sin
        f = d * r ** n + e * r ** -n
        fp = n * d * r ** (n - 1) - n * e * r ** (-n - 1)
        Brad += n * f / r * np.cos(n * theta)
        Bth += -fp * np.sin(n * theta)
    return Brad, Bth


def zhu_closed_form_peak_coefficient(Rr, Rm, Rs, p, alpha, Br, mur, k=1):
    """K_B(n) of Zhu et al. (1993), eq. for internal rotor, radial magnetisation, np != 1, for cross-checking."""
    n = k * p
    M = 2 * (Br / MU0) * alpha * math.sin(k * math.pi * alpha / 2) / (k * math.pi * alpha / 2)
    num = (n - 1) + 2 * (Rr / Rm) ** (n + 1) - (n + 1) * (Rr / Rm) ** (2 * n)
    den = (mur + 1) / mur * (1 - (Rr / Rs) ** (2 * n)) - (mur - 1) / mur * ((Rm / Rs) ** (2 * n) - (Rr / Rm) ** (2 * n))
    KB = MU0 * M / mur * n / (n * n - 1) * num / den
    return KB      # B_r(r) = KB [ (r/Rs)^(n-1) (Rm/Rs)^(n+1) + (Rm/r)^(n+1) ] cos(n theta)
