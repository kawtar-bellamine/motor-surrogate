"""2D magnetostatic finite element model of a surface-mounted permanent magnet machine (gmsh + scikit-fem).

Formulation in the magnetic vector potential A_z, first-order triangles:

    -div(nu grad A) = J + curl_z(nu_m Br),      A = 0 on the stator outer surface

Magnets are radially magnetised (B = mu0 mur H + Br, alternating poles). Iron is either linear (validation) or
saturating, with Brauer's analytic reluctivity law nu(B) = k1 exp(k2 B^2) + k3 (k1 = 3.8, k2 = 2.17, k3 = 396.2 m/H:
initial relative permeability about 2,000, saturation above 1.6 T; J. R. Brauer, IEEE Trans. Magn. 11(1), 1975), the
law used in many non-linear FEM papers; the non-linear problem is solved by Newton-Raphson with a line search. Torque by Arkkio's method: the Maxwell stress tensor averaged over an annulus in the
air gap, T = L / (mu0 (r2 - r1)) * integral of r Br Btheta dS, which is far less sensitive to the mesh than a
line integral.

The rotor is re-meshed at every position (a band of small elements in the air gap keeps the mesh error stable).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

MU0 = 4e-7 * math.pi
K1, K2, K3 = 3.8, 2.17, 396.2               # Brauer law, nu = K1 exp(K2 B^2) + K3 (m/H)
SC = 9.0                                    # B^2 above which the reluctivity is held constant (3 T)


@dataclass
class Machine:
    Q: int = 24              # stator slots
    p: int = 2               # pole pairs
    R: float = 0.040         # magnet outer radius, m
    g: float = 0.001         # mechanical air gap, m
    hm: float = 0.004        # magnet thickness, m
    alpha: float = 0.8       # magnet pole-arc ratio
    so: float = 0.3          # slot opening / slot pitch at the bore
    tw: float = 0.5          # tooth width / slot pitch
    Br: float = 1.2          # remanence, T
    mur: float = 1.05        # magnet recoil permeability
    NI: float = 0.0          # ampere-turns per slot (rms)
    psi: float = 0.0         # current angle from the q axis, electrical degrees
    L: float = 0.1           # stack length, m
    slotless: bool = False   # validation case: smooth stator, no slots
    iron_mur: float | None = None   # constant iron permeability (None = saturating law)

    @property
    def Rr(self):
        return self.R - self.hm

    @property
    def Rs(self):
        return self.R + self.g

    @property
    def hs(self):
        return 0.45 * self.R          # slot depth

    @property
    def slot_area(self):
        """Copper area of one slot (the slot body), m^2."""
        pitch = 2 * math.pi / self.Q
        return (1 - self.tw) * pitch * (self.Rs + 0.15 * self.hs) * 0.9 * self.hs

    @property
    def Ro(self):
        return self.Rs + self.hs + 0.30 * self.R


# ---------------------------------------------------------------- mesh
def mesh(m: Machine, pos: float, h_gap: float | None = None):
    """Triangular mesh at mechanical rotor angle pos. Returns nodes (2, N), triangles (3, M), region names per
    triangle and, for slot triangles, the slot index."""
    import gmsh
    gmsh.initialize(["-nt", "1"]); gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("General.NumThreads", 1)
    gmsh.model.add("m")
    occ = gmsh.model.occ
    Rr, R, Rs, Ro, hs = m.Rr, m.R, m.Rs, m.Ro, m.hs
    rm1, rm2 = R + m.g / 3, R + 2 * m.g / 3
    pitch = 2 * math.pi / m.Q
    b0 = m.so * pitch * Rs
    surf = [(2, occ.addDisk(0, 0, 0, Ro, Ro))]
    for r_ in (Rs, rm2, rm1, R, Rr):
        surf.append((2, occ.addDisk(0, 0, 0, r_, r_)))
    o = occ.addPoint(0, 0, 0)
    for k in range(2 * m.p):                                  # magnet edges
        for a_ in (pos + k * math.pi / m.p - m.alpha * math.pi / (2 * m.p),
                   pos + k * math.pi / m.p + m.alpha * math.pi / (2 * m.p)):
            p1 = occ.addPoint(Rr * math.cos(a_), Rr * math.sin(a_), 0)
            p2 = occ.addPoint(R * math.cos(a_), R * math.sin(a_), 0)
            surf.append((1, occ.addLine(p1, p2)))
    if not m.slotless:
        for k in range(m.Q):
            c = pitch * (k + 0.5)
            wb = (1 - m.tw) * pitch * (Rs + 0.15 * hs)            # slot body width (parallel slot)
            s1 = occ.addRectangle(Rs - 0.2 * m.g, -b0 / 2, 0, 0.1 * hs + 0.2 * m.g, b0)
            s2 = occ.addRectangle(Rs + 0.1 * hs, -wb / 2, 0, 0.9 * hs, wb)
            s = occ.fuse([(2, s1)], [(2, s2)])[0]
            occ.rotate(s, 0, 0, 0, 0, 0, 1, c)
            surf += s
    occ.fragment(surf[:1], surf[1:])
    occ.synchronize()
    hg = h_gap or m.g / 3
    hmax = R / 10

    f = gmsh.model.mesh.field
    f.add("MathEval", 1)                                   # element size grows with the distance to the air gap
    f.setString(1, "F", f"Min({hmax}, {hg} + 0.35 * Max(0, Abs(Sqrt(x*x + y*y) - {(R + Rs) / 2}) - {m.g / 2}))")
    f.setAsBackgroundMesh(1)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.model.mesh.generate(2)
    tags, xyz, _ = gmsh.model.mesh.getNodes()
    order = np.argsort(tags)
    lookup = np.empty(int(tags.max()) + 1, dtype=np.int64); lookup[tags[order]] = np.arange(tags.size)[order]
    P = xyz.reshape(-1, 3)[:, :2].T.copy()
    et, _, en = gmsh.model.mesh.getElements(2)
    T = lookup[en[list(et).index(2)].astype(np.int64)].reshape(-1, 3).T
    gmsh.finalize()
    # drop nodes not used by triangles (gmsh may keep some)
    used = np.unique(T); remap = -np.ones(P.shape[1], dtype=np.int64); remap[used] = np.arange(used.size)
    P, T = P[:, used], remap[T]
    reg, slot = classify(m, pos, P[:, T].mean(axis=1))
    return P, T, reg, slot, dict(rm1=rm1, rm2=rm2)


def classify(m: Machine, pos: float, c: np.ndarray):
    x, y = c; r = np.hypot(x, y); a = np.arctan2(y, x)
    reg = np.full(r.size, "stator", dtype=object); slot = -np.ones(r.size, dtype=int)
    reg[r < m.Rr] = "rotor"
    ring = (r >= m.Rr) & (r < m.R)
    k = np.round((a - pos) / (math.pi / m.p)).astype(int)
    d = np.angle(np.exp(1j * (a - pos - k * math.pi / m.p)))
    inside = np.abs(d) < m.alpha * math.pi / (2 * m.p)
    reg[ring & inside & (k % 2 == 0)] = "magnet_n"
    reg[ring & inside & (k % 2 == 1)] = "magnet_s"
    reg[ring & ~inside] = "air"
    reg[(r >= m.R) & (r < m.R + m.g / 3)] = "gap"
    reg[(r >= m.R + m.g / 3) & (r < m.R + 2 * m.g / 3)] = "band"
    reg[(r >= m.R + 2 * m.g / 3) & (r < m.Rs)] = "gap"
    if not m.slotless:
        pitch = 2 * math.pi / m.Q
        ks = np.floor((a % (2 * math.pi)) / pitch).astype(int) % m.Q
        ca = pitch * (ks + 0.5)
        u = r * np.cos(a - ca); v = r * np.sin(a - ca)          # local slot frame
        b0 = m.so * pitch * m.Rs; wb = (1 - m.tw) * pitch * (m.Rs + 0.15 * m.hs)
        opening = (u >= m.Rs - 0.2 * m.g) & (u < m.Rs + 0.1 * m.hs) & (np.abs(v) < b0 / 2)
        body = (u >= m.Rs + 0.1 * m.hs) & (u <= m.Rs + m.hs) & (np.abs(v) < wb / 2)
        st = (r >= m.Rs) & (opening | body)
        reg[st & opening & ~body] = "opening"
        reg[st & body] = "slot"
        slot[st & body] = ks[st & body]
    return reg, slot


# ---------------------------------------------------------------- winding
def winding(Q: int, p: int):
    """Phase (0, 1, 2) and sign of the conductors of each slot: single-layer 60-degree phase belts."""
    e = (p * 2 * np.pi * (np.arange(Q) + 0.5) / Q) % (2 * np.pi)
    belt = np.floor(e / (np.pi / 3)).astype(int)
    table = {0: (0, 1), 1: (2, -1), 2: (1, 1), 3: (0, -1), 4: (2, 1), 5: (1, -1)}
    return np.array([table[b] for b in belt])


def q_axis_phase(Q: int, p: int) -> tuple[float, int]:
    """(gamma0, s): with i_k = cos(gamma - 2 pi k / 3) and gamma = gamma0 + s p pos, the fundamental of the stator
    MMF stays on the q axis of the rotor (first north pole at pos). Computed from the winding, no fitting."""
    w = winding(Q, p)
    th = 2 * np.pi * (np.arange(Q) + 1) / Q                   # the MMF after slot k holds until slot k + 1

    def mmf_angle(gamma):
        i = np.cos(gamma - 2 * np.pi * np.arange(3) / 3)
        F = np.cumsum(w[:, 1] * i[w[:, 0]])                      # MMF after each slot (up to a constant)
        return -np.angle(np.sum((F - F.mean()) * np.exp(-1j * p * th))) / p   # F ~ cos(p th - p a)

    wrap = lambda x: float(np.angle(np.exp(1j * x)))
    a0 = mmf_angle(0.0)
    sgn = 1 if wrap(p * (mmf_angle(0.1) - a0)) > 0 else -1
    return sgn * wrap(p * (-math.pi / (2 * p) - a0)), sgn     # q axis on the side that gives motoring torque


# ---------------------------------------------------------------- solver
def _p1(P, T):
    x, y = P[0, T], P[1, T]
    area = 0.5 * ((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]))
    s = np.sign(area); area = np.abs(area)
    # gradients of the barycentric functions: b_i = (y_j - y_k) / 2A, c_i = (x_k - x_j) / 2A
    b = np.stack([y[1] - y[2], y[2] - y[0], y[0] - y[1]]) * s / (2 * area)
    c = np.stack([x[2] - x[1], x[0] - x[2], x[1] - x[0]]) * s / (2 * area)
    return area, b, c


def solve(m: Machine, pos: float, gamma: float | None = None, h_gap: float | None = None, mesh_data=None,
          tol: float = 1e-3):
    """Field at rotor position pos with phase currents i_k = sqrt(2) NI cos(gamma - 2 pi k / 3). Returns the torque
    (Arkkio), the flux densities per element and the mesh."""
    P, T, reg, slot, geo = mesh_data or mesh(m, pos, h_gap)
    n, ne = P.shape[1], T.shape[1]
    area, b, c = _p1(P, T)
    cx, cy = P[0, T].mean(0), P[1, T].mean(0); rr = np.hypot(cx, cy)
    # sources
    rhs = np.zeros(n)
    if m.NI and gamma is not None and not m.slotless:
        w = winding(m.Q, m.p)
        i = math.sqrt(2) * m.NI * np.cos(gamma - 2 * np.pi * np.arange(3) / 3)
        for k in range(m.Q):
            sel = slot == k
            if sel.any():
                J = w[k, 1] * i[w[k, 0]] / area[sel].sum()
                np.add.at(rhs, T[:, sel], (J * area[sel] / 3)[None, :].repeat(3, 0))
    sgn = np.where(reg == "magnet_n", 1.0, np.where(reg == "magnet_s", -1.0, 0.0))
    mag = sgn != 0
    numag = 1 / (MU0 * m.mur)
    Brx, Bry = sgn * m.Br * cx / rr, sgn * m.Br * cy / rr
    # weak form of curl(nu_m Br): integral nu_m (Brx dv/dy - Bry dv/dx)
    np.add.at(rhs, T[:, mag], (numag * (Brx[mag] * c[:, mag] - Bry[mag] * b[:, mag]) * area[mag]))
    iron = np.isin(reg, ["rotor", "stator"])
    nu = np.where(mag, numag, 1 / MU0).astype(float)
    if m.iron_mur:
        nu[iron] = 1 / (MU0 * m.iron_mur)
    else:
        nu[iron] = 1 / (MU0 * 50.0)            # initial guess: low permeability, so no element starts deep in saturation
    G = np.einsum("ie,je->ije", b, b) + np.einsum("ie,je->ije", c, c)   # grad.grad per element
    rows = np.repeat(T[:, None, :], 3, 1).ravel(); cols = np.repeat(T[None, :, :], 3, 0).ravel()
    fixed = np.hypot(P[0], P[1]) > m.Ro * (1 - 1e-6)
    free = ~fixed
    A = np.zeros(n)

    def nu_of(A_):
        gx, gy = (A_[T] * b).sum(0), (A_[T] * c).sum(0)
        B2 = np.minimum(gx ** 2 + gy ** 2, SC)                # constant reluctivity beyond 3 T (keeps it convex)
        out = nu.copy()
        out[iron] = K1 * np.exp(K2 * B2[iron]) + K3          # absolute reluctivity, m/H
        return out, gx, gy, B2

    def assemble(nu_):
        return sp.coo_matrix((((nu_ * area)[None, None, :] * G).ravel(), (rows, cols)), shape=(n, n)).tocsr()

    nu_lin = nu.copy()

    def energy(A_):
        """Magnetic co-energy functional whose minimum is the solution (convex for a monotone B-H law)."""
        gx, gy = (A_[T] * b).sum(0), (A_[T] * c).sum(0)
        s2 = gx ** 2 + gy ** 2
        B2 = np.minimum(s2, SC)
        w_iron = 0.5 * (K1 / K2 * (np.exp(K2 * B2) - 1) + K3 * B2 + (K1 * np.exp(K2 * SC) + K3) * (s2 - B2))
        w = np.where(iron, w_iron, 0.5 * nu_lin * s2)
        return float(np.sum(w * area) - rhs @ A_)

    it = 0
    if m.iron_mur:
        A[free] = spla.spsolve(assemble(nu)[free][:, free].tocsc(), rhs[free])
    else:
        # start from the linear solution (iron mu_r = 1000), then Newton-Raphson with a backtracking line search
        A[free] = spla.spsolve(assemble(nu)[free][:, free].tocsc(), rhs[free])
        norm0 = np.linalg.norm(rhs[free])
        for it in range(1, 40):
            nu_c, gx, gy, B2 = nu_of(A)
            K = assemble(nu_c)
            R_ = K @ A - rhs
            rn = np.linalg.norm(R_[free])
            if rn <= tol * norm0:
                break
            dnu = np.where(iron & (gx ** 2 + gy ** 2 < SC), K1 * K2 * np.exp(K2 * B2), 0.0)
            v = gx[None] * b + gy[None] * c                     # grad phi_i . grad A
            Jn = 2 * (dnu * area)[None, None, :] * v[:, None, :] * v[None, :, :]
            Jac = K + sp.coo_matrix((Jn.ravel(), (rows, cols)), shape=(n, n)).tocsr()
            dA = np.zeros(n)
            dA[free] = spla.spsolve(Jac[free][:, free].tocsc(), -R_[free])
            step, e0, slope = 1.0, energy(A), float(R_ @ dA)
            while step > 1 / 256 and energy(A + step * dA) > e0 + 1e-4 * step * slope:
                step /= 2
            if not np.isfinite(dA).all() or energy(A + step * dA) > e0:
                raise RuntimeError("Newton iterations did not converge")
            A += step * dA
        else:
            raise RuntimeError("Newton iterations did not converge in 40 steps")
        nu = nu_of(A)[0]
    gx, gy = (A[T] * b).sum(0), (A[T] * c).sum(0)
    Bx, By = gy, -gx
    Brad = (Bx * cx + By * cy) / rr; Bth = (-Bx * cy + By * cx) / rr
    band = reg == "band"
    torque = m.L / (MU0 * (geo["rm2"] - geo["rm1"])) * float(np.sum(rr[band] * Brad[band] * Bth[band] * area[band]))
    Bmag = np.hypot(Bx, By)
    return dict(torque=torque, iterations=it, nodes=n, elements=ne, B_iron_max=float(Bmag[iron].max()),
                B_tooth_p95=float(np.percentile(Bmag[reg == "stator"], 95)) if (reg == "stator").any() else 0.0,
                _A=A, _P=P, _T=T, _reg=reg, _Brad=Brad, _Bth=Bth, _c=(cx, cy))


# ---------------------------------------------------------------- machine-level quantities
def lcm(a, b):
    return a * b // math.gcd(a, b)


def evaluate(m: Machine, n_cog: int = 8, n_load: int = 6, h_gap: float | None = None) -> dict:
    """Cogging torque over one cogging period (zero current) and torque under load over one sixth of an electrical
    period (the period of the 6th-harmonic ripple of a three-phase machine), with currents on the q axis + psi."""
    period_cog = 2 * math.pi / lcm(m.Q, 2 * m.p)
    cog = [solve(dataclass_replace(m, NI=0.0), period_cog * k / n_cog, h_gap=h_gap)["torque"] for k in range(n_cog)]
    g0, sgn = q_axis_phase(m.Q, m.p)
    g0 += sgn * math.radians(m.psi)
    period_load = 2 * math.pi / (6 * m.p)
    load, info = [], None
    for k in range(n_load):
        pos = period_load * k / n_load
        r = solve(m, pos, gamma=g0 + sgn * m.p * pos, h_gap=h_gap)
        load.append(r["torque"]); info = info or r
    load = np.array(load); cog = np.array(cog)
    Tm = float(load.mean())
    return dict(torque_mean=Tm, torque_ripple_pct=float(100 * (load.max() - load.min()) / max(abs(Tm), 1e-9)),
                cogging_pp=float(cog.max() - cog.min()), B_tooth_p95=info["B_tooth_p95"],
                B_iron_max=info["B_iron_max"], elements=info["elements"], _load=load, _cog=cog)


def dataclass_replace(m: Machine, **kw) -> Machine:
    d = asdict(m); d.update(kw); return Machine(**d)
