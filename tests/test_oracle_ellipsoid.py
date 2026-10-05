"""Orakel für die Ellipsoid-Methode: (1) das Update eines tiefen Schnitts gegen das numerisch bestimmte kleinste Ellipsoid derselben Familie (rotationssymmetrisch um die Schnittnormale, durch den
Extrempunkt und den Rand der Schnittfläche), (2) die ganze Iterationsfolge (Mittelpunkte, Iterationszahl) gegen eine eigene Implementierung in der Wurzelform P = B B^T (Grötschel-Lovász-Schrijver),
(3) der Inkugelradius gegen ein eigenes LP (HiGHS)."""

import math

import numpy as np
import pytest

import ell_ellipsoid as E
import ell_evaluation as V
import ell_scenario as S

scipy_opt = pytest.importorskip("scipy.optimize")


def test_cut_update_is_the_minimum_volume_ellipsoid_of_the_family():
    rng = np.random.default_rng(0)
    for n in (2, 3, 5, 8):
        for alpha in [0.0, 0.2, 0.6, 0.9]:
            M = rng.normal(size=(n, n))
            P = M @ M.T + 0.3 * np.eye(n)
            a, x = rng.normal(size=n), rng.normal(size=n)
            s = math.sqrt(a @ P @ a)
            xn, Pn, _al, dlog = E._cut_update(x, P, a, float(a @ x - alpha * s), "deep", n)
            L = np.linalg.cholesky(P)
            v = L.T @ a / s                                                       # Schnittnormale in Koordinaten, in denen E die Einheitskugel ist

            def log_vol(t):                                                       # Extrempunkt u1 = -1 fest (h = 1 + t), Rand der Schnittfläche u1 = -alpha, Radius sqrt(1 - alpha^2)
                h = 1.0 + t
                denom = 1.0 - ((-alpha - t) / h) ** 2
                if h <= 0 or denom <= 1e-12:
                    return 1e9
                return math.log(h) + (n - 1) * 0.5 * math.log((1 - alpha ** 2) / denom)

            hi = -alpha if alpha > 0 else 1 - 1e-9
            t = scipy_opt.minimize_scalar(log_vol, bounds=(-1 + 1e-6, hi - 1e-9), method="bounded", options={"xatol": 1e-13}).x
            h = 1.0 + t
            r2 = (1 - alpha ** 2) / (1.0 - ((-alpha - t) / h) ** 2)
            P_num = L @ (h * h * np.outer(v, v) + r2 * (np.eye(n) - np.outer(v, v))) @ L.T
            assert xn == pytest.approx(x + (L @ v) * t, abs=1e-5)
            assert np.allclose(Pn, P_num, rtol=1e-4, atol=1e-5)
            assert dlog == pytest.approx(np.linalg.slogdet(Pn)[1] - np.linalg.slogdet(P)[1], abs=1e-9)
            if alpha == 0.0:
                assert 0.5 * dlog == pytest.approx(E.volume_log_ratio_central(n), abs=1e-9) and 0.5 * dlog <= -1.0 / (2 * (n + 1)) + 1e-12


def _root_form_run(inst, eps=1e-6, max_iter=3000, eq_tol=1e-3):
    """Zentrale Schnitte, erste verletzte Zeile, Objektivschnitt; Wurzelform B mit P = B B^T (Update ohne Symmetrisierung, ohne P)."""
    Am, b, c = inst.arrays()
    n = inst.n
    rows, rhs = [], []
    for i, s in enumerate(inst.senses):
        if s == S.LE:
            rows.append(Am[i]), rhs.append(b[i])
        elif s == S.GE:
            rows.append(-Am[i]), rhs.append(-b[i])
        else:
            d = eq_tol * max(1.0, abs(b[i]))
            rows.append(Am[i]), rhs.append(b[i] + d), rows.append(-Am[i]), rhs.append(-b[i] + d)
    for j in range(n):
        e = np.zeros(n)
        e[j] = -1.0
        rows.append(e), rhs.append(0.0)
    G, h = np.array(rows), np.array(rhs)
    U = np.full(n, np.inf)
    for i, s in enumerate(inst.senses):
        if s == S.GE or (Am[i] < 0).any():
            continue
        cap = b[i] + (eq_tol * max(1.0, abs(b[i])) if s == S.EQ else 0.0)
        for j in range(n):
            if Am[i, j] > 0:
                U[j] = min(U[j], cap / Am[i, j])
    x, B = U / 2, 0.5 * np.linalg.norm(U) * np.eye(n)
    best, have, centres = -np.inf, False, []
    for it in range(1, max_iter + 1):
        v = G @ x - h
        viol = [k for k in range(len(h)) if v[k] > 1e-12 * (1 + abs(h[k]))]
        if viol:
            a = G[viol[0]]
        else:
            if c @ x > best:
                best, have = float(c @ x), True
            a = -c
        centres.append(x.copy())
        if have and c @ x + math.sqrt(c @ B @ B.T @ c) - best <= eps * (1 + abs(best)):
            return it, best, centres
        bb = B.T @ a
        bb /= np.linalg.norm(bb)
        x = x - (B @ bb) / (n + 1)
        B = (n / math.sqrt(n * n - 1)) * (B - (1 - math.sqrt((n - 1) / (n + 1))) * np.outer(B @ bb, bb))
    return max_iter, best, centres


def test_iteration_sequence_equals_an_independent_root_form_implementation():
    insts = [S.textbook_instance(), S.centre_instance(), S.degenerate_instance(), S.klee_minty_instance(3)]
    insts += [S.generate("random", n, n, 0.5, sd) for n in (2, 3, 4) for sd in range(3)] + [S.generate("mixed", 5, 4, 0.5, sd) for sd in range(3)]
    for inst in insts:
        it, best, centres = _root_form_run(inst)
        r = E.ellipsoid(inst, eps=1e-6, cut="central", select="first", keep=True, keep_mats=False)
        assert r.status == "optimal" and r.iterations == it, inst.kind
        assert np.abs(np.array(centres) - np.array(r.centres)).max() < 1e-6
        assert r.obj == pytest.approx(best, abs=1e-6 * (1 + abs(best)))


def test_inscribed_radius_equals_a_chebyshev_lp_and_the_ball_fits():
    linprog = scipy_opt.linprog
    for sd in range(25):
        for m, n in ((3, 3), (5, 4)):
            inst = S.generate("random", m, n, 0.5, sd)
            Am, b, _c = inst.arrays()
            ua = [list(Am[i]) + [np.linalg.norm(Am[i])] for i in range(m)] + [[-1.0 if j == k else 0.0 for j in range(n)] + [1.0] for k in range(n)]
            h = linprog([0] * n + [-1], A_ub=np.array(ua), b_ub=list(b) + [0] * n, bounds=[(None, None)] * n + [(0, None)], method="highs")
            assert V.inscribed_radius(inst) == pytest.approx(h.x[-1], rel=1e-6, abs=1e-8)
            pts = np.random.default_rng(sd).normal(size=(30, n))
            pts = h.x[:n] + h.x[-1] * 0.999 * pts / np.linalg.norm(pts, axis=1)[:, None]
            assert (pts @ Am.T <= b + 1e-9).all() and (pts >= -1e-9).all()
