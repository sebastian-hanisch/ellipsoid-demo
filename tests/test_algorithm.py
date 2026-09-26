"""Korrektheitskette: Volumenformel, Invariante "Optimum liegt in jeder Ellipse", Optimum gegen HiGHS und Simplex, Unzulässig/Unbeschränkt, Theorie-Schranke, Gleichungen, Sonderfälle, Buchführung."""

import math
import random

import numpy as np
import pytest

import ell_algorithm as A
import ell_ellipsoid as E
import ell_scenario as S
from tests.test_scenario import _highs, reference_status


def _custom(rows, b, c, senses):
    n = len(c)
    return S.Instance(tuple(tuple(float(v) for v in r) for r in rows), tuple(float(v) for v in b), tuple(float(v) for v in c), tuple(senses), tuple(f"x{j}" for j in range(n)), tuple(f"r{i}" for i in range(len(b))), "custom")


def _instances(seeds=8):
    yield S.textbook_instance()
    yield S.centre_instance()
    yield S.degenerate_instance()
    for seed in range(seeds):
        yield S.generate("random", 5, 6, 0.5, seed)
        yield S.generate("random", 8, 4, 0.5, seed)
        yield S.generate("random", 3, 2, 0.5, seed)


def _opt_point(inst):
    h = _highs(inst)
    assert h.status == 0
    return np.array(h.x), -h.fun


def _quad(x_star, x, P):
    d = x_star - x
    return float(d @ np.linalg.solve(P, d))


# --- 1. Update-Formeln ---------------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("n", range(2, 21))
def test_central_cut_volume_ratio_equals_the_closed_form_and_beats_the_bound(n):
    rng = np.random.default_rng(n)
    M = rng.normal(size=(n, n))
    P = M @ M.T + n * np.eye(n)
    x = rng.normal(size=n)
    a = rng.normal(size=n)
    x2, P2, alpha, dlog = E._cut_update(x, P, a, float(a @ x), "central", n)
    ratio = 0.5 * (np.linalg.slogdet(P2)[1] - np.linalg.slogdet(P)[1])
    assert alpha == pytest.approx(0.0, abs=1e-12)
    assert ratio == pytest.approx(E.volume_log_ratio_central(n), abs=1e-9) and 0.5 * dlog == pytest.approx(ratio, abs=1e-9)
    assert ratio <= -1.0 / (2 * (n + 1)) and np.allclose(P2, P2.T) and np.linalg.eigvalsh(P2)[0] > 0


def test_closed_form_matches_the_textbook_constant_for_n_two():
    assert math.exp(E.volume_log_ratio_central(2)) == pytest.approx(math.sqrt((2 / 3) ** 3 * 2), rel=1e-12)
    assert math.exp(E.volume_log_ratio_central(2)) == pytest.approx(0.7698, abs=1e-4)


@pytest.mark.parametrize("n", [2, 3, 5, 9])
def test_deep_cut_contains_the_cut_ellipsoid_and_is_smaller_than_the_central_one(n):
    rng = np.random.default_rng(100 + n)
    M = rng.normal(size=(n, n))
    P = M @ M.T + n * np.eye(n)
    x = rng.normal(size=n)
    a = rng.normal(size=n)
    s = math.sqrt(float(a @ P @ a))
    for alpha in (0.1, 0.4, 0.8):
        beta = float(a @ x) - alpha * s
        x2, P2, al, dlog = E._cut_update(x, P, a, beta, "deep", n)
        _xc, _Pc, _al, dlog_c = E._cut_update(x, P, a, float(a @ x), "central", n)
        assert al == pytest.approx(alpha) and dlog < dlog_c
        assert np.linalg.eigvalsh(P2)[0] > 0 and 0.5 * dlog == pytest.approx(0.5 * (np.linalg.slogdet(P2)[1] - np.linalg.slogdet(P)[1]), abs=1e-9)
        # Punkte von E mit a z <= beta liegen in E+ (Stichprobe auf der Oberfläche und im Inneren)
        L = np.linalg.cholesky(P)
        inside = 0
        for _ in range(4000):
            u = rng.normal(size=n)
            u *= rng.random() ** (1.0 / n) / np.linalg.norm(u)
            z = x + L @ u
            if float(a @ z) <= beta:
                inside += 1
                assert _quad(z, x2, P2) <= 1.0 + 1e-9
        assert inside > 0


def test_deep_cut_boundary_point_lies_on_the_new_ellipsoid():
    n = 3
    P = np.diag([4.0, 1.0, 0.25])
    x = np.array([1.0, -1.0, 0.5])
    a = np.array([1.0, 2.0, -1.0])
    s = math.sqrt(float(a @ P @ a))
    alpha = 0.5
    beta = float(a @ x) - alpha * s
    x2, P2, _al, _d = E._cut_update(x, P, a, beta, "deep", n)
    # Punkt der Ellipse in Richtung -a, der auf der Schnittebene liegt: z = x - alpha * Pa/s
    z = x - alpha * (P @ a) / s
    assert float(a @ z) == pytest.approx(beta, abs=1e-12) and _quad(z, x, P) == pytest.approx(alpha * alpha, abs=1e-12)
    # Das Schnittrand-Ellipsoid der Schnittfläche liegt auf dem Rand von E+: Randpunkt der Schnittfläche ist z + (Vektor orthogonal in der P-Metrik)
    w = np.array([2.0, -1.0, 0.0])
    w = w - (float(a @ w) / float(a @ P @ a)) * (P @ a)
    w *= math.sqrt(1 - alpha * alpha) / math.sqrt(float(w @ np.linalg.solve(P, w)))
    zb = z + w
    assert float(a @ zb) == pytest.approx(beta, abs=1e-9) and _quad(zb, x, P) == pytest.approx(1.0, abs=1e-9)
    assert _quad(zb, x2, P2) == pytest.approx(1.0, abs=1e-9)


def test_cut_covering_the_whole_ellipsoid_is_reported():
    P = np.eye(2)
    x2, P2, alpha, _d = E._cut_update(np.zeros(2), P, np.array([1.0, 0.0]), -1.5, "deep", 2)
    assert x2 is None and P2 is None and alpha == pytest.approx(1.5)


def test_logdet_is_tracked_exactly_over_a_run():
    inst = S.centre_instance()
    res = E.ellipsoid(inst, eps=1e-4, cut="deep", keep=True, keep_eig=True)
    P0 = res.mats[0]
    for k in (0, 5, len(res.mats) // 2, len(res.mats) - 1):
        want = 0.5 * (np.linalg.slogdet(res.mats[k])[1] - np.linalg.slogdet(P0)[1])
        assert res.log_volume[k] == pytest.approx(want, abs=1e-6)
    assert all(e > 0 for e in res.min_eig) and len(res.min_eig) == res.iterations


# --- 2. Invariante: das Optimum liegt in jeder Ellipse -----------------------------------------------------------------------------------------------

def test_the_optimum_lies_in_every_ellipsoid_for_both_cut_types():
    checked = 0
    worst = 0.0
    for inst in _instances(3):
        x_star, _opt = _opt_point(inst)
        for cut in E.CUTS:
            for select in E.SELECTS:
                res = E.ellipsoid(inst, eps=1e-5, cut=cut, select=select, keep=True)
                assert res.status == "optimal"
                for x, P in zip(res.centres, res.mats):
                    q = _quad(x_star, x, P)
                    worst = max(worst, q)
                    assert q <= 1.0 + 1e-6, (inst.kind, cut, select, q)
                    checked += 1
    assert checked > 1000 and worst > 0.05


def test_start_ball_contains_the_box_and_the_optimum():
    for inst in _instances(3):
        centre, radius, U, free = E.start_ball(inst)
        x_star, _ = _opt_point(inst)
        assert np.all(x_star <= U + 1e-9) and np.all(x_star >= -1e-12) and not free
        for corner in ([0.0] * inst.n, list(U)):
            assert np.linalg.norm(np.array(corner) - centre) <= radius * (1 + 1e-12)
        bigger = E.start_ball(inst, 4.0)[1]
        assert bigger == pytest.approx(4.0 * radius)


# --- 3. Optimum gegen HiGHS und Simplex ------------------------------------------------------------------------------------------------------------

def test_optimum_equals_highs_and_the_simplex_over_hundreds_of_instances():
    count = 0
    for inst in _instances(60):
        opt = _opt_point(inst)[1]
        simplex = A.solve(inst)
        assert simplex.obj == pytest.approx(opt, rel=1e-7, abs=1e-6)
        for cut, select in (("central", "first"), ("deep", "most")):
            res = E.ellipsoid(inst, eps=1e-7, cut=cut, select=select)
            assert res.status == "optimal", (inst.kind, cut, select, res.status, res.note)
            assert res.obj <= opt + 1e-6 * (1 + abs(opt)) and res.obj >= opt - 1e-6 * (1 + abs(opt)) - 1e-7, (inst.kind, res.obj, opt)
            assert res.upper >= opt - 1e-9 and res.gap <= 1e-7 * (1 + abs(res.obj)) + 1e-12
            A_, b_, _c = inst.arrays()
            assert np.all(A_ @ np.array(res.x) <= b_ + 1e-8) and np.all(np.array(res.x) >= -1e-9)
            count += 1
    assert count >= 300


def test_mixed_instances_with_equalities_reach_the_optimum_within_the_relaxation():
    rows = 0
    for seed in range(12):
        inst = S.generate("mixed", 5, 5, 0.5, seed)
        if reference_status(inst) != "optimal":
            continue
        opt = _highs(inst)
        res = E.ellipsoid(inst, eps=1e-6, eq_tol=1e-6, cut="deep", select="most")
        assert res.status == "optimal", (seed, res.status, res.note)
        assert res.obj == pytest.approx(-opt.fun, rel=1e-3, abs=1e-3)
        rows += 1
    assert rows >= 6


def test_simplex_copy_reproduces_the_centre():
    assert A.solve(S.centre_instance()).obj == pytest.approx(720.0)


# --- 4. Unzulässig / Unbeschränkt ---------------------------------------------------------------------------------------------------------------------

def test_infeasible_instances_are_recognised_and_never_reported_as_optimal():
    for cut in E.CUTS:
        res = E.ellipsoid(S.infeasible_instance(), eps=1e-6, cut=cut, select="most")
        assert res.status == "infeasible" and not res.x and math.isnan(res.obj) and res.iterations == 1 and "alpha" in res.note
    found = 0
    for seed in range(60):
        rng = random.Random(seed)
        n = rng.randint(2, 4)
        rows = [[rng.uniform(0.5, 3) for _ in range(n)] for _ in range(3)]
        b = [rng.uniform(5, 20) for _ in range(3)]
        rows.append(list(rows[0]))                                                                # zweite Nebenbedingung: gleiche Zeile als >=, weit über dem Bestand
        b.append(b[0] + rng.uniform(1.0, 5.0))
        inst = _custom(rows, b, [1.0] * n, [S.LE] * 3 + [S.GE])
        assert reference_status(inst) == "infeasible"
        for cut in E.CUTS:
            r = E.ellipsoid(inst, eps=1e-6, cut=cut, select="most", max_iter=20000)
            assert r.status == "infeasible" and not r.x, (seed, cut, r.status, r.note)
        found += 1
    assert found == 60


def test_unbounded_instance_ends_at_the_ball_boundary():
    res = E.ellipsoid(S.unbounded_instance(), eps=1e-6, cut="deep", select="most", max_iter=5000)
    assert res.status == "ball" and "Startkugel" in res.note and res.obj > 100.0
    bigger = E.ellipsoid(S.unbounded_instance(), eps=1e-6, cut="deep", select="most", radius_factor=4.0, max_iter=5000)
    assert bigger.status in ("ball", "numerical") and bigger.obj > res.obj                       # bei 4-facher Kugel wird P schlecht konditioniert (gemessen: Bruch nach 34 Iterationen)


# --- 5. Theorie-Schranke ------------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("n", range(2, 7))
def test_feasibility_search_is_within_the_theory_bound_when_a_ball_fits(n):
    r = 0.05
    inst = _custom([[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)] + [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)], [1.0] * n + [0.9] * n, [0.0] * n, [S.LE] * n + [S.GE] * n)
    for cut in E.CUTS:
        res = E.ellipsoid(inst, cut=cut, select="first")
        assert res.status == "optimal" and res.first_feasible >= 1
        assert res.first_feasible <= E.theory_bound(n, res.r0, r) + 1
        x = np.array(res.x)
        assert np.all(x >= 0.9 - 1e-12) and np.all(x <= 1.0 + 1e-12)


def test_iterations_do_not_fall_with_a_tighter_accuracy():
    inst = S.centre_instance()
    its = [E.ellipsoid(inst, eps=10.0 ** -k).iterations for k in (2, 4, 6, 8)]
    assert its == sorted(its) and its[-1] > its[0] + 50


def test_first_feasible_iteration_grows_with_ln_of_the_equality_relaxation():
    """Die Gesamtiterationen sind nicht monoton (eine lockerere Aufweitung ändert das Optimum), aber die Suche nach dem ersten zulässigen Punkt wächst mit ln(1/eq_tol)."""
    for seed in (3, 4):
        inst = S.generate("mixed", 5, 5, 0.5, seed)
        first = [E.ellipsoid(inst, eps=1e-4, eq_tol=t, cut="deep", select="most", max_iter=200_000).first_feasible for t in (1e-2, 1e-4, 1e-6, 1e-8)]
        assert first == sorted(first) and first[-1] > first[0] + 15 and len(set(first)) == 4


# --- 6. Sonderfälle, Buchführung, Zweige ---------------------------------------------------------------------------------------------------------------

def test_one_dimensional_instance_is_bisection():
    inst = _custom([[2.0]], [7.0], [3.0], [S.LE])
    for cut in E.CUTS:
        res = E.ellipsoid(inst, eps=1e-9, cut=cut)
        assert res.status == "optimal" and res.x[0] == pytest.approx(3.5, abs=1e-6) and res.obj == pytest.approx(10.5, abs=1e-6)


def test_single_row_and_origin_feasible_cases():
    inst = _custom([[1.0, 2.0]], [10.0], [1.0, 1.0], [S.LE])
    res = E.ellipsoid(inst, eps=1e-7, cut="deep")
    assert res.status == "optimal" and res.obj == pytest.approx(10.0, abs=1e-5)
    zero = _custom([[1.0, 1.0]], [4.0], [0.0, 0.0], [S.LE])
    r0 = E.ellipsoid(zero)
    assert r0.status == "optimal" and "Machbarkeitsproblem" in r0.note and r0.iterations == 1


def test_interior_optimum_at_the_ball_centre_is_found():
    inst = _custom([[1.0, 0.0], [0.0, 1.0]], [10.0, 10.0], [1e-12, 1e-12], [S.LE, S.LE])
    res = E.ellipsoid(inst, eps=1e-3)
    assert res.status == "optimal"


def test_determinism_and_bookkeeping():
    inst = S.centre_instance()
    a = E.ellipsoid(inst, eps=1e-6, keep=True)
    b = E.ellipsoid(inst, eps=1e-6, keep=True)
    assert a.x == b.x and a.iterations == b.iterations and len(a.best_path) == len(a.upper_path) == a.iterations == len(a.centres) == len(a.mats) == len(a.cuts)
    assert a.feasibility_cuts + a.objective_cuts == a.iterations
    m_rows = len(E.rows_of(inst)[1])
    assert a.flops == a.iterations * E.flops_per_iteration(m_rows, inst.n) and E.flops_per_iteration(m_rows, inst.n) == 2 * m_rows * inst.n + 6 * inst.n ** 2 + 4 * inst.n
    assert a.first_feasible >= 1 and math.isnan(a.best_path[0]) is (a.first_feasible > 1)
    assert all(a.upper_path[k] >= a.best_path[k] - 1e-9 for k in range(a.first_feasible - 1, a.iterations))
    bests = [v for v in a.best_path if not math.isnan(v)]
    assert bests == sorted(bests)
    with pytest.raises(ValueError):
        E.ellipsoid(inst, cut="shallow")
    with pytest.raises(ValueError):
        E.ellipsoid(inst, select="random")


def test_every_branch_is_executed():
    kinds = set()
    for inst in _instances(2):
        for cut in E.CUTS:
            for select in E.SELECTS:
                res = E.ellipsoid(inst, eps=1e-5, cut=cut, select=select, keep=True)
                kinds |= {k for k, _r, _a in res.cuts}
    assert kinds == {"feasibility", "objective"}
    statuses = {E.ellipsoid(S.infeasible_instance(), cut="deep", select="most").status, E.ellipsoid(S.unbounded_instance(), max_iter=3000).status, E.ellipsoid(S.textbook_instance()).status,
                E.ellipsoid(S.centre_instance(), max_iter=20).status}
    assert statuses == {"infeasible", "ball", "optimal", "limit"}


def test_numerical_breakdown_is_reported_not_hidden():
    inst = S.generate("random", 10, 12, 0.5, 7)
    res = E.ellipsoid(inst, eps=1e-15, cut="central", max_iter=400_000)
    assert res.status in ("numerical", "limit", "optimal")
    if res.status == "numerical":
        assert res.note
