"""Auswertung: Einzellauf gegen den Simplex, Größen-/Genauigkeits-/Radius-Kurven, Schnittarten, Theorie-Schranke, Skalierung, Gleichungen."""

import math

import pytest

import ell_constants as C
import ell_ellipsoid as E
import ell_evaluation as ev
import ell_scenario as S
from ell_evaluation import Settings


def test_settings_clamp_the_index_controls():
    s = Settings(eps_i=99, radius_i=-3, scale_i=99)
    assert s.eps == 1e-14 and s.radius_factor == 1 and s.scale_exp == 12
    assert Settings().eps == 1e-6 and Settings(eps_i=0).eps == 1e-2 and Settings(radius_i=4).radius_factor == 64


def test_analyse_compares_with_the_simplex_and_is_cached():
    a = ev.analyse(Settings())
    assert a is ev.analyse(Settings())
    assert a.res.status == "optimal" and a.simplex.status == "optimal" and a.ref_obj == pytest.approx(720.0) and abs(a.obj_error) < 1e-3
    assert a.flops_simplex == 400 and a.flops_ell == a.res.iterations * E.flops_per_iteration(len(E.rows_of(a.inst)[1]), a.inst.n) and a.cut == "central"
    assert len(a.res.centres) == a.res.iterations and a.res.mats == []                                           # Matrizen nur bei zwei Variablen
    two = ev.analyse(Settings("textbook"))
    assert len(two.res.mats) == two.res.iterations and len(two.res.min_eig) == two.res.iterations
    inf, unb = ev.analyse(Settings("infeasible")), ev.analyse(Settings("unbounded", cut="deep", select="most"))
    assert inf.res.status == "infeasible" and inf.ref_status == "infeasible" and math.isnan(inf.obj_error)
    assert unb.res.status == "ball" and unb.simplex.status == "unbounded" and unb.flops_simplex == 0


def test_scaled_instance_keeps_the_true_reference_value():
    a = ev.analyse(Settings("random", 6, 8, 35, scale_i=3))
    assert a.inst != a.base and a.ref_obj == pytest.approx(ev.analyse(Settings("random", 6, 8, 35)).ref_obj)
    assert S.column_scaled(a.base, 0) is a.base
    assert a.inst.c[-1] == pytest.approx(a.base.c[-1] * 10.0 ** 6) and a.inst.c[0] == pytest.approx(a.base.c[0])


def test_size_sweep_iterations_grow_like_n_squared():
    rows = ev.size_sweep(Settings("random"))
    assert [r["n"] for r in rows] == list(C.SWEEP_SIZES) and all(r["optimal"] == 5 for r in rows)
    assert all(22.0 <= r["per_n2"] <= 27.0 for r in rows), [r["per_n2"] for r in rows]
    its = [r["iterations"] for r in rows]
    assert its == sorted(its) and all(r["min"] <= r["iterations"] <= r["max"] for r in rows)
    assert all(r["ratio"] > 50 for r in rows) and rows[-1]["ratio"] > rows[0]["ratio"] and all(r["flops_ell"] > 20 * r["flops_simplex"] for r in rows)
    assert ev.size_sweep(Settings("random")) is rows


def test_eps_sweep_is_linear_in_the_number_of_digits_and_below_the_theory_slope():
    sw = ev.eps_sweep(Settings("random", 8, 8, 35))
    its = [r["iterations"] for r in sw["rows"]]
    assert [r["k"] for r in sw["rows"]] == list(range(2, 11)) and its == sorted(its) and all(r["optimal"] == 5 for r in sw["rows"])
    diffs = [b - a for a, b in zip(its, its[1:])]
    assert max(diffs) < 1.3 * min(diffs)                                                                       # gleichmäßig: fast dieselbe Zahl je Stelle
    assert sw["theory_slope"] == pytest.approx(2 * 8 * 9 * math.log(10)) and 0.7 < sw["share"] < 1.0


def test_radius_sweep_grows_only_logarithmically():
    rows = ev.radius_sweep(Settings("random", 8, 8, 35))
    its = [r["iterations"] for r in rows]
    assert [r["factor"] for r in rows] == list(C.RADIUS_FACTORS) and its == sorted(its) and its[-1] < 1.6 * its[0]
    first = [r["first_feasible"] for r in rows]
    assert first == sorted(first) and first[-1] > 5 * first[0]


def test_cut_compare_deep_cuts_save_iterations():
    rows = ev.cut_compare(Settings("random", 8, 8, 35))
    assert [(r["cut"], r["select"]) for r in rows] == [("central", "first"), ("central", "most"), ("deep", "first"), ("deep", "most")]
    assert rows[0]["saving"] == 0.0 and rows[2]["saving"] > 0.08 and rows[3]["saving"] > 0.08 and abs(rows[1]["saving"]) < 0.05


def test_inscribed_radius_of_a_box_and_of_the_textbook():
    box = S.Instance(((1.0, 0.0), (0.0, 1.0)), (2.0, 4.0), (1.0, 1.0), (S.LE, S.LE), ("a", "b"), ("r1", "r2"), "custom")
    assert ev.inscribed_radius(box) == pytest.approx(1.0)
    assert ev.inscribed_radius(S.textbook_instance()) == pytest.approx(2.0)


def test_feasibility_bound_is_never_exceeded_and_far_from_tight():
    rows = ev.feasibility_bound(Settings("random"))
    assert [r["n"] for r in rows] == list(C.BOUND_SIZES)
    assert all(r["max_share"] < 1.0 for r in rows) and all(r["share"] < 0.2 for r in rows) and rows[-1]["bound"] > 20 * rows[-1]["first_feasible"]


def test_scale_sweep_iterations_grow_and_precision_fails_only_at_large_scales():
    rows = ev.scale_sweep(Settings("random", 6, 8, 35))
    assert [r["k"] for r in rows] == list(C.SCALE_EXPS)
    its = [r["iterations"] for r in rows[:6]]
    assert its == sorted(its) and its[-1] > 1.5 * its[0]
    assert all(r["ell_wrong"] == 0 and r["sim_wrong"] == 0 and r["ell_broke"] == 0 for r in rows[:4])
    assert rows[-1]["ell_broke"] >= 1 and rows[-1]["ell_wrong"] >= 1 and rows[-1]["sim_wrong"] >= 1 and rows[-1]["ell_max"] > 1e-3


def test_equality_sweep_first_feasible_grows_and_breaks_at_the_thinnest():
    sw = ev.equality_sweep(Settings("mixed", 8, 8, 35))
    assert sw["eq_rows"] >= 1 and (sw["m"], sw["n"], sw["seed"]) == (8, 8, 35)
    ok = [r for r in sw["rows"] if r["first_feasible"] > 0]
    ff = [r["first_feasible"] for r in ok]
    assert len(ok) >= 4 and ff == sorted(ff) and ff[-1] > ff[0] + 30
    assert sw["rows"][-1]["status"] in ("numerical", "limit", "optimal") and (sw["rows"][-1]["first_feasible"] < 0 or sw["rows"][-1]["first_feasible"] > ff[-1])
    other = ev.equality_sweep(Settings("random"))
    assert (other["m"], other["n"]) == (8, 8) and other["seed"] == C.DEFAULT_SEED


def test_cube_sweep_the_simplex_needs_two_to_the_n_minus_one_pivots_and_the_ellipsoid_wins_late():
    sw = ev.cube_sweep(Settings(cut="deep", select="most"))
    rows = sw["rows"]
    assert [r["n"] for r in rows] == list(C.CUBE_SIZES) and all(r["pivots"] == 2 ** r["n"] - 1 and r["status"] == "optimal" for r in rows)
    assert all(r["ratio"] > 20 for r in rows[:5]) and rows[-1]["ratio"] < 0.8 and sw["crossover"] in (12, 13, 14) and set(sw["wins"]) >= {14}
    ratios = [r["ratio"] for r in rows]
    assert ratios[3:] == sorted(ratios[3:], reverse=True)
    central = ev.cube_sweep(Settings(cut="central", select="first"))
    assert central["crossover"] in (13, 14, None) and central["rows"][-1]["ratio"] < 1.15 and central["rows"][-3]["ratio"] > 1.4
