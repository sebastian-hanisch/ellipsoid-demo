"""Jede Zahl aus README und App über die echten Auswertungsfunktionen (Iterationen mit Band, Anteile als gerundete Prozent; Grenzen mit Sicherheitsabstand: Rundung kann plattformabhängig kleine Unterschiede machen)."""

import math

import pytest

import ell_constants as C
import ell_ellipsoid as E
import ell_evaluation as ev
import ell_scenario as S
from ell_evaluation import Settings


def near(x, want, rel=0.02, tol=2):
    return abs(x - want) <= max(rel * abs(want), tol)


def test_readme_iterations_are_about_25_n_squared_for_every_size():
    rows = {r["n"]: r for r in ev.size_sweep(Settings("random"))}
    want = {2: 102, 4: 389, 6: 916, 8: 1611, 10: 2380, 12: 3677, 16: 6135, 20: 9748}
    assert all(near(rows[n]["iterations"], w) for n, w in want.items())
    assert all(23.5 <= r["per_n2"] <= 25.9 for r in rows.values()) and all(r["optimal"] == 5 for r in rows.values())


def test_readme_deep_cuts_and_row_choice():
    central, deep, most = ev.size_sweep(Settings("random")), ev.size_sweep(Settings("random", cut="deep")), ev.size_sweep(Settings("random", select="most"))
    saving = {a["n"]: round(100 * (1 - b["iterations"] / a["iterations"])) for a, b in zip(central, deep)}
    assert all(abs(saving[n] - w) <= 1 for n, w in {2: 22, 4: 19, 6: 17, 8: 13, 10: 13, 12: 12, 16: 15, 20: 13}.items())
    assert all(abs(b["iterations"] / a["iterations"] - 1) < 0.025 for a, b in zip(central, most))


def test_readme_accuracy_slope_over_n():
    shares = {n: ev.eps_sweep(Settings("random", 6, n, 35)) for n in (4, 8, 12)}
    assert [round(shares[n]["slope"]) for n in (4, 8, 12)] == pytest.approx([73, 291, 658], abs=4)
    assert [round(100 * shares[n]["share"]) for n in (4, 8, 12)] == pytest.approx([79, 88, 92], abs=2) and [round(shares[n]["theory_slope"]) for n in (4, 8, 12)] == [92, 332, 718]


def test_readme_start_ball():
    rows = {r["factor"]: r for r in ev.radius_sweep(Settings("random", 8, 8, 35))}
    assert near(rows[1]["iterations"], 1611) and near(rows[64]["iterations"], 2141) and round(100 * (rows[64]["iterations"] / rows[1]["iterations"] - 1)) == pytest.approx(33, abs=2)
    assert near(rows[1]["first_feasible"], 10) and near(rows[64]["first_feasible"], 407, rel=0.05, tol=10)


def test_readme_operations_against_the_simplex():
    rows = {r["n"]: r for r in ev.size_sweep(Settings("random"))}
    assert rows[2]["pivots"] == 1 and rows[20]["pivots"] == 14 and round(rows[2]["ratio"]) == pytest.approx(163, abs=8) and round(rows[20]["ratio"]) == pytest.approx(1650, rel=0.03)
    assert min(r["ratio"] for r in rows.values()) > 150 and rows[12]["ratio"] > 1500 and rows[8]["ratio"] > 1000


def test_readme_theory_bound():
    rows = {r["n"]: r for r in ev.feasibility_bound(Settings("random"))}
    assert [round(100 * rows[n]["share"], 1) for n in (2, 3, 4, 6, 8, 10)] == pytest.approx([11.7, 4.7, 4.4, 2.6, 2.8, 1.6], abs=1.0)
    assert all(rows[n]["max_share"] < 0.13 for n in rows) and round(rows[10]["bound"]) == 624


def test_readme_klee_minty_cube():
    deep, central = ev.cube_sweep(Settings(cut="deep", select="most")), ev.cube_sweep(Settings())
    d = {r["n"]: r for r in deep["rows"]}
    c = {r["n"]: r for r in central["rows"]}
    assert d[14]["pivots"] == 16383 and near(d[14]["iterations"], 4052, rel=0.03) and near(c[14]["iterations"], 6856, rel=0.03) and near(d[10]["iterations"], 2185, rel=0.03)
    assert [round(d[n]["ratio"], 1) for n in (2, 6, 10)] == pytest.approx([43.2, 27.1, 4.8], rel=0.06) and round(d[14]["ratio"], 2) == pytest.approx(0.57, abs=0.03) and round(c[14]["ratio"], 2) == pytest.approx(0.97, abs=0.06)
    assert deep["crossover"] == 13 and central["crossover"] == 14 and round(d[12]["ratio"], 1) == pytest.approx(1.7, abs=0.1) and round(c[12]["ratio"], 1) == pytest.approx(2.7, abs=0.2)
    assert c[12]["ratio"] > 2.0 and d[13]["ratio"] == pytest.approx(1.0, abs=0.05) and all(r["status"] == "optimal" for r in deep["rows"] + central["rows"])
    cross = {C.EPS_EXPS[i]: ev.cube_sweep(Settings(cut="deep", select="most", eps_i=i))["crossover"] for i in (0, 1, 3, 4, 5)}
    assert cross[2] in (10, 11, 12) and cross[3] in (11, 12, 13) and cross[6] == 13 and cross[8] in (13, 14, None) and cross[10] is None


def test_readme_accuracy_floor_is_the_same_for_every_size():
    for n in (4, 14, 40):
        for k, want in ((10, "optimal"), (12, "optimal"), (13, "numerical"), (14, "numerical")):
            for sd in C.SWEEP_SEEDS[:2]:
                r = E.ellipsoid(S.generate("random", n, n, C.DENSITY, sd), eps=10.0 ** -k, cut="central", max_iter=800_000)
                assert r.status == want, (n, k, sd, r.status)
                if want == "numerical":
                    assert r.x and "Rundung" in r.note


def test_readme_scaling_and_equalities():
    rows = {r["k"]: r for r in ev.scale_sweep(Settings("random", 6, 8, 35))}
    assert [rows[k]["ell_broke"] for k in (0, 6, 8, 10, 12)] == [0, 0, 1, 4, 5] and [rows[k]["ell_wrong"] for k in (10, 12)] == [0, 3] and [rows[k]["sim_wrong"] for k in (10, 12)] == [0, 3]
    assert near(rows[0]["iterations"], 1615) and rows[12]["sim_max"] > 0.5 and rows[10]["ell_max"] < 1e-3 and rows[10]["sim_max"] < 1e-12
    sw = ev.equality_sweep(Settings("mixed", 8, 8, 35))
    ff = [r["first_feasible"] for r in sw["rows"]]
    assert [near(ff[i], w, tol=3) for i, w in enumerate((28, 46, 64, 83))] == [True] * 4 and sw["rows"][-1]["k"] == 10


def test_readme_infeasible_and_unbounded():
    inf = ev.analyse(Settings("infeasible"))
    assert inf.res.status == "infeasible" and inf.res.iterations == 1
    unb = ev.analyse(Settings("unbounded", cut="deep", select="most"))
    assert unb.res.status == "ball" and unb.res.obj == pytest.approx(2000.0, abs=0.01) and near(unb.res.iterations, 34, tol=4)
    bigger = ev.analyse(Settings("unbounded", cut="deep", select="most", radius_i=2))
    assert bigger.res.status in ("ball", "numerical") and bigger.res.obj > unb.res.obj


def test_named_instances_in_the_readme():
    out = {}
    for kind in ("textbook", "centre", "degenerate"):
        c = ev.analyse(Settings(kind)).res.iterations
        d = ev.analyse(Settings(kind, cut="deep")).res.iterations
        out[kind] = (c, d)
    assert near(out["textbook"][0], 100) and near(out["textbook"][1], 78) and near(out["centre"][0], 594) and near(out["centre"][1], 490) and out["degenerate"] == out["textbook"]
    assert math.isclose(ev.analyse(Settings("centre")).ref_obj, 720.0)
