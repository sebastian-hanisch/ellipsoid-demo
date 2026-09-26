"""Presets: gültige Werte und jede Zahl der Hilfetexte gegen die echten Auswertungsfunktionen (Iterationszahlen mit kleinem Band: Gleitkomma-Rundung kann sie plattformabhängig um wenige verschieben)."""

import math

import pytest

import ell_constants as C
import ell_evaluation as ev
from ell_evaluation import Settings
from ell_presets import PRESET_KEYS, SETTING_SPECS


def near(x, want, rel=0.02, tol=2):
    return abs(x - want) <= max(rel * abs(want), tol)


def _settings(name, **over):
    p = {**C.PRESETS[name], **over}
    return Settings(p["kind"], p["m"], p["n"], p["seed"], p["cut"], p["sel"], p["eps"], p["rad"], p["scale"])


def _has(name, *values):
    for v in values:
        assert v in C.PRESET_HELP[name], (name, v)


def test_every_preset_has_valid_values_and_a_help_text():
    assert list(C.PRESETS) == list(C.PRESET_HELP) and len(C.PRESETS) == 12
    for name, p in C.PRESETS.items():
        assert set(p) <= set(PRESET_KEYS) and {"kind", "step", "cut", "sel", "eps", "rad", "scale"} <= set(p), name
        for key, state_key in PRESET_KEYS.items():
            if key in p and state_key in SETTING_SPECS:
                spec = SETTING_SPECS[state_key]
                assert spec.caster(p[key]) == p[key], (name, key)
                if spec.lo is not None:
                    assert spec.lo <= p[key] <= spec.hi, (name, key)
        assert C.PRESET_HELP[name].strip()
        if "iter_k" in p:
            assert p["step"] == 1 and p["iter_k"] > 1


def test_help_lehrbuch_and_zentrum():
    a = ev.analyse(_settings("Lehrbuch: die ersten Ellipsen"))
    deep = ev.analyse(_settings("Lehrbuch: die ersten Ellipsen", cut="deep"))
    assert near(a.res.iterations, 100) and near(deep.res.iterations, 78) and a.simplex.pivots == 2 and a.res.obj == pytest.approx(36.0, abs=1e-4)
    assert a.res.cuts[0][0] == "objective" and a.res.cuts[2][0] == "feasibility" and a.inst.row_names[a.res.cuts[2][1]] == "Lagerfläche" and a.res.best_path[0] == pytest.approx(21.0) and a.res.centres[0].tolist() == [2.0, 3.0]
    _has("Lehrbuch: die ersten Ellipsen", "[0, 4] × [0, 6]", "(2, 3)", "Zielwert 21", "Iteration 3", "Lagerfläche", "100 Iterationen", "36", "78", "2 Pivots")
    z = ev.analyse(_settings("Zentrum: Ellipsoid gegen Simplex"))
    assert near(z.res.iterations, 594) and near(z.res.feasibility_cuts, 484) and near(z.res.objective_cuts, 110) and z.simplex.pivots == 4 and z.flops_simplex == 400 and near(z.flops_ell / z.flops_simplex, 386, rel=0.03)
    _has("Zentrum: Ellipsoid gegen Simplex", "594 Iterationen", "484", "110", "720", "4 Pivots", "154.440 gegen 400", "386-Fache")


def test_help_size_and_accuracy():
    a = ev.analyse(_settings("Iterationen wachsen mit n²"))
    assert near(a.res.iterations, 1625)
    rows = {r["n"]: r for r in ev.size_sweep(_settings("Iterationen wachsen mit n²"))}
    assert near(rows[2]["iterations"], 102) and near(rows[6]["iterations"], 916) and near(rows[10]["iterations"], 2380) and near(rows[20]["iterations"], 9748)
    per = [r["per_n2"] for r in rows.values()]
    assert 23.5 <= min(per) and max(per) <= 25.9
    _has("Iterationen wachsen mit n²", "1625", "24 bis 25.5", "102 bei n = 2", "916 bei n = 6", "2380 bei n = 10", "9748 bei n = 20")
    sw = ev.eps_sweep(_settings("Genauigkeit: Iterationen je Zehnerpotenz"))
    its = {r["k"]: r["iterations"] for r in sw["rows"]}
    assert near(its[2], 448) and near(its[10], 2777) and near(sw["slope"], 291, rel=0.02) and round(sw["theory_slope"]) == 332 and round(100 * sw["share"]) == 88
    _has("Genauigkeit: Iterationen je Zehnerpotenz", "291", "448 bei 10^-2", "2777 bei 10^-10", "332", "88 %")


def test_help_deep_cuts_and_theory_bound():
    s = _settings("Tiefe Schnitte")
    deep, central = ev.analyse(s), ev.analyse(_settings("Tiefe Schnitte", cut="central"))
    assert near(deep.res.iterations, 1405) and near(central.res.iterations, 1625) and round(100 * (1 - deep.res.iterations / central.res.iterations)) == 14
    rows = ev.cut_compare(s)
    assert near(rows[2]["iterations"], 1400) and near(rows[0]["iterations"], 1611) and near(rows[1]["iterations"], 1627) and round(100 * rows[2]["saving"]) == 13
    _has("Tiefe Schnitte", "1405", "1625", "−14 %", "1400 gegen 1611", "−13 %", "1627 gegen 1611")
    b = {r["n"]: r for r in ev.feasibility_bound(_settings("Die Theorie-Schranke ist weit weg"))}
    assert near(b[8]["first_feasible"], 10) and round(b[8]["bound"]) == 360 and round(100 * b[8]["first_feasible"] / b[8]["bound"], 1) == pytest.approx(2.8, abs=0.5)
    assert near(b[10]["first_feasible"], 10) and round(b[10]["bound"]) == 624 and round(100 * b[10]["first_feasible"] / b[10]["bound"], 1) == pytest.approx(1.6, abs=0.5)
    _has("Die Theorie-Schranke ist weit weg", "n = 8 im Median 10 Iterationen", "360 (2.8 %)", "624 (1.6 %)")


def test_help_scaling_equalities_infeasible_unbounded():
    s = _settings("Schlechte Skalierung")
    a = ev.analyse(s)
    assert a.res.status == "numerical" and near(a.res.iterations, 1978, rel=0.1) and a.res.obj == pytest.approx(249.94, abs=0.3) and a.ref_obj == pytest.approx(246.79, abs=0.01) and round(100 * (a.res.obj - a.ref_obj) / a.ref_obj, 1) == pytest.approx(1.3, abs=0.2)
    rows = {r["k"]: r for r in ev.scale_sweep(s)}
    assert rows[8]["ell_broke"] == 1 and rows[10]["ell_broke"] >= 3 and rows[12]["ell_broke"] == 5 and rows[10]["sim_wrong"] == 0 and rows[12]["sim_wrong"] == 3 and rows[12]["sim_max"] > 0.5
    _has("Schlechte Skalierung", "10^12", "249.94", "1.3 %", "246.79", "bei 10^8 in 1", "bei 10^10 in 4", "bei 10^12 in 5 von 5", "bis 10^10 exakt", "3 von 5")
    sw = ev.equality_sweep(_settings("Gleichungen: dünnes Inneres"))
    ff = {r["k"]: r["first_feasible"] for r in sw["rows"]}
    assert sw["eq_rows"] == 1 and near(ff[2], 28, tol=3) and near(ff[4], 46, tol=3) and near(ff[6], 64, tol=3) and near(ff[8], 83, tol=3)
    last = sw["rows"][-1]
    assert last["k"] == 10 and (last["status"] == "numerical" and near(last["iterations"], 84, rel=0.1) or last["first_feasible"] > ff[8])
    _has("Gleichungen: dünnes Inneres", "Iteration 28", "in 46", "in 64", "in 83", "bei 10^-10 scheitert die Numerik nach 84")
    inf = ev.analyse(_settings("Unzulässig: Ellipse ganz abgeschnitten"))
    assert inf.res.status == "infeasible" and inf.res.iterations == 1 and inf.res.cuts[0][1] == 2 and inf.res.cuts[0][2] == pytest.approx(1.11, abs=0.01) and inf.inst.senses[2] == ">="
    _has("Unzulässig: Ellipse ganz abgeschnitten", "x1 ≥ 6", "Zeile 2", "1.11", "höchstens 4")
    unb = ev.analyse(_settings("Unbeschränkt: Kugelrand"))
    assert unb.res.status == "ball" and near(unb.res.iterations, 34, tol=4) and unb.res.obj == pytest.approx(1999.998, abs=0.01) and unb.simplex.status == "unbounded" and not math.isnan(unb.res.obj)
    _has("Unbeschränkt: Kugelrand", "34 Iterationen", "1999.998", "1000", "unbeschränkt")


def test_help_klee_minty_cube():
    name = "Klee-Minty-Würfel: hier gewinnt das Ellipsoid"
    s = _settings(name)
    a = ev.analyse(s)
    assert s.kind == "klee_minty" and a.inst.n == 14 and a.simplex.pivots == 2 ** 14 - 1 and near(a.res.iterations, 4052, rel=0.03)
    assert round(a.flops_ell / 1e6, 1) == pytest.approx(8.2, abs=0.2) and round(a.flops_simplex / 1e6, 1) == pytest.approx(14.3, abs=0.05) and round(a.flops_ell / a.flops_simplex, 2) == pytest.approx(0.57, abs=0.03)
    sw = ev.cube_sweep(s)
    central = ev.cube_sweep(_settings(name, cut="central", sel="first"))
    assert sw["crossover"] == 13 and central["crossover"] == 14 and central["rows"][-1]["ratio"] == pytest.approx(0.97, abs=0.06)
    _has(name, "n = 14", "2^14", "16383", "4052", "8.2 Mio. gegen 14.3 Mio.", "0.57", "n = 13 (tief)", "n = 14 (zentral", "0.97", "Steepest Edge")


def test_help_accuracy_limit():
    name = "Genauigkeitsgrenze: 10^-14 geht nicht"
    a = ev.analyse(_settings(name))
    assert _settings(name).eps == 1e-14 and a.res.status == "numerical" and near(a.res.iterations, 3491, rel=0.05)
    assert 1e-10 <= abs(a.res.obj - a.ref_obj) <= 5e-10
    _has(name, "ε = 10^-14", "3491 Iterationen", "2 · 10^-10", "je fünf Zufallsinstanzen", "n = 4 bis 40", "10^-12", "ab 10^-13")
