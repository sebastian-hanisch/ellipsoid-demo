"""AppTest-Rauchtests: Voreinstellung, jedes Preset, jeder Schritt für jede Instanz und Schnittart, Iterations-Regler, Regler-Randwerte, Permalink-Grenzen, bedingte Regler, Berechnungen auf Abruf, Footer."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import ell_constants as C
import ell_ellipsoid as E
import ell_scenario as S

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run(step=1, **state):
    at = AppTest.from_file(APP, default_timeout=300)
    state.setdefault("ell_step", step)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    return at


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]


def _metric(at, label):
    return next(m for m in at.metric if m.label == label)


def _click(at, key):
    next(b for b in at.button if b.key == key).click().run()


def test_default_run_shows_the_centre_case():
    at = _run()
    _ok(at)
    assert {"Iterationen", "Pivots (Simplex)", "Operationen Ell. / Simplex", "Ergebnis"} == {m.label for m in at.metric}
    assert any("Optimum" in s.value and "Iterationen" in s.value for s in at.success)
    assert _metric(at, "Pivots (Simplex)").value == "4" and _metric(at, "Ergebnis").value == "720.00"
    assert at.get("plotly_chart") and any(s.key == "iter_k" for s in at.slider)


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_button_runs(name):
    at = _run()
    _click(at, f"preset_{name}")
    _ok(at)
    p = C.PRESETS[name]
    ss = at.session_state
    assert (ss["kind_select"], ss["cut_select"], ss["select_select"], ss["eps_select"], ss["radius_select"], ss["scale_select"], ss["ell_step"]) == (p["kind"], p["cut"], p["sel"], p["eps"], p["rad"], p["scale"], p["step"])
    if "iter_k" in p:
        assert ss["iter_k"] == p["iter_k"]


@pytest.mark.parametrize("step", [1, 2, 3, 4])
@pytest.mark.parametrize("kind", list(S.KINDS))
@pytest.mark.parametrize("cut", list(E.CUTS))
def test_every_step_runs_for_every_kind_and_cut(step, kind, cut):
    at = _run(step=step, kind_select=kind, cut_select=cut, m_slider=5, n_slider=4)
    _ok(at)
    assert at.session_state["ell_step"] == step


def test_iteration_slider_walks_through_the_two_dimensional_picture():
    at = _run(kind_select="textbook", iter_k=1)
    _ok(at)
    slider = next(s for s in at.slider if s.key == "iter_k")
    assert slider.max == 100 and slider.min == 1
    for k in (1, 3, 40, 100):
        next(s for s in at.slider if s.key == "iter_k").set_value(k).run()
        _ok(at)
        assert any(f"Iteration {k} von 100" in m.value for m in at.markdown)
    at.radio(key="zoom_select").set_value("Ganze Startkugel").run()
    _ok(at)
    assert len(at.get("plotly_chart")) == 2


def test_higher_dimensions_show_a_table_instead_of_the_picture():
    at = _run(kind_select="random", m_slider=6, n_slider=5)
    _ok(at)
    assert at.dataframe and len(at.get("plotly_chart")) == 1 and not any(r.key == "zoom_select" for r in at.radio)
    next(s for s in at.slider if s.key == "iter_k").set_value(10).run()
    _ok(at)
    assert any("Iteration 10 von" in m.value for m in at.markdown)


def test_infeasible_and_unbounded_show_their_messages():
    inf = _run(kind_select="infeasible", cut_select="deep")
    _ok(inf)
    assert any("unzulässig" in i.value for i in inf.info)
    unb = _run(kind_select="unbounded", cut_select="deep", select_select="most")
    _ok(unb)
    assert any("Kugelrand" in w.value or "Startkugel" in w.value for w in unb.warning)


def test_scaling_shows_the_numerical_failure():
    at = _run(kind_select="random", m_slider=6, n_slider=8, seed_input=35, scale_select=6)
    _ok(at)
    assert any("Numerisch gescheitert" in e.value for e in at.error)


def test_step_two_on_demand_experiments():
    at = _run(step=2)
    _ok(at)
    assert len(at.get("plotly_chart")) == 1
    for key, extra in (("size_start", "Iterationen / n²"), ("eps_start", None), ("radius_start", None), ("bound_start", "Anteil der Schranke")):
        _click(at, key)
        _ok(at)
        if extra:
            assert any(extra in c for d in at.dataframe for c in d.value.columns), key
    assert len(at.get("plotly_chart")) == 5


def test_step_three_and_four_on_demand_experiments():
    at = _run(step=3)
    _ok(at)
    assert any("Fache" in m.value for m in at.markdown)
    _click(at, "flops_start")
    _ok(at)
    assert len(at.get("plotly_chart")) == 1 and list(at.dataframe[-1].value["n"]) == list(C.SWEEP_SIZES)
    _click(at, "cube_start")
    _ok(at)
    assert len(at.get("plotly_chart")) == 2 and any(c == "Pivots (Dantzig)" for d in at.dataframe for c in d.value.columns)
    four = _run(step=4)
    _ok(four)
    assert len(four.get("plotly_chart")) == 1
    for key in ("cut_start", "scale_start", "eq_start"):
        _click(four, key)
        _ok(four)
    assert len(four.get("plotly_chart")) == 4
    scale_table = [d for d in four.dataframe if "Ellipsoid abgebrochen" in d.value.columns][0].value
    assert list(scale_table["Spalten über"])[-1] == "10^12"


@pytest.mark.parametrize("kw", [dict(kind_select="random", m_slider=C.M_MIN, n_slider=C.N_MIN), dict(kind_select="random", m_slider=C.M_MAX, n_slider=C.N_MAX), dict(kind_select="mixed", m_slider=C.M_MAX, n_slider=C.N_MAX),
                                dict(kind_select="mixed", m_slider=C.M_MIN, n_slider=C.N_MIN), dict(kind_select="random", m_slider=C.M_MIN, n_slider=C.N_MAX),
                                dict(kind_select="klee_minty", n_slider=C.N_MAX), dict(kind_select="klee_minty", n_slider=C.N_MIN)])
def test_extreme_sizes_run_on_every_step(kw):
    for step in (1, 2, 3, 4):
        _ok(_run(step=step, **kw))


@pytest.mark.parametrize("kw", [dict(eps_select=0), dict(eps_select=len(C.EPS_EXPS) - 1), dict(radius_select=len(C.RADIUS_FACTORS) - 1), dict(scale_select=len(C.SCALE_EXPS) - 1), dict(eps_select=len(C.EPS_EXPS) - 1, scale_select=3)])
def test_extreme_accuracy_radius_and_scale_on_every_kind(kw):
    for kind in ("textbook", "centre", "random", "mixed", "infeasible", "unbounded"):
        _ok(_run(step=1, kind_select=kind, m_slider=5, n_slider=5, **kw))


def test_dice_button_changes_the_seed():
    at = _run(kind_select="random")
    old = at.session_state["seed_input"]
    next(b for b in at.button if b.label == "🎲 Neue Instanz generieren").click().run()
    _ok(at)
    assert at.session_state["seed_input"] != old and at.session_state["seed_widget"] == at.session_state["seed_input"]


def test_permalink_values_are_clamped_and_invalid_choices_fall_back_to_the_default():
    at = AppTest.from_file(APP, default_timeout=300)
    for k, v in dict(m="999", n="1", step="9", kind="nope", cut="x", sel="zufall", eps="99", rad="-2", scale="50", seed="-4").items():
        at.query_params[k] = v
    at.run()
    _ok(at)
    ss = at.session_state
    assert (ss["m_slider"], ss["n_slider"], ss["ell_step"], ss["kind_select"], ss["cut_select"], ss["select_select"], ss["eps_select"], ss["radius_select"], ss["scale_select"], ss["seed_input"]) == (
        C.M_MAX, C.N_MIN, 1, "centre", "central", "first", len(C.EPS_EXPS) - 1, 0, len(C.SCALE_EXPS) - 1, 0)


def test_permalink_accepts_valid_values():
    at = AppTest.from_file(APP, default_timeout=300)
    for k, v in dict(kind="mixed", m="8", n="7", seed="7", cut="deep", sel="most", eps="1", rad="2", scale="3", step="3").items():
        at.query_params[k] = v
    at.run()
    _ok(at)
    ss = at.session_state
    assert (ss["kind_select"], ss["m_slider"], ss["n_slider"], ss["seed_input"], ss["cut_select"], ss["select_select"], ss["eps_select"], ss["radius_select"], ss["scale_select"], ss["ell_step"]) == (
        "mixed", 8, 7, 7, "deep", "most", 1, 2, 3, 3)


def test_sidebar_shows_the_controls_that_belong_to_the_instance():
    fixed = _run()
    assert not any(w.key in ("m_widget", "n_widget") for w in fixed.slider) and not any(n.key == "seed_widget" for n in fixed.number_input)
    cube = _run(kind_select="klee_minty")
    assert any(w.key == "n_widget" for w in cube.slider) and not any(w.key == "m_widget" for w in cube.slider) and not any(n.key == "seed_widget" for n in cube.number_input)
    rnd = _run(kind_select="random")
    assert any(w.key == "m_widget" for w in rnd.slider) and any(w.key == "n_widget" for w in rnd.slider) and any(n.key == "seed_widget" for n in rnd.number_input)


def test_changing_kind_and_step_on_later_steps_does_not_crash():
    for step in (1, 2, 3, 4):
        at = _run(step=step)
        _ok(at)
        for kw in (dict(kind_select="mixed", m_slider=8, n_slider=8), dict(kind_select="infeasible"), dict(kind_select="unbounded"), dict(kind_select="degenerate"), dict(kind_select="textbook"),
                   dict(kind_select="random", m_slider=4, n_slider=3), dict(kind_select="centre", cut_select="deep")):
            for k, v in kw.items():
                at.session_state[k] = v
            at.run()
            _ok(at)


def test_footer_limits_and_literature_are_present():
    at = _run()
    assert any("Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net)" in c.value for c in at.caption)
    assert any("Wo die Annahmen enden" in s.value for s in at.subheader)
    assert any("Khachiyan" in m.value and "Grötschel" in m.value for e in at.expander for m in e.markdown)
