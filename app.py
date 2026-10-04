"""Ellipsoid-Methode – polynomial in der Theorie, in der Praxis weit hinter dem Simplex - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Siebtes Stück der Lineare-Programmierung-Reihe der "Konzepte"-Reihe: Khachiyans Ellipsoid-Methode löst LPs ohne Ecken und Pivots: ein Ellipsoid, das die Lösung sicher enthält, wird bei jeder Iteration durch einen Schnitt
verkleinert. Die Demo zeigt die schrumpfenden Ellipsen, zählt die Iterationen und stellt sie dem Simplex gegenüber.

Lauffähig mit: streamlit run app.py
"""

import math

import pandas as pd
import streamlit as st

import ell_constants as C
import ell_ellipsoid as E
import ell_evaluation as ev
import ell_scenario as S
from ell_evaluation import Settings, analyse
from ell_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    store_from_widget,
    sync_query_params,
)
from ell_visualization import (
    build_bound,
    build_cube,
    build_cuts,
    build_ellipses,
    build_eps,
    build_equality,
    build_flops,
    build_gap,
    build_min_eig,
    build_radius,
    build_scale,
    build_size,
    build_volume,
)

st.set_page_config(page_title="Ellipsoid-Methode – Sebastian Hanisch", layout="wide")


def num(x, digits=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{0.0 if abs(x) < 5e-13 else x:.{digits}f}"


def big(x):
    return f"{x:,.0f}".replace(",", ".")


STATUS_TEXT = {"optimal": "Optimum", "infeasible": "unzulässig", "ball": "Kugelrand", "numerical": "numerisch gescheitert", "limit": "Iterationsgrenze erreicht"}

st.title("🥚 Ellipsoid-Methode – ein Ei, das die Lösung einschließt")
st.markdown(
    """
**Siebtes Stück der Lineare-Programmierung-Reihe.** Bisher rechnete jedes Stück Simplex-Varianten: von Ecke zu Ecke, im schlimmsten Fall exponentiell viele Pivots (Klee-Minty). **Khachiyan (1979)** zeigte, dass sich LPs auch **polynomial** lösen lassen - mit einem
ganz anderen Verfahren: ein **Ellipsoid**, das die Lösung sicher enthält, wird in jeder Iteration durch eine Nebenbedingung angeschnitten und durch das kleinste Ellipsoid ersetzt, das die Restmenge umschließt. Das Volumen schrumpft je Iteration mindestens
um den Faktor exp(−1/(2(n+1))). Die Theorie sagt "polynomial", die Praxis sagt "unbrauchbar" - hier **gemessen**, nicht behauptet. Vier Fragen: **(1) Die Ellipsen** - was passiert in einer Iteration? **(2) Iterationen** - wie viele braucht es, und wovon hängt es ab?
**(3) Gegen den Simplex** - was kostet es in Operationen? **(4) Grenzen** - Schnittarten, Skalierung, Gleichungen.
"""
)
st.caption("Kind der Wurzel [Tableau-Simplex](https://github.com/sebastian-hanisch/tableau-simplex-demo); Kontrast zu [Klee-Minty](https://github.com/sebastian-hanisch/klee-minty-demo). Folgestücke: [Innere Punkte](https://github.com/sebastian-hanisch/innere-punkte-demo), [PDLP](https://github.com/sebastian-hanisch/pdlp-demo), [Präsolve](https://github.com/sebastian-hanisch/praesolve-demo).")

with st.expander("So funktioniert die Ellipsoid-Methode", expanded=True):
    st.markdown(
        """
1. **Start:** eine Kugel, die alle zulässigen Punkte enthält (hier um den Kasten, den die Ressourcen erlauben).
2. **Mittelpunkt prüfen:** verletzt der Mittelpunkt eine Nebenbedingung, gilt der **Machbarkeitsschnitt**: die Lösung liegt auf der anderen Seite. Ist er zulässig, gilt der **Objektivschnitt**: nur Punkte mit mindestens so gutem Zielwert kommen noch infrage.
3. **Halbieren und umschließen:** das kleinste Ellipsoid, das die verbliebene Hälfte umschließt, ist das neue. Ein **zentraler** Schnitt geht durch den Mittelpunkt; ein **tiefer** Schnitt geht bis an die verletzte Nebenbedingung und spart Iterationen.
4. **Zertifikat:** jedes optimale x liegt im Ellipsoid, also ist c·x + √(cᵀPc) eine obere Schranke. Sobald sie nur noch um ε vom besten gefundenen Wert abweicht, ist das Optimum bewiesen. Schneidet ein Schnitt das ganze Ellipsoid ab, ist die Instanz **unzulässig**.
5. **Gleichungen** werden zu zwei Ungleichungen mit einer kleinen Aufweitung (sonst hat die zulässige Menge kein Inneres und das Volumenargument bricht zusammen).
        """
    )

if C.PRESETS:
    st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
    preset_names = list(C.PRESETS.keys())
    for row in (preset_names[:4], preset_names[4:8], preset_names[8:]):
        if not row:
            continue
        cols = st.columns(len(row))
        for col, name in zip(cols, row):
            with col:
                st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

ss = st.session_state
with st.sidebar:
    st.header("⚙️ Einstellungen")
    kind = st.selectbox("Instanz", options=list(S.KINDS), format_func=lambda v: S.KIND_LABELS[v], key="kind_select",
                        help="Lehrbuchbeispiel, Zentrum und entartete Ecke sind fest; Zufall (≤-Ressourcen) und Mischung (mit ≥ und =) sind regelbar; der Klee-Minty-Würfel hat nur die Größe n. Unzulässig und Unbeschränkt zeigen, was die Methode dann meldet.")
    cube = kind in S.CUBE_KINDS
    random_kind = kind not in S.FIXTURE_KINDS and not cube
    if random_kind:
        m = st.slider("Ressourcen m", *bounds("m_slider"), value=int(ss["m_slider"]), key="m_widget", on_change=store_from_widget, args=("m_slider",), help="Zahl der Bedingungen.")
    else:
        m = C.DEFAULT_M
    if random_kind or cube:
        n = st.slider("Dienste n", *bounds("n_slider"), value=int(ss["n_slider"]), key="n_widget", on_change=store_from_widget, args=("n_slider",), help="Zahl der Variablen = Dimension des Ellipsoids (beim Würfel auch die Zahl der Ressourcen).")
    else:
        n = C.DEFAULT_N
    if random_kind:
        seed = st.number_input("Zufalls-Seed der Instanz", *bounds("seed_input"), value=int(ss["seed_input"]), key="seed_widget", step=1, on_change=store_from_widget, args=("seed_input",))
        st.button("🎲 Neue Instanz generieren", width="stretch", on_click=randomize_seed)
    else:
        seed = C.DEFAULT_SEED
    cut = st.radio("Schnittart", options=list(E.CUTS), format_func=lambda c: C.CUT_LABELS[c], key="cut_select", help="Tiefe Schnitte nutzen, wie weit der Mittelpunkt die Nebenbedingung verletzt.")
    select = st.radio("Wahl der verletzten Zeile", options=list(E.SELECTS), format_func=lambda c: C.SELECT_LABELS[c], key="select_select", help="Welche der verletzten Nebenbedingungen den Schnitt liefert.")
    eps_i = st.select_slider("Genauigkeit ε", options=list(range(len(C.EPS_EXPS))), format_func=C.eps_label, key="eps_select", help="Relative Lücke zwischen oberer Schranke und bestem Wert, bei der das Verfahren stoppt.")
    radius_i = st.select_slider("Startkugel", options=list(range(len(C.RADIUS_FACTORS))), format_func=C.radius_label, key="radius_select", help="Faktor auf die halbe Diagonale des Kastens, der die zulässige Menge enthält.")
    scale_i = st.select_slider("Skalierung der Spalten", options=list(range(len(C.SCALE_EXPS))), format_func=C.scale_label, key="scale_select",
                               help="Die Dienste werden mit Faktoren zwischen 1 und 10^k umgerechnet (Optimalwert gleich, Zahlen sehr verschieden groß): ein Test für die Numerik.")

sync_query_params({"kind_select": kind, "m_slider": int(ss["m_slider"]), "n_slider": int(ss["n_slider"]), "seed_input": int(ss["seed_input"]), "cut_select": cut, "select_select": select, "eps_select": int(eps_i),
                   "radius_select": int(radius_i), "scale_select": int(scale_i), "ell_step": int(ss["ell_step"])})

settings = Settings(kind, int(m), int(n), int(seed), cut, select, int(eps_i), int(radius_i), int(scale_i))
with st.spinner("Rechne..."):
    a = analyse(settings)
res, inst = a.res, a.inst

# --- In Aktion ---------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ein Ellipsoid schließt die Lösung ein")
step = st.select_slider("Schritt", options=list(C.STEPS), key="ell_step", format_func=lambda s: C.STEPS[s])

if res.status == "optimal":
    rel = abs(a.obj_error) / (1.0 + abs(a.ref_obj)) if math.isfinite(a.obj_error) else float("nan")
    if math.isfinite(rel) and rel > 1e-3:
        st.warning(f"⚠️ Die Ellipsoid-Methode meldet ein Optimum ({num(res.obj)}), aber der Optimalwert weicht um {rel:.1%} vom Referenzwert ({num(a.ref_obj)}) ab - die schlechte Skalierung hat die Toleranzen unterlaufen.")
    else:
        st.success(f"✅ Optimum **{num(res.obj)}** nach **{big(res.iterations)}** Iterationen; die obere Schranke {num(res.upper)} beweist es bis ε = {settings.eps:g} (Simplex: {num(a.ref_obj)} nach {a.simplex.pivots} Pivots).")
elif res.status == "infeasible":
    st.info(f"Die Ellipsoid-Methode erkennt die Instanz nach {res.iterations} Iterationen als **unzulässig**: {res.note}.")
elif res.status == "ball":
    st.warning(f"⚠️ {res.note} (Zielwert {num(res.obj)}). Der Simplex meldet: {STATUS_TEXT.get(a.simplex.status, a.simplex.status)}. Ohne endliche Schranke gibt es keine Startkugel, die die Menge sicher enthält; das Ergebnis ist kein Beweis.")
elif res.status == "numerical":
    st.error(f"❌ Numerisch gescheitert nach {big(res.iterations)} Iterationen: {res.note}." + (f" Bester zulässiger Wert bis dahin: {num(res.obj)} (Referenz {num(a.ref_obj)})." if res.x and math.isfinite(a.ref_obj) else ""))
else:
    st.warning(f"Iterationsgrenze ({big(C.MAX_ITER)}) erreicht ohne Zertifikat" + (f"; bester Wert {num(res.obj)}." if res.x else "; kein zulässiger Punkt gefunden."))
if kind == "mixed" and any(x == S.EQ for x in inst.senses):
    st.caption(f"Gleichungen werden als zwei Ungleichungen mit der Aufweitung {C.EQ_TOL_APP:g} behandelt.")

G_rows, h_rows, _n_struct, origin = E.rows_of(inst, C.EQ_TOL_APP)


def row_name(k):
    o = origin[k]
    return f"{inst.names[-1 - o]} ≥ 0" if o < 0 else inst.row_names[o]


if step == 1:
    total = res.iterations
    if total == 0:
        st.info("Es gab keine Iteration.")
    else:
        if "iter_k" in ss:
            ss["iter_k"] = min(max(1, int(ss["iter_k"])), total)
        k = st.slider("Iteration", 1, total, key="iter_k", help="1 = Startkugel; danach Schnitt für Schnitt.") if total > 1 else 1
        kind_k, row_k, alpha_k = res.cuts[k - 1]
        cq = res.centres[k - 1]
        if kind_k == "feasibility":
            what = f"Der Mittelpunkt verletzt **{row_name(row_k)}**: Machbarkeitsschnitt" + (f" (Tiefe α = {alpha_k:.2f})" if a.cut == "deep" else "") + "."
        else:
            what = f"Der Mittelpunkt ist zulässig (Zielwert {num(res.best_path[k - 1])}): Objektivschnitt, nur bessere Punkte bleiben."
        gap_txt = f" Obere Schranke {num(res.upper_path[k - 1])}, bester Wert {num(res.best_path[k - 1])}." if math.isfinite(res.best_path[k - 1]) else " Noch kein zulässiger Punkt gefunden."
        st.markdown(f"**Iteration {k} von {big(total)}:** {what}{gap_txt}")
        if inst.n == 2:
            zoom = st.radio("Ansicht", options=["Ausschnitt um die zulässige Menge", "Ganze Startkugel"], horizontal=True, key="zoom_select") == "Ausschnitt um die zulässige Menge"
            st.plotly_chart(build_ellipses(a, k, zoom), width="stretch", key=f"s1_ellipse_{k}_{zoom}")
            st.caption("Orange: das Ellipsoid der Iteration und sein Mittelpunkt; violett: die bisherigen Mittelpunkte; rot gepunktet: die Schnittgerade; grün: zulässige Menge; grau gestrichelt: Startkugel; Stern: Optimum des Simplex.")
        else:
            st.dataframe(pd.DataFrame({"Größe": ["Mittelpunkt (erste Koordinaten)", "Machbarkeitsschnitte bisher", "Objektivschnitte bisher", "erster zulässiger Mittelpunkt"],
                                       "Wert": [", ".join(num(v) for v in cq[:4]) + (" …" if inst.n > 4 else ""), str(sum(1 for c in res.cuts[:k] if c[0] == "feasibility")), str(sum(1 for c in res.cuts[:k] if c[0] == "objective")),
                                                (str(res.first_feasible) if 0 < res.first_feasible <= k else "noch nicht")]}), hide_index=True, width="stretch")
            st.caption("Ab drei Dienstvariablen gibt es kein Bild mehr; die Volumenkurve unten zeigt trotzdem, wie das Ellipsoid schrumpft.")
        st.markdown(f"**Volumen:** je zentralem Schnitt höchstens ×exp(−1/(2(n+1))) = ×{math.exp(-1.0 / (2 * (inst.n + 1))):.3f}; tiefe Schnitte schrumpfen mehr.")
        st.plotly_chart(build_volume(res, inst.n), width="stretch", key="s1_volume")
elif step == 2:
    st.markdown(f"**Diese Instanz:** {big(res.iterations)} Iterationen; erster zulässiger Mittelpunkt in Iteration {res.first_feasible if res.first_feasible > 0 else '-'}; {res.feasibility_cuts} Machbarkeits- und {res.objective_cuts} Objektivschnitte.")
    if res.best_path:
        st.plotly_chart(build_gap(res, a.ref_obj, settings.eps), width="stretch", key="s2_gap")
        st.caption("Blau: die Zertifikatslücke (obere Schranke minus bester Wert) - sie fällt gleichmäßig um dieselbe Zehnerpotenz je feste Zahl von Iterationen; orange: der tatsächliche Fehler gegen das Optimum des Simplex; gepunktet: verlangte Genauigkeit.")
    st.markdown("**Wovon hängt die Zahl der Iterationen ab?** (🔬 auf Abruf; Zufallsinstanzen, Median über fünf feste Instanzen)")
    tok_size = (settings.cut, settings.select, settings.eps_i, settings.radius_i)
    if st.button("Iterationen über die Größe n berechnen", key="size_start"):
        ss["size_done"] = tok_size
    if ss.get("size_done") == tok_size:
        with st.spinner("Rechne..."):
            rows = ev.size_sweep(settings)
        st.plotly_chart(build_size(rows), width="stretch", key="s2_size")
        st.dataframe(pd.DataFrame([{"n": r["n"], "Iterationen": big(r["iterations"]), "Spanne": f"{big(r['min'])} bis {big(r['max'])}", "Iterationen / n²": num(r["per_n2"], 1)} for r in rows]), hide_index=True, width="stretch")
        st.caption("Die Iterationen wachsen wie n²: das Verhältnis Iterationen / n² bleibt über alle Größen fast gleich.")
    tok_eps = (settings.n, settings.cut, settings.select, settings.radius_i)
    if st.button("Iterationen über die Genauigkeit berechnen", key="eps_start"):
        ss["eps_done"] = tok_eps
    if ss.get("eps_done") == tok_eps:
        with st.spinner("Rechne..."):
            sw = ev.eps_sweep(settings)
        st.plotly_chart(build_eps(sw), width="stretch", key="s2_eps")
        st.caption(f"Bei n = {settings.n}: {num(sw['slope'], 0)} Iterationen je Zehnerpotenz Genauigkeit, die Theorie 2n(n+1) ln 10 = {num(sw['theory_slope'], 0)} erwartet; gemessen {sw['share']:.0%} davon. Jede weitere Stelle kostet also gleich viel.")
    tok_rad = (settings.n, settings.cut, settings.select, settings.eps_i)
    if st.button("Iterationen über die Startkugel berechnen", key="radius_start"):
        ss["radius_done"] = tok_rad
    if ss.get("radius_done") == tok_rad:
        with st.spinner("Rechne..."):
            rows = ev.radius_sweep(settings)
        st.plotly_chart(build_radius(rows), width="stretch", key="s2_radius")
        st.caption("Eine größere Startkugel kostet nur logarithmisch viele Iterationen mehr - vor allem bis zum ersten zulässigen Mittelpunkt.")
    tok_bound = (settings.cut, settings.select, settings.eps_i, settings.radius_i)
    if st.button("Theorie-Schranke prüfen", key="bound_start"):
        ss["bound_done"] = tok_bound
    if ss.get("bound_done") == tok_bound:
        with st.spinner("Rechne..."):
            rows = ev.feasibility_bound(settings)
        st.plotly_chart(build_bound(rows), width="stretch", key="s2_bound")
        st.dataframe(pd.DataFrame([{"n": r["n"], "erster zulässiger Mittelpunkt": num(r["first_feasible"], 0), "Schranke 2n(n+1) ln(r0/r)": num(r["bound"], 0), "Anteil der Schranke": f"{r['share']:.1%}"} for r in rows]), hide_index=True, width="stretch")
        st.caption("Die Schranke folgt aus dem Volumen: solange kein Mittelpunkt zulässig ist, muss das Ellipsoid mindestens so groß sein wie eine Kugel vom Radius r (die größte in die zulässige Menge passende, mit dem Simplex berechnet). Sie ist ein Worst-Case-Wert und liegt weit über dem, was auftritt.")
elif step == 3:
    st.markdown("**Ellipsoid gegen Simplex auf dieser Instanz** (Operationsmodell, kein Wandzeit-Vergleich):")
    c1, c2, c3 = st.columns(3)
    c1.metric("Iterationen / Pivots", f"{big(res.iterations)} / {a.simplex.pivots}", delta="Ellipsoid / Simplex", delta_color="off")
    c2.metric("Operationen Ellipsoid", big(a.flops_ell), delta="Modell", delta_color="off")
    c3.metric("Operationen Simplex", big(a.flops_simplex), delta="Modell", delta_color="off")
    if a.flops_simplex:
        st.markdown(f"Der Ellipsoid braucht auf dieser Instanz das **{a.flops_ell / a.flops_simplex:,.0f}-Fache** der Operationen des dichten Tableau-Simplex.".replace(",", "."))
    st.caption("Modell: eine Iteration kostet 2m'n (Zeilenwerte) + 6n² + 4n Operationen (m' = Zahl der Ungleichungen einschließlich x ≥ 0); ein Pivot des dichten Tableaus 2(m+1)(Spalten+1).")
    tok = (settings.cut, settings.select, settings.eps_i, settings.radius_i)
    if st.button("Operationen über die Größe berechnen", key="flops_start"):
        ss["flops_done"] = tok
    if ss.get("flops_done") == tok:
        with st.spinner("Rechne..."):
            rows = ev.size_sweep(settings)
        st.plotly_chart(build_flops(rows), width="stretch", key="s3_flops")
        st.dataframe(pd.DataFrame([{"n": r["n"], "Iterationen": big(r["iterations"]), "Pivots": num(r["pivots"], 0), "Operationen Ellipsoid": big(r["flops_ell"]), "Operationen Simplex": big(r["flops_simplex"]), "Verhältnis": big(r["ratio"])} for r in rows]),
                     hide_index=True, width="stretch")
        st.caption("Zufallsinstanzen m = n, Median über fünf feste Instanzen, ε wie in der Seitenleiste. Der Simplex braucht wenige Pivots, der Ellipsoid Tausende von Iterationen; einen Kreuzungspunkt gibt es in diesem Bereich nicht.")
    st.markdown("**Der schlimmste Fall des Simplex:** der Klee-Minty-Würfel aus Stück 3 (🔬 auf Abruf).")
    if st.button("Auf dem Würfel vergleichen", key="cube_start"):
        ss["cube_done"] = tok
    if ss.get("cube_done") == tok:
        with st.spinner("Rechne..."):
            cs = ev.cube_sweep(settings)
        st.plotly_chart(build_cube(cs), width="stretch", key="s3_cube")
        st.dataframe(pd.DataFrame([{"n": r["n"], "Iterationen": big(r["iterations"]), "Pivots (Dantzig)": big(r["pivots"]), "Operationen Ellipsoid": big(r["flops_ell"]), "Operationen Simplex": big(r["flops_simplex"]), "Verhältnis": num(r["ratio"])} for r in cs["rows"]]),
                     hide_index=True, width="stretch")
        st.caption((f"Ab n = {cs['crossover']} braucht das Ellipsoid weniger Operationen als der Simplex mit der Dantzig-Regel" if cs["crossover"] else "In diesem Bereich braucht das Ellipsoid immer mehr Operationen als der Simplex")
                   + ": der Simplex besucht alle 2^n Ecken des Würfels, das Ellipsoid wächst mit einer Potenz von n. Gegen Steepest Edge (ein Pivot, Stück 3) gäbe es keinen Kreuzungspunkt.")
else:
    st.markdown("**Wie gut konditioniert bleibt das Ellipsoid?** Der kleinste Eigenwert von P zeigt, wie dünn es in einer Richtung wird:")
    if res.min_eig:
        st.plotly_chart(build_min_eig(res), width="stretch", key="s4_eig")
    tok_cut = (settings.n, settings.eps_i, settings.radius_i)
    if st.button("Schnittarten vergleichen", key="cut_start"):
        ss["cut_done"] = tok_cut
    if ss.get("cut_done") == tok_cut:
        with st.spinner("Rechne..."):
            rows = ev.cut_compare(settings)
        st.plotly_chart(build_cuts(rows), width="stretch", key="s4_cuts")
        st.caption(f"Zufallsinstanzen m = n = {settings.n}, Median über fünf feste Instanzen. Tiefe Schnitte sparen {rows[2]['saving']:.0%} (erste Zeile) bzw. {rows[3]['saving']:.0%} (stärkste Zeile) gegen zentrale Schnitte; die Wahl der Zeile ändert bei zentralen Schnitten kaum etwas.")
    tok_scale = (settings.kind, settings.m, settings.n)
    if st.button("Schlechte Skalierung testen", key="scale_start"):
        ss["scale_done"] = tok_scale
    if ss.get("scale_done") == tok_scale:
        with st.spinner("Rechne..."):
            rows = ev.scale_sweep(settings)
        st.plotly_chart(build_scale(rows), width="stretch", key="s4_scale")
        st.dataframe(pd.DataFrame([{"Spalten über": f"10^{r['k']}", "Iterationen": big(r["iterations"]), "Ellipsoid abgebrochen": f"{r['ell_broke']} von 5", "falsch (Ellipsoid)": f"{r['ell_wrong']} von 5", "falsch (Simplex)": f"{r['sim_wrong']} von 5"} for r in rows]), hide_index=True, width="stretch")
        st.caption("Dieselben Instanzen, deren Spalten mit Faktoren zwischen 1 und 10^k umgerechnet sind (der Optimalwert bleibt gleich). 'Falsch' heißt: relativer Fehler des Optimalwerts über 0.1 %. Die Iterationen wachsen linear mit k; ab großen k geraten die absoluten Toleranzen beider Löser aus dem Tritt; das Ellipsoid bricht dann durch Rundung ab.")
    tok_eq = (settings.kind, settings.m, settings.n, settings.seed)
    if st.button("Gleichungen: Aufweitung verkleinern", key="eq_start"):
        ss["eq_done"] = tok_eq
    if ss.get("eq_done") == tok_eq:
        with st.spinner("Rechne..."):
            sw = ev.equality_sweep(settings)
        st.plotly_chart(build_equality(sw), width="stretch", key="s4_eq")
        st.caption(f"Mischinstanz {sw['m']} × {sw['n']} (Seed {sw['seed']}) mit {sw['eq_rows']} Gleichung(en). Je kleiner die Aufweitung, desto dünner die zulässige Menge und desto später findet die Methode einen zulässigen Mittelpunkt; ist sie zu dünn, scheitert die Numerik.")

st.markdown("---")
st.markdown("## ⚙️ Der gewählte Fall")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Iterationen", big(res.iterations), delta=STATUS_TEXT[res.status], delta_color="off")
m2.metric("Pivots (Simplex)", str(a.simplex.pivots), delta=STATUS_TEXT.get(a.simplex.status, a.simplex.status), delta_color="off")
m3.metric("Operationen Ell. / Simplex", f"{a.flops_ell / a.flops_simplex:,.0f}×".replace(",", ".") if a.flops_simplex else "-", delta="Modell", delta_color="off")
m4.metric("Ergebnis", num(res.obj) if res.x else "-", delta=("Referenz " + num(a.ref_obj)) if math.isfinite(a.ref_obj) else "keine Referenz", delta_color="off")

st.markdown("---")

# --- Grenzen -----------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Polynomial heißt schnell.** | Die Zahl der Iterationen wächst wie n² ln(1/ε), jede Iteration kostet n² bis mn Operationen; der Simplex braucht auf denselben Instanzen ein paar Pivots. Im Operationsmodell liegt das Ellipsoid auf Zufallsinstanzen mindestens um den Faktor 150 zurück; nur auf dem Klee-Minty-Würfel dreht es sich, und dort erst bei n = 13 bis 14. | Innere Punkte |
| **Die Startkugel enthält die zulässige Menge.** | Ohne endliche Schranke (unbeschränkte Instanz) gibt es keine Kugel; die Methode endet am Rand der künstlichen Kugel und beweist nichts. | Vorverarbeitung |
| **Die zulässige Menge hat ein Inneres.** | Gleichungen werden aufgeweitet; je dünner, desto mehr Iterationen bis zum ersten zulässigen Punkt, und irgendwann bricht die Numerik. | Innere Punkte |
| **Gleitkomma ist genug.** | Khachiyans Beweis rechnet mit gerundeten Zahlen und Fehlerschranken in exakter Arithmetik; hier läuft alles in Gleitkomma. Schlechte Skalierung unterläuft die absoluten Toleranzen, und ab ε = 10^-13 lässt sich das Zertifikat nicht mehr schließen. | Präsolve und Skalierung |
| **Der Ellipsoid löst Optimierung.** | Hier durch Objektivschnitte; die Theorie (Grötschel, Lovász, Schrijver) zeigt die Gleichwertigkeit von Optimieren und Trennen für viel allgemeinere Mengen - der eigentliche Wert der Methode. | Kombinatorische Optimierung |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Ellipsoid:** $E(x, P) = \{z : (z - x)^\top P^{-1} (z - x) \le 1\}$. **Zentraler Schnitt** $a^\top z \le a^\top x$: $x^+ = x - \frac{1}{n+1}\frac{Pa}{\sqrt{a^\top P a}}$, $P^+ = \frac{n^2}{n^2-1}\left(P - \frac{2}{n+1}\frac{Paa^\top P}{a^\top P a}\right)$; das Volumen
sinkt um den Faktor $\left(\left(\tfrac{n}{n+1}\right)^{n+1}\left(\tfrac{n}{n-1}\right)^{n-1}\right)^{1/2} \le e^{-1/(2(n+1))}$. **Tiefer Schnitt** $a^\top z \le \beta$ mit $\alpha = (a^\top x - \beta)/\sqrt{a^\top P a}$: $x^+ = x - \frac{1+n\alpha}{n+1}\frac{Pa}{\sqrt{a^\top Pa}}$,
$P^+ = \frac{n^2(1-\alpha^2)}{n^2-1}\left(P - \frac{2(1+n\alpha)}{(n+1)(1+\alpha)}\frac{Paa^\top P}{a^\top Pa}\right)$, gültig für $\alpha < 1$ (bei $\alpha \ge 1$ ist die Schnittmenge leer). **Zertifikat:** $c^\top z^* \le c^\top x + \sqrt{c^\top P c}$.
**Iterationsschranke:** enthält die zulässige Menge eine Kugel vom Radius $r$, ist nach $2n(n+1)\ln(r_0/r)$ Iterationen ein zulässiger Mittelpunkt gefunden.

**Literatur.** Khachiyan, L. G. (1979). *A polynomial algorithm in linear programming.* Doklady Akademii Nauk SSSR 244(5), 1093-1096. Bland, R. G., Goldfarb, D., & Todd, M. J. (1981). *The ellipsoid method: a survey.* Operations Research 29(6), 1039-1091.
Grötschel, M., Lovász, L., & Schrijver, A. (1988). *Geometric Algorithms and Combinatorial Optimization.* Springer.

Implementiert in `ell_ellipsoid.py` (Schnitte, Zertifikate), `ell_algorithm.py` (Tableau-Simplex als Vergleich), `ell_evaluation.py`, `ell_scenario.py`.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Lineare Programmierung: vom Tableau zum Crossover](https://sebastianhanisch.net/konzepte-lineare-programmierung.html)."
)
