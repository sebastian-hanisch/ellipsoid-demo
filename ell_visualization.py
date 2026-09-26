"""Plotly-Abbildungen: schrumpfende Ellipsen (n = 2), Zertifikatslücke, Volumen, Iterationen über n / Genauigkeit / Startkugel, Operationen gegen den Simplex, Schnittarten, Theorie-Schranke, Skalierung, Gleichungen.
Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht zoomen; bei gleichem Achsenmaßstab (scaleanchor) gibt es keine expliziten Bereiche."""

import math

import numpy as np
import plotly.graph_objects as go

import ell_constants as C
import ell_ellipsoid as E

TEAL, ORANGE, RED, BLUE, GREY, PURPLE = "#2F6B65", "#e8a13a", "#d62728", "#1f4e9c", "#8a8f98", "#7b3fbf"


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height, legend_y=-0.25):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=legend_y), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def feasible_polygon(inst, eq_tol=C.EQ_TOL_APP):
    """Zulässiges Vieleck einer Instanz mit zwei Variablen: Kasten [0, U] (U aus der Startkugel), nacheinander an allen Zeilen abgeschnitten (Sutherland-Hodgman). Leere Liste, wenn nichts übrig bleibt."""
    _centre, _radius, U, _free = E.start_ball(inst, 1.0, eq_tol)
    poly = [np.array([0.0, 0.0]), np.array([U[0], 0.0]), np.array([U[0], U[1]]), np.array([0.0, U[1]])]
    G, h, _ns, _o = E.rows_of(inst, eq_tol)
    for a, beta in zip(G, h):
        out = []
        for i, p in enumerate(poly):
            q = poly[(i + 1) % len(poly)]
            sp, sq = float(a @ p) - beta, float(a @ q) - beta
            if sp <= 0:
                out.append(p)
            if (sp < 0 < sq) or (sq < 0 < sp):
                out.append(p + (q - p) * (sp / (sp - sq)))
        poly = out
        if not poly:
            break
    return poly


def _ellipse(x, P, k=240):
    w, V = np.linalg.eigh(P)
    L = V @ np.diag(np.sqrt(np.maximum(w, 0.0)))
    th = np.linspace(0.0, 2.0 * math.pi, k)
    pts = np.array([x + L @ np.array([math.cos(t), math.sin(t)]) for t in th])
    return pts[:, 0], pts[:, 1]


def _clip_window(xs, ys, win):
    x0, x1, y0, y1 = win
    inside = (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)
    return np.where(inside, xs, np.nan), np.where(inside, ys, np.nan)


def _line_in_window(a, beta, win):
    """Gerade a·z = beta, auf das Fenster begrenzt (Schnittpunkte mit dem Rechteck)."""
    x0, x1, y0, y1 = win
    pts = []
    if abs(a[1]) > 1e-15:
        for xv in (x0, x1):
            yv = (beta - a[0] * xv) / a[1]
            if y0 - 1e-9 <= yv <= y1 + 1e-9:
                pts.append((xv, yv))
    if abs(a[0]) > 1e-15:
        for yv in (y0, y1):
            xv = (beta - a[1] * yv) / a[0]
            if x0 - 1e-9 <= xv <= x1 + 1e-9:
                pts.append((xv, yv))
    if len(pts) < 2:
        return [], []
    pts = sorted(set((round(p[0], 12), round(p[1], 12)) for p in pts))
    return [pts[0][0], pts[-1][0]], [pts[0][1], pts[-1][1]]


def build_ellipses(a, k, zoom):
    """Zwei Variablen: zulässiges Vieleck, Startkugel, bisherige Mittelpunkte, die Ellipse der Iteration k mit Mittelpunkt, Schnittgerade und Höhenlinie durch den Bestwert. zoom: nur ein Fenster um das Vieleck."""
    inst, res = a.inst, a.res
    poly = feasible_polygon(inst)
    centre0, radius0, U, _free = E.start_ball(inst, 1.0)
    if poly:
        px, py = [p[0] for p in poly], [p[1] for p in poly]
    else:
        px, py = [0.0, U[0]], [0.0, U[1]]
    x0, x1, y0, y1 = min(px), max(px), min(py), max(py)
    span = max(x1 - x0, y1 - y0, 1e-9)
    win = (x0 - 0.6 * span, x1 + 0.6 * span, y0 - 0.6 * span, y1 + 0.6 * span)
    fig = go.Figure()
    if poly:
        fig.add_trace(go.Scatter(x=px + [px[0]], y=py + [py[0]], fill="toself", fillcolor="rgba(47,107,101,0.18)", line=dict(color=TEAL, width=2), name="zulässige Menge", hoverinfo="skip"))
    th = np.linspace(0.0, 2.0 * math.pi, 240)
    bx, by = centre0[0] + radius0 * np.cos(th), centre0[1] + radius0 * np.sin(th)
    if zoom:
        bx, by = _clip_window(bx, by, win)
    fig.add_trace(go.Scatter(x=bx, y=by, mode="lines", line=dict(color=GREY, width=1.5, dash="dash"), name="Startkugel", hoverinfo="skip"))
    idx = max(1, min(int(k), res.iterations)) - 1
    cs = np.array(res.centres[: idx + 1])
    if len(cs) > 1:
        cx, cy = (cs[:, 0], cs[:, 1]) if not zoom else _clip_window(cs[:, 0], cs[:, 1], win)
        fig.add_trace(go.Scatter(x=cx, y=cy, mode="lines+markers", marker=dict(size=4, color=PURPLE), line=dict(color=PURPLE, width=1), name="bisherige Mittelpunkte", hoverinfo="skip"))
    ex, ey = _ellipse(res.centres[idx], res.mats[idx])
    if zoom:
        ex, ey = _clip_window(ex, ey, win)
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color=ORANGE, width=3), name=f"Ellipse der Iteration {idx + 1}", hoverinfo="skip"))
    x = res.centres[idx]
    kind, row, alpha = res.cuts[idx]
    G, h, _ns, _o = E.rows_of(inst, C.EQ_TOL_APP)
    if kind == "feasibility":
        av = G[row]
        line_beta = float(av @ x) if a.cut == "central" else float(h[row])
    else:
        av, line_beta = -inst.arrays()[2], -res.best_path[idx]
    full = (centre0[0] - radius0, centre0[0] + radius0, centre0[1] - radius0, centre0[1] + radius0)
    lx, ly = _line_in_window(av, line_beta, win if zoom else full)
    if lx:
        fig.add_trace(go.Scatter(x=lx, y=ly, mode="lines", line=dict(color=RED, width=2, dash="dot"), name="Nebenbedingung" if kind == "feasibility" else "Höhenlinie: bisher bester Wert", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[x[0]], y=[x[1]], mode="markers", marker=dict(size=10, color=ORANGE, line=dict(color="white", width=1)), name="Mittelpunkt", hoverinfo="skip"))
    if a.simplex.status == "optimal":
        xs = a.simplex.x
        fig.add_trace(go.Scatter(x=[xs[0]], y=[xs[1]], mode="markers", marker=dict(size=13, color=BLUE, symbol="star"), name="Optimum (Simplex)", hoverinfo="skip"))
    if zoom:
        fig.add_trace(go.Scatter(x=[win[0], win[1]], y=[win[2], win[3]], mode="markers", marker=dict(size=1, opacity=0), showlegend=False, hoverinfo="skip"))
    fig.update_xaxes(title_text="Dienst 1", scaleanchor="y", scaleratio=1)
    fig.update_yaxes(title_text="Dienst 2")
    return _base(fig, 460, legend_y=-0.3)


def build_gap(res, ref_obj, eps):
    """Zertifikatslücke ub - best und tatsächlicher Fehler ref - best über die Iterationen (log-y); die Linie zeigt die verlangte Genauigkeit."""
    its = np.arange(1, len(res.best_path) + 1)
    best, up = np.array(res.best_path, dtype=float), np.array(res.upper_path, dtype=float)
    gap = np.where(np.isfinite(best) & (up - best > 0), up - best, np.nan)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=its, y=gap, mode="lines", line=dict(color=BLUE, width=3), name="Zertifikatslücke (obere Schranke - bester Wert)"))
    if math.isfinite(ref_obj):
        err = np.where(np.isfinite(best) & (ref_obj - best > 0), ref_obj - best, np.nan)
        fig.add_trace(go.Scatter(x=its, y=err, mode="lines", line=dict(color=ORANGE, width=2), name="tatsächlicher Fehler (Optimum - bester Wert)"))
        fig.add_hline(y=eps * (1 + abs(ref_obj)), line=dict(color=GREY, dash="dot"))
    fig.update_yaxes(type="log", title_text="Wert")
    fig.update_xaxes(title_text="Iteration")
    return _base(fig, 320, legend_y=-0.4)


def build_volume(res, n):
    its = np.arange(0, len(res.log_volume))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=its, y=res.log_volume, mode="lines", line=dict(color=TEAL, width=3), name="ln Volumen(Ellipsoid) - ln Volumen(Start)"))
    fig.add_trace(go.Scatter(x=its, y=-its / (2.0 * (n + 1)), mode="lines", line=dict(color=GREY, dash="dash"), name="Garantie: höchstens -k / (2 (n+1))"))
    fig.update_xaxes(title_text="Iteration")
    fig.update_yaxes(title_text="ln Volumenverhältnis")
    return _base(fig, 300, legend_y=-0.4)


def build_min_eig(res):
    its = np.arange(1, len(res.min_eig) + 1)
    fig = go.Figure(go.Scatter(x=its, y=np.maximum(res.min_eig, 1e-320), mode="lines", line=dict(color=PURPLE, width=3), name="kleinster Eigenwert von P"))
    fig.update_yaxes(type="log", title_text="kleinster Eigenwert von P")
    fig.update_xaxes(title_text="Iteration")
    return _base(fig, 280, legend_y=-0.4)


def build_size(rows):
    """Iterationen über n (Median, Spanne) gegen die Kurve c n^2 mit dem mittleren Iterationen / n^2."""
    ns = [r["n"] for r in rows]
    c = float(np.mean([r["per_n2"] for r in rows]))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ns + ns[::-1], y=[r["max"] for r in rows] + [r["min"] for r in rows][::-1], fill="toself", fillcolor="rgba(47,107,101,0.15)", line=dict(width=0), name="Spanne der fünf Instanzen", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=ns, y=[r["iterations"] for r in rows], mode="lines+markers", line=dict(color=TEAL, width=3), name="Iterationen (Median)"))
    grid = np.linspace(min(ns), max(ns), 60)
    fig.add_trace(go.Scatter(x=grid, y=c * grid ** 2, mode="lines", line=dict(color=GREY, dash="dash"), name=f"{c:.1f} · n²"))
    fig.update_xaxes(title_text="Größe n (Zufall, m = n)")
    fig.update_yaxes(title_text="Iterationen")
    return _base(fig, 320, legend_y=-0.4)


def build_flops(rows):
    ns = [r["n"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ns, y=[r["flops_ell"] for r in rows], mode="lines+markers", line=dict(color=ORANGE, width=3), name="Ellipsoid"))
    fig.add_trace(go.Scatter(x=ns, y=[max(r["flops_simplex"], 1) for r in rows], mode="lines+markers", line=dict(color=TEAL, width=3), name="Simplex"))
    fig.update_yaxes(type="log", title_text="Operationen (Modell)")
    fig.update_xaxes(title_text="Größe n (Zufall, m = n)")
    return _base(fig, 320, legend_y=-0.4)


def build_eps(sweep):
    ks = [r["k"] for r in sweep["rows"]]
    its = [r["iterations"] for r in sweep["rows"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ks, y=its, mode="lines+markers", line=dict(color=TEAL, width=3), name="Iterationen (Median)"))
    fig.add_trace(go.Scatter(x=ks, y=[its[0] + sweep["theory_slope"] * (k - ks[0]) for k in ks], mode="lines", line=dict(color=GREY, dash="dash"), name="Theorie-Steigung 2n(n+1) ln 10 je Zehnerpotenz"))
    fig.update_xaxes(title_text="Genauigkeit 10^-k", dtick=1)
    fig.update_yaxes(title_text="Iterationen")
    return _base(fig, 320, legend_y=-0.4)


def build_radius(rows):
    fs = [str(r["factor"]) for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=fs, y=[r["iterations"] for r in rows], marker_color=TEAL, name="Iterationen bis zur Genauigkeit"))
    fig.add_trace(go.Bar(x=fs, y=[r["first_feasible"] for r in rows], marker_color=ORANGE, name="bis zum ersten zulässigen Mittelpunkt"))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text="Faktor der Startkugel", type="category")
    fig.update_yaxes(title_text="Iterationen")
    return _base(fig, 300, legend_y=-0.4)


def build_bound(rows):
    ns = [str(r["n"]) for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=ns, y=[r["bound"] for r in rows], marker_color="rgba(138,143,152,0.6)", name="Theorie-Schranke 2n(n+1) ln(r0 / r)"))
    fig.add_trace(go.Bar(x=ns, y=[r["first_feasible"] for r in rows], marker_color=TEAL, name="gemessen: erster zulässiger Mittelpunkt"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(type="log", title_text="Iterationen")
    fig.update_xaxes(title_text="Größe n (Zufall, m = n)", type="category")
    return _base(fig, 300, legend_y=-0.4)


def build_cuts(rows):
    labels = [f"{'tief' if r['cut'] == 'deep' else 'zentral'}, {'erste' if r['select'] == 'first' else 'stärkste'} Zeile" for r in rows]
    fig = go.Figure(go.Bar(x=labels, y=[r["iterations"] for r in rows], marker_color=[GREY, GREY, TEAL, TEAL], text=[f"{r['iterations']:.0f}" for r in rows], textposition="outside"))
    fig.update_yaxes(title_text="Iterationen (Median)", rangemode="tozero")
    return _base(fig, 300)


def build_scale(rows):
    ks = [r["k"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ks, y=[r["iterations"] for r in rows], mode="lines+markers", line=dict(color=TEAL, width=3), name="Iterationen (Ellipsoid)"))
    fig.add_trace(go.Scatter(x=ks, y=[max(min(r["ell_max"], 10.0), 1e-17) for r in rows], mode="lines+markers", yaxis="y2", line=dict(color=ORANGE, width=2), name="größter relativer Fehler Ellipsoid"))
    fig.add_trace(go.Scatter(x=ks, y=[max(min(r["sim_max"], 10.0), 1e-17) for r in rows], mode="lines+markers", yaxis="y2", line=dict(color=BLUE, width=2, dash="dot"), name="größter relativer Fehler Simplex"))
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", type="log", title="relativer Fehler des Optimalwerts", showgrid=False, fixedrange=True, tickformat=".0e"), yaxis_title_text="Iterationen")
    fig.update_xaxes(title_text="Spalten über 10^k gestreut", dtick=2)
    return _base(fig, 340, legend_y=-0.4)


def build_equality(sweep):
    rows = sweep["rows"]
    ks = [r["k"] for r in rows if r["first_feasible"] > 0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ks, y=[r["first_feasible"] for r in rows if r["first_feasible"] > 0], mode="lines+markers", line=dict(color=TEAL, width=3), name="erster zulässiger Mittelpunkt"))
    broken = [r for r in rows if r["first_feasible"] < 0]
    if broken:
        fig.add_trace(go.Scatter(x=[r["k"] for r in broken], y=[r["iterations"] for r in broken], mode="markers", marker=dict(size=12, color=RED, symbol="x"), name="numerisch gescheitert (Iteration des Abbruchs)"))
    fig.update_xaxes(title_text="Aufweitung der Gleichungen 10^-k", dtick=2)
    fig.update_yaxes(title_text="Iterationen")
    return _base(fig, 300, legend_y=-0.4)


def build_cube(sweep):
    """Klee-Minty-Würfel: Operationen von Ellipsoid und Simplex (Dantzig-Regel: 2^n - 1 Pivots) über n, log-y; senkrechte Linie am Kreuzungspunkt."""
    rows = sweep["rows"]
    ns = [r["n"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ns, y=[r["flops_ell"] for r in rows], mode="lines+markers", line=dict(color=ORANGE, width=3), name="Ellipsoid"))
    fig.add_trace(go.Scatter(x=ns, y=[r["flops_simplex"] for r in rows], mode="lines+markers", line=dict(color=TEAL, width=3), name="Simplex (Dantzig-Regel)"))
    if sweep["crossover"]:
        fig.add_vline(x=sweep["crossover"] - 0.5, line=dict(color=GREY, dash="dot"))
    fig.update_yaxes(type="log", title_text="Operationen (Modell)")
    fig.update_xaxes(title_text="Größe n des Würfels", dtick=2)
    return _base(fig, 320, legend_y=-0.4)
