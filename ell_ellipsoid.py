"""Ellipsoid-Methode (Schor, Judin/Nemirowski, Khachiyan) für  max c·x,  A x (<=|>=|=) b,  x >= 0.

Ein Ellipsoid E = {z : (z - x)^T P^-1 (z - x) <= 1} enthält alle zulässigen Punkte mit c·z >= Bestwert. Je Iteration liegt der Mittelpunkt x entweder außerhalb einer Nebenbedingung (Machbarkeitsschnitt) oder ist
zulässig (Objektivschnitt: c·z >= Bestwert); der Schnitt halbiert (zentral) bzw. verkleinert (tief) E, das kleinste umschließende Ellipsoid der Restmenge ersetzt E. Das Volumen fällt je zentralem Schnitt mindestens
um den Faktor exp(-1/(2(n+1))). Gleichungen werden als zwei Ungleichungen mit Aufweitung eq_tol behandelt (sonst hat die zulässige Menge kein Inneres und die Methode verliert ihr Volumenargument).

Zertifikate: obere Schranke ub = c·x + sqrt(c^T P c) (jedes optimale z liegt in E), Unzulässigkeit, wenn ein Schnitt mit alpha >= 1 die ganze Ellipse abschneidet."""

import math
from dataclasses import dataclass, field

import numpy as np

import ell_scenario as S

LE, GE, EQ = S.LE, S.GE, S.EQ
BIG_FACTOR = 100.0                          # Schranke für Dienste ohne endliche Obergrenze: BIG_FACTOR mal die größte endliche
DEFAULT_BIG = 1000.0
FEAS_TOL = 1e-12                            # relative Toleranz "Mittelpunkt erfüllt die Zeile"
EQ_TOL = 1e-3                               # Aufweitung der Gleichungen (relativ zu max(1, |b|))
CUTS = ("central", "deep")
SELECTS = ("first", "most")


def flops_per_iteration(m_rows, n):
    """Operationsmodell je Iteration (Multiply-Add = 2): Zeilenwerte G x (2 m' n), P a (2 n^2), a^T P a (2 n), Mittelpunkt (2 n), P-Update mit Symmetrisierung (4 n^2)."""
    return 2 * m_rows * n + 6 * n * n + 4 * n


def theory_bound(n, r0, r):
    """Iterationen, nach denen das Volumen unter das einer Kugel vom Radius r fällt: k = 2 n (n+1) ln(r0 / r). Enthält die zulässige Menge eine Kugel vom Radius r, ist bis dahin ein zulässiger Mittelpunkt gefunden."""
    return 2.0 * n * (n + 1) * math.log(max(r0 / r, 1.0))


def volume_log_ratio_central(n):
    """ln(Vol(E+) / Vol(E)) eines zentralen Schnitts, geschlossene Form: ((n/(n+1))^(n+1) (n/(n-1))^(n-1))^(1/2) (n >= 2); n = 1: ln(1/2)."""
    if n == 1:
        return math.log(0.5)
    return 0.5 * ((n + 1) * math.log(n / (n + 1.0)) + (n - 1) * math.log(n / (n - 1.0)))


def rows_of(inst, eq_tol=EQ_TOL):
    """Alle Bedingungen als G z <= h: <= unverändert, >= gespiegelt, = als zwei Zeilen mit Aufweitung; danach -z_j <= 0. Rückgabe: G, h, Zahl der Strukturzeilen, Herkunft je Zeile (Instanzzeile oder -1 - j für z_j >= 0)."""
    A, b, _c = inst.arrays()
    rows, rhs, origin = [], [], []
    for i, s in enumerate(inst.senses):
        if s == LE:
            rows.append(A[i]), rhs.append(b[i]), origin.append(i)
        elif s == GE:
            rows.append(-A[i]), rhs.append(-b[i]), origin.append(i)
        else:
            d = eq_tol * max(1.0, abs(b[i]))
            rows.append(A[i]), rhs.append(b[i] + d), origin.append(i)
            rows.append(-A[i]), rhs.append(-b[i] + d), origin.append(i)
    n_struct = len(rows)
    for j in range(inst.n):
        e = np.zeros(inst.n)
        e[j] = -1.0
        rows.append(e), rhs.append(0.0), origin.append(-1 - j)
    return np.array(rows), np.array(rhs), n_struct, origin


def start_ball(inst, radius_factor=1.0, eq_tol=EQ_TOL):
    """Startkugel: Kasten [0, U] aus den Zeilen mit positivem Koeffizienten (a_ij > 0 in <=- und =-Zeilen: z_j <= b_i / a_ij, nur wenn alle anderen Koeffizienten der Zeile >= 0 sind, sonst gibt es keine Schranke);
    Dienste ohne Schranke bekommen BIG_FACTOR mal die größte endliche. Kugel um die Mitte des Kastens, Radius radius_factor mal der halben Diagonale. Rückgabe: Mittelpunkt, Radius, U, Menge der Dienste ohne Schranke."""
    A, b, _c = inst.arrays()
    n = inst.n
    U = np.full(n, np.inf)
    for i, s in enumerate(inst.senses):
        if s == GE or np.any(A[i] < 0):
            continue
        cap = b[i] + (eq_tol * max(1.0, abs(b[i])) if s == EQ else 0.0)
        for j in range(n):
            if A[i, j] > 0:
                U[j] = min(U[j], cap / A[i, j])
    free = {j for j in range(n) if not math.isfinite(U[j])}
    finite = [U[j] for j in range(n) if j not in free]
    big = BIG_FACTOR * max(finite) if finite else DEFAULT_BIG
    for j in free:
        U[j] = big
    centre = U / 2.0
    radius = radius_factor * 0.5 * float(np.linalg.norm(U))
    return centre, radius, U, free


@dataclass
class EllipsoidResult:
    status: str                                       # "optimal" | "infeasible" | "ball" | "numerical" | "limit"
    x: tuple = ()                                     # bester zulässiger Punkt (leer, wenn keiner gefunden)
    obj: float = float("nan")
    upper: float = float("nan")                       # letzte obere Schranke c·x + sqrt(c^T P c)
    iterations: int = 0
    first_feasible: int = -1                          # Iteration des ersten zulässigen Mittelpunkts (-1: nie)
    feasibility_cuts: int = 0
    objective_cuts: int = 0
    best_path: list = field(default_factory=list)     # Bestwert nach jeder Iteration (nan bis zum ersten zulässigen Punkt)
    upper_path: list = field(default_factory=list)    # obere Schranke je Iteration
    log_volume: list = field(default_factory=list)    # ln Vol(E_k) - ln Vol(E_0) (Start: 0)
    min_eig: list = field(default_factory=list)       # kleinster Eigenwert von P je Iteration (nur mit keep_eig)
    centres: list = field(default_factory=list)       # Mittelpunkte (nur mit keep)
    mats: list = field(default_factory=list)          # P je Iteration (nur mit keep und keep_mats)
    cuts: list = field(default_factory=list)          # (Art, Zeilennummer oder -1 für den Objektivschnitt, alpha) je Iteration (nur mit keep)
    flops: int = 0
    r0: float = 0.0
    note: str = ""

    @property
    def gap(self):
        return self.upper - self.obj if self.x else float("nan")


def _cut_update(x, P, a, beta, cut, n):
    """Ein Schnitt a·z <= beta durch (zentral: a·x) bzw. (tief: beta). Rückgabe: neuer Mittelpunkt, neues P, alpha, ln(det P+ / det P); bei alpha >= 1 (Ellipse ganz abgeschnitten) (None, None, alpha, 0).
    Verletzte Zeilen und Objektivschnitte haben immer alpha >= 0, der Bereich -1/n < alpha < 0 kommt hier nicht vor."""
    Pa = P @ a
    s2 = float(a @ Pa)
    if not (s2 > 0.0 and math.isfinite(s2)):
        raise ArithmeticError("a^T P a nicht positiv")
    s = math.sqrt(s2)
    alpha = (float(a @ x) - beta) / s
    if alpha >= 1.0:                                                                            # die Zeile schneidet die ganze Ellipse ab (auch bei zentralem Schnitt ein Beweis)
        return None, None, alpha, 0.0
    alpha_used = 0.0 if cut == "central" else alpha
    if n == 1:
        r = math.sqrt(float(P[0, 0]))
        lo, hi = float(x[0]) - r, float(x[0]) + r
        limit = beta / a[0] if cut == "deep" else float(x[0])
        if a[0] > 0:
            hi = min(hi, limit)
        else:
            lo = max(lo, limit)
        if not hi > lo:
            return None, None, max(alpha, 1.0), 0.0
        half = 0.5 * (hi - lo)
        return np.array([0.5 * (hi + lo)]), np.array([[half * half]]), alpha, 2.0 * math.log(half / r)
    tau = (1.0 + n * alpha_used) / (n + 1.0)
    sigma = 2.0 * (1.0 + n * alpha_used) / ((n + 1.0) * (1.0 + alpha_used))
    delta = n * n * (1.0 - alpha_used * alpha_used) / (n * n - 1.0)
    x_new = x - tau * (Pa / s)
    P_new = delta * (P - sigma * np.outer(Pa, Pa) / s2)
    P_new = 0.5 * (P_new + P_new.T)
    logdet = n * math.log(delta) + math.log1p(-sigma)
    return x_new, P_new, alpha, logdet


def ellipsoid(inst, eps=1e-6, cut="central", select="first", radius_factor=1.0, eq_tol=EQ_TOL, max_iter=200_000, keep=False, keep_eig=False, keep_mats=True):
    """Ellipsoid-Methode mit Objektivschnitt. eps: relative Genauigkeit der Zertifikatslücke ub - best <= eps * (1 + |best|)."""
    if cut not in CUTS or select not in SELECTS:
        raise ValueError((cut, select))
    n = inst.n
    _A, _b, c = inst.arrays()
    G, h, n_struct, _origin = rows_of(inst, eq_tol)
    m_rows = len(h)
    gnorm = np.linalg.norm(G, axis=1)
    centre, radius, U, free = start_ball(inst, radius_factor, eq_tol)
    x = centre.copy()
    P = np.eye(n) * radius * radius
    res = EllipsoidResult(status="limit", r0=radius)
    best, best_x = -math.inf, None
    logvol = 0.0
    it_flops = flops_per_iteration(m_rows, n)
    res.log_volume.append(0.0)
    obj_row = -c
    for it in range(1, max_iter + 1):
        v = G @ x - h
        scale = FEAS_TOL * (1.0 + np.abs(h))
        viol = np.nonzero(v > scale)[0]
        cq = float(c @ x)
        try:
            width = math.sqrt(max(float(c @ P @ c), 0.0))
            if len(viol):
                k = int(viol[0]) if select == "first" else int(viol[np.argmax(v[viol] / np.maximum(gnorm[viol], 1e-300))])
                a, beta, kind, row = G[k], float(h[k]), "feasibility", k
                res.feasibility_cuts += 1
            else:
                if res.first_feasible < 0:
                    res.first_feasible = it
                if cq > best:
                    best, best_x = cq, x.copy()
                if not c.any():                                          # keine Zielfunktion: jeder zulässige Punkt ist optimal
                    res.status, res.note = "optimal", "Machbarkeitsproblem: zulässiger Mittelpunkt gefunden"
                    res.iterations = it
                    res.best_path.append(best), res.upper_path.append(best)
                    res.flops += it_flops
                    break
                a, beta, kind, row = obj_row, -best, "objective", -1
                res.objective_cuts += 1
            x_new, P_new, alpha, dlog = _cut_update(x, P, a, beta, cut, n)
        except ArithmeticError:
            res.status, res.note = "numerical", "P ist nicht mehr positiv definit"
            break
        upper = cq + width
        res.best_path.append(best if best_x is not None else float("nan"))
        res.upper_path.append(upper if best_x is not None else float("nan"))
        if keep:
            res.centres.append(x.copy())
            if keep_mats:
                res.mats.append(P.copy())
            res.cuts.append((kind, row, alpha))
        if keep_eig:
            res.min_eig.append(float(np.linalg.eigvalsh(P)[0]))
        res.flops += it_flops
        res.iterations = it
        if best_x is not None and upper - best <= eps * (1.0 + abs(best)):
            res.status = "optimal"
            break
        if x_new is None:                                             # alpha >= 1: die ganze Ellipse liegt außerhalb der Zeile
            if kind == "feasibility" and best_x is None:
                res.status, res.note = "infeasible", f"Zeile {row} schneidet die ganze Ellipse ab (alpha = {alpha:.3g})"
            elif kind == "feasibility":                               # Es gibt einen zulässigen Punkt, also muss das Optimum in der Ellipse liegen: nur Rundungsfehler können das verletzen
                res.status, res.note = "numerical", f"Rundung: Zeile {row} schneidet die ganze Ellipse ab, obwohl ein zulässiger Punkt bekannt ist (alpha = {alpha:.3g})"
            else:
                res.status, res.note = "optimal", "Objektivschnitt trifft nur noch den Mittelpunkt"
            break
        x, P = x_new, P_new
        logvol += 0.5 * dlog                                          # Volumen ~ sqrt(det P)
        res.log_volume.append(logvol)
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(P))):
            res.status, res.note = "numerical", "nicht endliche Werte"
            break
    else:
        res.status = "limit"
    if best_x is not None:
        res.x, res.obj = tuple(float(t) for t in best_x), float(best)
        res.upper = res.upper_path[-1] if res.upper_path else float("nan")
        if res.status in ("optimal", "limit") and free and any(best_x[j] >= 0.9 * U[j] for j in free):
            res.status, res.note = "ball", "Optimum am Rand der Startkugel: die Instanz ist vermutlich unbeschränkt"
    elif res.status == "limit":
        res.note = "kein zulässiger Punkt gefunden"
    return res
