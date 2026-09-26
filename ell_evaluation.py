"""Auswertung der Ellipsoid-Methode: ein Lauf gegen den Simplex, Iterationen über n / Genauigkeit / Startkugel, Schnittarten, Theorie-Schranke, Skalierung, Gleichungen."""

import math
import statistics
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import ell_algorithm as A
import ell_constants as C
import ell_ellipsoid as E
import ell_scenario as S


@dataclass(frozen=True)
class Settings:
    kind: str = "centre"
    m: int = C.DEFAULT_M
    n: int = C.DEFAULT_N
    seed: int = C.DEFAULT_SEED
    cut: str = "central"
    select: str = "first"
    eps_i: int = C.DEFAULT_EPS_I
    radius_i: int = C.DEFAULT_RADIUS_I
    scale_i: int = C.DEFAULT_SCALE_I

    @property
    def eps(self):
        return 10.0 ** -C.EPS_EXPS[min(max(self.eps_i, 0), len(C.EPS_EXPS) - 1)]

    @property
    def radius_factor(self):
        return C.RADIUS_FACTORS[min(max(self.radius_i, 0), len(C.RADIUS_FACTORS) - 1)]

    @property
    def scale_exp(self):
        return C.SCALE_EXPS[min(max(self.scale_i, 0), len(C.SCALE_EXPS) - 1)]


def base_instance(s):
    return S.generate(s.kind, s.m, s.n, C.DENSITY, s.seed)


def instance_of(s):
    return S.column_scaled(base_instance(s), s.scale_exp)


def simplex_flops(inst, solution):
    """Operationsmodell des dichten Tableau-Simplex: je Pivot 2 (m+1)(ncols+1) (Zeilenoperationen des ganzen Tableaus)."""
    ncols = A.standard_form(inst)[2]["ncols"]
    return solution.pivots * 2 * (inst.m + 1) * (ncols + 1)


@dataclass
class Analysis:
    inst: object
    base: object
    res: object                      # EllipsoidResult
    simplex: object                  # Solution auf der (ggf. skalierten) Instanz
    ref_status: str
    ref_obj: float                   # Optimalwert des Simplex auf der unskalierten Instanz
    obj_error: float                 # Ellipsoid - Referenz (nan ohne Ergebnis)
    simplex_error: float
    flops_ell: int
    flops_simplex: int
    cut: str = "central"


@lru_cache(maxsize=64)
def analyse(s):
    inst, base = instance_of(s), base_instance(s)
    res = E.ellipsoid(inst, eps=s.eps, cut=s.cut, select=s.select, radius_factor=s.radius_factor, eq_tol=C.EQ_TOL_APP, max_iter=C.MAX_ITER, keep=True, keep_eig=True, keep_mats=(inst.n == 2))
    sol = A.solve(inst)
    ref = A.solve(base)
    ref_obj = ref.obj if ref.status == "optimal" else float("nan")
    return Analysis(inst, base, res, sol, ref.status, ref_obj, (res.obj - ref_obj) if res.x and ref.status == "optimal" else float("nan"),
                    (sol.obj - ref_obj) if sol.status == "optimal" and ref.status == "optimal" else float("nan"), res.flops, simplex_flops(inst, sol) if sol.status == "optimal" else 0, s.cut)


def _random(n, seed, m=None):
    return S.generate("random", n if m is None else m, n, C.DENSITY, seed)


def _med(values):
    return statistics.median(values) if values else float("nan")


@lru_cache(maxsize=32)
def size_sweep(s):
    """Zufallsinstanzen m = n über die Größen: Iterationen, Iterationen / n^2, Operationen gegen den Simplex (Median über fünf feste Instanzen)."""
    rows = []
    for n in C.SWEEP_SIZES:
        its, fe, pv, fs = [], [], [], []
        ok = 0
        for sd in C.SWEEP_SEEDS:
            inst = _random(n, sd)
            r = E.ellipsoid(inst, eps=s.eps, cut=s.cut, select=s.select, radius_factor=s.radius_factor, max_iter=400_000)
            sol = A.solve(inst)
            ok += r.status == "optimal"
            its.append(r.iterations), fe.append(r.flops), pv.append(sol.pivots), fs.append(simplex_flops(inst, sol))
        rows.append({"n": n, "iterations": _med(its), "min": min(its), "max": max(its), "per_n2": _med(its) / (n * n), "flops_ell": _med(fe), "pivots": _med(pv), "flops_simplex": _med(fs),
                     "ratio": _med(fe) / max(_med(fs), 1.0), "optimal": ok})
    return rows


@lru_cache(maxsize=32)
def eps_sweep(s):
    """Iterationen über die Genauigkeit (Zufall, m = n aus dem Regler): gemessene Iterationen je Zehnerpotenz gegen die Theorie 2 n (n+1) ln 10."""
    n = s.n
    rows = []
    for k in range(2, 11):
        its, ok = [], 0
        for sd in C.SWEEP_SEEDS:
            r = E.ellipsoid(_random(n, sd), eps=10.0 ** -k, cut=s.cut, select=s.select, radius_factor=s.radius_factor, max_iter=400_000)
            its.append(r.iterations)
            ok += r.status == "optimal"
        rows.append({"k": k, "iterations": _med(its), "optimal": ok})
    slope = (rows[-1]["iterations"] - rows[0]["iterations"]) / (rows[-1]["k"] - rows[0]["k"])
    theory = 2.0 * n * (n + 1) * math.log(10.0)
    return {"rows": rows, "slope": slope, "theory_slope": theory, "share": slope / theory}


@lru_cache(maxsize=32)
def radius_sweep(s):
    """Iterationen über den Faktor der Startkugel (Zufall, m = n): nur logarithmisch, ln(R) je Verdopplung ~ 2 n (n+1) ln 2 im schlimmsten Fall."""
    rows = []
    for f in C.RADIUS_FACTORS:
        its, first = [], []
        for sd in C.SWEEP_SEEDS:
            r = E.ellipsoid(_random(s.n, sd), eps=s.eps, cut=s.cut, select=s.select, radius_factor=f, max_iter=400_000)
            its.append(r.iterations), first.append(r.first_feasible)
        rows.append({"factor": f, "iterations": _med(its), "first_feasible": _med(first)})
    return rows


@lru_cache(maxsize=32)
def cut_compare(s):
    """Vier Kombinationen aus Schnittart und Zeilenwahl auf fünf Zufallsinstanzen m = n (Größe aus dem Regler)."""
    rows = []
    for cut in E.CUTS:
        for select in E.SELECTS:
            its = [E.ellipsoid(_random(s.n, sd), eps=s.eps, cut=cut, select=select, radius_factor=s.radius_factor, max_iter=400_000).iterations for sd in C.SWEEP_SEEDS]
            rows.append({"cut": cut, "select": select, "iterations": _med(its), "min": min(its), "max": max(its)})
    base = rows[0]["iterations"]
    for r in rows:
        r["saving"] = 1.0 - r["iterations"] / base
    return rows


def inscribed_radius(inst):
    """Radius der größten Kugel im zulässigen Bereich (Tschebyschow-Radius) für <=-Instanzen: max r mit a_i x + |a_i| r <= b_i, x_j >= r; gelöst mit dem Simplex dieser Demo."""
    A_, b, _c = inst.arrays()
    n = inst.n
    rows = [list(A_[i]) + [float(np.linalg.norm(A_[i]))] for i in range(inst.m)] + [[-1.0 if j == k else 0.0 for j in range(n)] + [1.0] for k in range(n)]
    bb = list(b) + [0.0] * n
    custom = S.Instance(tuple(tuple(float(v) for v in r) for r in rows), tuple(float(v) for v in bb), tuple([0.0] * n + [1.0]), (S.LE,) * len(bb), tuple(f"v{j}" for j in range(n + 1)),
                        tuple(f"r{i}" for i in range(len(bb))), "custom")
    sol = A.solve(custom)
    return float(sol.obj) if sol.status == "optimal" else float("nan")


@lru_cache(maxsize=32)
def feasibility_bound(s):
    """Iteration des ersten zulässigen Mittelpunkts gegen die Theorie-Schranke 2 n (n+1) ln(r0 / r_in) (r_in: größte Kugel im zulässigen Bereich)."""
    rows = []
    for n in C.BOUND_SIZES:
        first, bound, share = [], [], []
        for sd in C.SWEEP_SEEDS:
            inst = _random(n, sd)
            r = E.ellipsoid(inst, eps=s.eps, cut=s.cut, select=s.select, radius_factor=s.radius_factor, max_iter=400_000)
            b = E.theory_bound(n, r.r0, inscribed_radius(inst))
            first.append(r.first_feasible), bound.append(b), share.append(r.first_feasible / b)
        rows.append({"n": n, "first_feasible": _med(first), "bound": _med(bound), "share": _med(share), "max_share": max(share)})
    return rows


@lru_cache(maxsize=32)
def scale_sweep(s):
    """Schlechte Skalierung: dieselben fünf Zufallsinstanzen (m, n aus dem Regler) mit Spalten über 10^k gestreut; Iterationen und Fehler des Optimalwerts gegen den unskalierten Wert, Ellipsoid gegen Simplex."""
    m, n = (s.m, s.n) if s.kind in ("random", "mixed") else (6, 6)
    rows = []
    for k in C.SCALE_EXPS:
        its, ell_err, sim_err, broke = [], [], [], 0
        for sd in C.SWEEP_SEEDS:
            base = S.generate("random", m, n, C.DENSITY, sd)
            true = A.solve(base).obj
            inst = S.column_scaled(base, k)
            r = E.ellipsoid(inst, eps=1e-6, cut="central", max_iter=400_000)
            sol = A.solve(inst)
            its.append(r.iterations)
            broke += r.status == "numerical"
            ell_err.append(abs(r.obj - true) / (1.0 + abs(true)) if r.x else float("inf"))
            if sol.status == "optimal":
                sim_err.append(abs(sol.obj - true) / (1.0 + abs(true)))
            else:
                sim_err.append(float("inf"))
        rows.append({"k": k, "iterations": _med(its), "ell_broke": broke, "ell_wrong": sum(1 for e in ell_err if e > 1e-3), "sim_wrong": sum(1 for e in sim_err if e > 1e-3), "ell_max": max(ell_err), "sim_max": max(sim_err)})
    return rows


@lru_cache(maxsize=32)
def equality_sweep(s):
    """Gleichungen als zwei Ungleichungen mit Aufweitung 10^-k (Mischinstanz): Iteration des ersten zulässigen Mittelpunkts, das Innere der zulässigen Menge wird dünn."""
    m, n = (s.m, s.n) if s.kind == "mixed" else (8, 8)
    seed = s.seed if s.kind == "mixed" else C.DEFAULT_SEED
    inst = S.generate("mixed", m, n, C.DENSITY, seed)
    rows = []
    for k in C.EQ_EXPS:
        r = E.ellipsoid(inst, eps=1e-6, eq_tol=10.0 ** -k, cut="deep", select="most", max_iter=C.MAX_ITER * 2)
        rows.append({"k": k, "first_feasible": r.first_feasible, "iterations": r.iterations, "status": r.status})
    return {"rows": rows, "eq_rows": sum(1 for x in inst.senses if x == S.EQ), "m": m, "n": n, "seed": seed}


@lru_cache(maxsize=32)
def cube_sweep(s):
    """Klee-Minty-Würfel n = 2..14: Iterationen des Ellipsoids gegen die 2^n - 1 Pivots des Simplex (gemessen mit dem Löser dieser Demo), Operationen im Modell und Kreuzungspunkt (kleinstes n, ab dem das Ellipsoid weniger Operationen braucht)."""
    rows = []
    for n in C.CUBE_SIZES:
        inst = S.klee_minty_instance(n)
        r = E.ellipsoid(inst, eps=s.eps, cut=s.cut, select=s.select, radius_factor=s.radius_factor, max_iter=2_000_000)
        sol = A.solve(inst)
        fs = simplex_flops(inst, sol)
        rows.append({"n": n, "iterations": r.iterations, "pivots": sol.pivots, "flops_ell": r.flops, "flops_simplex": fs, "ratio": r.flops / max(fs, 1), "status": r.status, "optimal": 5.0 ** n})
    wins = [r["n"] for r in rows if r["flops_ell"] < r["flops_simplex"]]
    cross = None
    for r in rows:
        if all(q["flops_ell"] < q["flops_simplex"] for q in rows if q["n"] >= r["n"]):
            cross = r["n"]
            break
    return {"rows": rows, "crossover": cross, "wins": wins}
