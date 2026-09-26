"""Konstanten der Demo Ellipsoid-Methode: Regler-Bereiche, Stufen, feste Instanzen für die Auswertung, Presets."""
M_MIN, M_MAX, DEFAULT_M = 2, 14, 6
N_MIN, N_MAX, DEFAULT_N = 2, 14, 8
DENSITY = 0.5                                        # Dichte der Zufalls- und Mischinstanzen (fest)
SEED_MAX = 999999
DEFAULT_SEED = 35
EPS_EXPS = (2, 3, 4, 6, 8, 10, 12, 14)                     # Genauigkeit 10^-k (relative Zertifikatslücke)
DEFAULT_EPS_I = 3                                    # 1e-6
RADIUS_FACTORS = (1, 2, 4, 16, 64)                   # Startkugel: Faktor auf die halbe Diagonale des Kastens
DEFAULT_RADIUS_I = 0
SCALE_EXPS = (0, 2, 4, 6, 8, 10, 12)                 # Spaltenskalierung 10^k (Einträge über k Zehnerpotenzen gestreut)
DEFAULT_SCALE_I = 0
CUT_LABELS = {"central": "Zentraler Schnitt (durch den Mittelpunkt)", "deep": "Tiefer Schnitt (bis an die verletzte Nebenbedingung)"}
SELECT_LABELS = {"first": "Erste verletzte Zeile", "most": "Am stärksten verletzte Zeile"}
STEPS = {1: "1 · Die Ellipsen", 2: "2 · Iterationen", 3: "3 · Gegen den Simplex", 4: "4 · Grenzen"}
MAX_ITER = 100_000
EQ_TOL_APP = 1e-6                                    # Aufweitung der Gleichungen in der App
SWEEP_SEEDS = tuple(range(100000, 100005))
SWEEP_SIZES = (2, 4, 6, 8, 10, 12, 16, 20)           # Zufallsinstanzen m = n für die Größenkurve
BOUND_SIZES = (2, 3, 4, 6, 8, 10)                    # Theorie-Schranke: Inkugelradius über den Simplex berechnet
CUBE_SIZES = tuple(range(2, 15))                     # Klee-Minty-Würfel (n bis 14: alle Zahlen exakt)
EQ_EXPS = (2, 4, 6, 8, 10)                           # Aufweitung der Gleichungen 10^-k
_BASE = {"kind": "centre", "m": 6, "n": 8, "seed": 35, "cut": "central", "sel": "first", "eps": 3, "rad": 0, "scale": 0, "step": 1}
PRESETS = {
    "Lehrbuch: die ersten Ellipsen": {**_BASE, "kind": "textbook", "step": 1, "iter_k": 4},
    "Zentrum: Ellipsoid gegen Simplex": {**_BASE, "step": 3},
    "Iterationen wachsen mit n²": {**_BASE, "kind": "random", "m": 8, "n": 8, "step": 2},
    "Genauigkeit: Iterationen je Zehnerpotenz": {**_BASE, "kind": "random", "m": 8, "n": 8, "step": 2},
    "Tiefe Schnitte": {**_BASE, "kind": "random", "m": 8, "n": 8, "cut": "deep", "step": 4},
    "Die Theorie-Schranke ist weit weg": {**_BASE, "kind": "random", "m": 8, "n": 8, "step": 2},
    "Genauigkeitsgrenze: 10^-14 geht nicht": {**_BASE, "kind": "random", "m": 8, "n": 8, "eps": 7, "step": 1},
    "Schlechte Skalierung": {**_BASE, "kind": "random", "m": 6, "n": 8, "scale": 6, "step": 4},
    "Gleichungen: dünnes Inneres": {**_BASE, "kind": "mixed", "m": 8, "n": 8, "cut": "deep", "sel": "most", "step": 4},
    "Klee-Minty-Würfel: hier gewinnt das Ellipsoid": {**_BASE, "kind": "klee_minty", "n": 14, "cut": "deep", "sel": "most", "step": 3},
    "Unzulässig: Ellipse ganz abgeschnitten": {**_BASE, "kind": "infeasible", "cut": "deep", "sel": "most", "step": 1},
    "Unbeschränkt: Kugelrand": {**_BASE, "kind": "unbounded", "cut": "deep", "sel": "most", "step": 1},
}
PRESET_HELP = {
    "Lehrbuch: die ersten Ellipsen": "Start ist die Kugel um den Kasten [0, 4] × [0, 6] mit dem Mittelpunkt (2, 3). Iteration 1: der Mittelpunkt ist zulässig (Zielwert 21), Objektivschnitt; Iteration 3: die Lagerfläche wird verletzt, Machbarkeitsschnitt. Nach 100 Iterationen ist das Optimum 36 bis ε = 10^-6 bewiesen (mit tiefen Schnitten 78); der Simplex braucht 2 Pivots.",
    "Zentrum: Ellipsoid gegen Simplex": "594 Iterationen (484 Machbarkeits-, 110 Objektivschnitte) beweisen das Optimum 720; der Simplex braucht 4 Pivots. Im Operationsmodell sind das 154.440 gegen 400, das 386-Fache.",
    "Iterationen wachsen mit n²": "Zufall 8 × 8 (Seed 35): 1625 Iterationen. Über die Größen n = 2 bis 20 liegt das Verhältnis Iterationen / n² bei 24 bis 25.5 (Median über fünf Instanzen: 102 bei n = 2, 916 bei n = 6, 2380 bei n = 10, 9748 bei n = 20). Klick auf 'Iterationen über die Größe n berechnen'.",
    "Genauigkeit: Iterationen je Zehnerpotenz": "Bei n = 8 kostet jede weitere Stelle Genauigkeit rund 291 Iterationen (448 bei 10^-2, 2777 bei 10^-10). Die Theorie 2n(n+1) ln 10 = 332 erwartet mehr: gemessen sind es 88 % davon. Klick auf 'Iterationen über die Genauigkeit berechnen'.",
    "Tiefe Schnitte": "Zufall 8 × 8 (Seed 35): 1405 Iterationen mit tiefen Schnitten gegen 1625 mit zentralen (−14 %). Über fünf Instanzen n = 8: 1400 gegen 1611 (−13 %); die Wahl der Zeile ändert bei zentralen Schnitten kaum etwas (1627 gegen 1611). Klick auf 'Schnittarten vergleichen'.",
    "Die Theorie-Schranke ist weit weg": "Bis zum ersten zulässigen Mittelpunkt braucht die Methode bei n = 8 im Median 10 Iterationen, die Schranke 2n(n+1) ln(r0 / r) erlaubt 360 (2.8 %); bei n = 10 sind es 10 gegen 624 (1.6 %). Klick auf 'Theorie-Schranke prüfen'.",
    "Genauigkeitsgrenze: 10^-14 geht nicht": "Zufall 8 × 8 (Seed 35), ε = 10^-14: nach 3491 Iterationen bricht die Methode ab (eine Nebenbedingung schneidet die ganze Ellipse ab, obwohl ein zulässiger Punkt bekannt ist). Der beste Wert stimmt dann schon bis auf 2 · 10^-10 mit dem Simplex überein, aber die Lücke lässt sich nicht mehr beweisen. Über je fünf Zufallsinstanzen mit n = 4 bis 40 gelingt das Zertifikat bis ε = 10^-12 immer und scheitert ab 10^-13 immer - unabhängig von n.",
    "Schlechte Skalierung": "Zufall 6 × 8 (Seed 35), Spalten über 10^12 gestreut: nach 1978 Iterationen bricht die Methode numerisch ab (eine Nebenbedingung schneidet die ganze Ellipse ab, obwohl ein zulässiger Punkt bekannt ist); der beste Wert bis dahin, 249.94, liegt 1.3 % über dem wahren 246.79. Über fünf Instanzen bricht sie bei 10^8 in 1, bei 10^10 in 4 und bei 10^12 in 5 von 5 ab. Der Simplex bleibt bis 10^10 exakt und liefert bei 10^12 in 3 von 5 Fällen stillschweigend einen falschen Wert. Klick auf 'Schlechte Skalierung testen'.",
    "Gleichungen: dünnes Inneres": "Mischung 8 × 8 (Seed 35, eine Gleichung): der erste zulässige Mittelpunkt kommt bei der Aufweitung 10^-2 in Iteration 28, bei 10^-4 in 46, bei 10^-6 in 64 und bei 10^-8 in 83 (ln(1/δ)); bei 10^-10 scheitert die Numerik nach etwa 80 bis 100 Iterationen (Windows 84, Linux 97). Klick auf 'Gleichungen: Aufweitung verkleinern'.",
    "Klee-Minty-Würfel: hier gewinnt das Ellipsoid": "Würfel mit n = 14 (tiefe Schnitte): der Simplex mit Dantzig-Regel besucht alle 2^14 Ecken, 16383 Pivots; das Ellipsoid braucht 4052 Iterationen. Im Operationsmodell sind das 8.2 Mio. gegen 14.3 Mio. (Verhältnis 0.57): der einzige Fall dieser Demo, in dem es gewinnt. Der Kreuzungspunkt liegt bei n = 13 (tief) bzw. n = 14 (zentral, Verhältnis 0.97); Steepest Edge löst den Würfel dagegen in einem Pivot (Stück 3). Klick auf 'Auf dem Würfel vergleichen'.",
    "Unzulässig: Ellipse ganz abgeschnitten": "Schon die erste Iteration beweist es: die Mindestmenge x1 ≥ 6 (Zeile 2) schneidet die ganze Startkugel ab (α = 1.11 ≥ 1), weil die Ressourcen x1 auf höchstens 4 begrenzen.",
    "Unbeschränkt: Kugelrand": "Nach 34 Iterationen endet die Methode am Rand der künstlichen Startkugel (Zielwert 1999.998, die Schranke der Dienste ist 1000): kein Beweis. Der Simplex meldet unbeschränkt.",
}


def eps_label(i):
    return f"10^-{EPS_EXPS[i]}"


def radius_label(i):
    return f"{RADIUS_FACTORS[i]}-fach"


def scale_label(i):
    return "unskaliert" if SCALE_EXPS[i] == 0 else f"über 10^{SCALE_EXPS[i]}"
