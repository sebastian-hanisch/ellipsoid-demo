# Ellipsoid-Methode – polynomial in der Theorie, weit hinter dem Simplex in der Praxis – Streamlit-Demo

Siebtes Stück der **Lineare-Programmierung-Reihe** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning", Kind der Wurzel [tableau-simplex-demo](https://github.com/sebastian-hanisch/tableau-simplex-demo) und Kontrast zu [klee-minty-demo](https://github.com/sebastian-hanisch/klee-minty-demo). Bisher rechnete jedes Stück Simplex-Varianten: von Ecke zu Ecke, im schlimmsten Fall exponentiell viele Pivots. **Khachiyan (1979)** zeigte, dass sich LPs auch **polynomial** lösen lassen – mit einem ganz anderen Verfahren: ein **Ellipsoid**, das die Lösung sicher enthält, wird in jeder Iteration durch eine Nebenbedingung angeschnitten und durch das kleinste Ellipsoid ersetzt, das die Restmenge umschließt. Das Volumen schrumpft je Schnitt mindestens um den Faktor exp(−1/(2(n+1))). Die Theorie sagt "polynomial", die Praxis sagt "unbrauchbar" – die Demo **misst**, was davon stimmt. Vier Fragen: **(1) Die Ellipsen** – was passiert in einer Iteration? **(2) Iterationen** – wie viele braucht es und wovon hängt es ab? **(3) Gegen den Simplex** – was kostet es in Operationen, und gibt es einen Fall, in dem das Ellipsoid gewinnt? **(4) Grenzen** – Schnittarten, Skalierung, Gleichungen, Genauigkeit.

**Einordnung in die Reihe:** geplant sind zwölf Stücke, dies ist das siebte (Details in `lp-planung/PLAN.md` des Portfolio-Ordners):

```
Tableau-Simplex (Wurzel)                                                                  [gebaut: tableau-simplex-demo]
 ├─ Pivotregeln & Entartung ─ Simplex im schlimmsten und im typischen Fall (Klee-Minty)   [gebaut: pivotregeln-demo, klee-minty-demo]
 ├─ Revised Simplex ─ Präsolve, Skalierung & Numerik                                     [gebaut: revised-simplex-demo]  →  [nicht gebaut]
 ├─ Dualität & Sensitivität ─ Dualer Simplex & Neuoptimierung                            [gebaut: lp-dualitaet-demo, dualer-simplex-demo]
 ├─ Ellipsoid-Methode (Kontrast: polynomial in der Theorie)                              [DIESES STÜCK]
 └─ Innere Punkte ─ PDLP (Verfahren erster Ordnung) ─ Crossover & Simplex gegen Innere Punkte gegen PDLP  [nicht gebaut]
```

Ergebnis in Kürze: **Die Iterationen wachsen sauber wie n² · ln(1/ε), aber die Operationen liegen auf Zufallsinstanzen um den Faktor 160 bis 2100 hinter dem Simplex – nur auf dem Klee-Minty-Würfel dreht sich das, ab n = 13 bis 14.** Auf Zufallsinstanzen braucht das Ellipsoid bei ε = 10^-6 **24 bis 25.5 · n² Iterationen** (102 bei n = 2, 916 bei n = 6, 2380 bei n = 10, 9748 bei n = 20), der Simplex 1 bis 14 Pivots. Jede weitere Stelle Genauigkeit kostet gleich viele Iterationen (bei n = 8: 291, das sind 88 % der Theorie-Steigung 2n(n+1) ln 10 = 332). **Tiefe Schnitte** sparen 12 bis 22 % der Iterationen. Die **Theorie-Schranke** für den ersten zulässigen Mittelpunkt liegt weit über dem Gemessenen (bei n = 8 gemessen 10 Iterationen gegen 360). Auf dem **Klee-Minty-Würfel** (n = 14) braucht der Simplex mit Dantzig-Regel 16383 Pivots, das Ellipsoid 4052 Iterationen: **8.2 Mio. gegen 14.3 Mio. Operationen (Verhältnis 0.57)** – aber nur gegen die schlechte Pivotregel; Steepest Edge löst den Würfel in einem Pivot. **Numerik:** das Zertifikat gelingt bis ε = 10^-12 in allen getesteten Instanzen (n = 4 bis 40) und scheitert ab 10^-13 in allen, unabhängig von n; bei schlecht skalierten Spalten (10^12) bricht es durch Rundung ab, und der Simplex liefert dann in 3 von 5 Fällen stillschweigend einen falschen Wert.

| Frage | Ergebnis (Auslastungsplanung als Standard-LP max c·x; Zufallsinstanzen m = n, Median über 5 feste Instanzen, Seeds 100000–100004; Operationen im Modell, keine Wandzeit; vollständig deterministisch) |
|---|---|
| **Stimmt das Verfahren?** | ✅ Volumenverhältnis je zentralem Schnitt gleich der geschlossenen Form (n = 2 bis 20, 1e-9) und höchstens exp(−1/(2(n+1))); **das Optimum liegt in jeder Ellipse** (über 1000 Iterationen, zentrale und tiefe Schnitte, beide Zeilenwahlen); Optimum gleich HiGHS und Simplex auf 366 Läufen bis ε = 10^-7; Zertifikatslücke ≤ ε; 60 konstruierte unzulässige Instanzen werden mit beiden Schnittarten erkannt, nie als Optimum ausgegeben; Gleichungen bis Aufweitung 10^-6 |
| **Lehrbuch von Hand** | Start ist die Kugel um den Kasten [0, 4] × [0, 6] (Mittelpunkt (2, 3)); Iteration 1: Mittelpunkt zulässig, Zielwert 21, Objektivschnitt; Iteration 3: die Lagerfläche wird verletzt. Nach **100 Iterationen** (tiefe Schnitte 78) ist das Optimum 36 bis 10^-6 bewiesen; der Simplex braucht 2 Pivots. Zentrum: 594 Iterationen (tief 490) gegen 4 Pivots, 154.440 gegen 400 Operationen (386-Fache) |
| **Iterationen über n** | ε = 10^-6, zentrale Schnitte: n = 2 / 4 / 6 / 8 / 10 / 12 / 16 / 20: **102 / 389 / 916 / 1611 / 2380 / 3677 / 6135 / 9748**; Iterationen / n² = 25.5 / 24.3 / 25.4 / 25.2 / 23.8 / 25.5 / 24.0 / 24.4 |
| **Iterationen über ε** | n = 8: 448 / 733 / 1022 / 1611 / 2199 / 2777 bei ε = 10^-2 / 10^-3 / 10^-4 / 10^-6 / 10^-8 / 10^-10, das sind **291 je Zehnerpotenz** (88 % der Theorie 332); n = 4: 73 (79 % von 92), n = 12: 658 (92 % von 718) |
| **Startkugel** | n = 8: Faktor 1 / 2 / 4 / 16 / 64: 1611 / 1702 / 1787 / 1955 / 2141 Iterationen (+33 % bei 64-fach), bis zum ersten zulässigen Mittelpunkt 10 / 12 / 55 / 224 / 407: nur logarithmisch, aber der Weg zum ersten zulässigen Punkt wächst stark |
| **Schnittarten** | Tiefe gegen zentrale Schnitte sparen bei n = 2 / 4 / 6 / 8 / 10 / 12 / 16 / 20: **22 / 19 / 17 / 13 / 13 / 12 / 15 / 13 %**; erste gegen am stärksten verletzte Zeile: höchstens 2 % Unterschied (bei zentralen Schnitten; mal besser, mal schlechter) |
| **Theorie-Schranke** | Erster zulässiger Mittelpunkt gegen 2n(n+1) ln(r0/r) (r: größte Kugel in der zulässigen Menge, mit dem Simplex berechnet): n = 2 / 3 / 4 / 6 / 8 / 10: gemessen 1 / 2 / 3 / 5 / 10 / 10 gegen 8.5 / 42 / 68 / 187 / 360 / 624, also **12 / 4.7 / 4.4 / 2.6 / 2.8 / 1.6 %** der Schranke; nie über 13 % |
| **Operationen gegen den Simplex** | Zufall m = n, ε = 10^-6: Verhältnis Ellipsoid / Simplex bei n = 2 / 4 / 6 / 8 / 10 / 12 / 16 / 20: **163 / 380 / 644 / 1179 / 765 / 2104 / 2050 / 1650**; der Simplex braucht dabei 1 / 2 / 3 / 3 / 7 / 4 / 7 / 14 Pivots. **Kein Kreuzungspunkt** im Bereich n ≤ 20 |
| **Klee-Minty-Würfel** | Simplex (Dantzig) braucht **2^n − 1 Pivots** (3 bei n = 2, 16383 bei n = 14); Ellipsoid tief: 81 / 810 / 2185 / 3099 / 3541 / 4052 Iterationen bei n = 2 / 6 / 10 / 12 / 13 / 14. Verhältnis der Operationen (Ellipsoid / Simplex) bei n = 2 / 6 / 10 / 12 / 13 / 14: **43 / 27 / 4.8 / 1.7 / 1.0 / 0.57**; zentrale Schnitte: 2.7 bei n = 12 und **0.97** bei n = 14. **Kreuzungspunkt bei n = 13 (tief) bzw. 14 (zentral)**; je gröber ε, desto früher (tief: n = 11 / 12 / 13 / 14 bei ε = 10^-2 / 10^-3 / 10^-6 / 10^-8; ab 10^-10 gibt es im Bereich n ≤ 14 keinen) |
| **Genauigkeitsgrenze** | Zufall n = 4, 14, 40 (je zwei Instanzen, zentrale Schnitte): das Zertifikat gelingt bei ε = 10^-10 und 10^-12 immer und scheitert bei 10^-13 und 10^-14 immer, unabhängig von n; Zufall 8 × 8, ε = 10^-14: Abbruch nach 3491 Iterationen, der beste Wert weicht dann nur um 2 · 10^-10 vom Simplex ab |
| **Schlechte Skalierung** | Zufall 6 × 8, Spalten mit Faktoren zwischen 1 und 10^k umgerechnet (Optimalwert gleich): Iterationen 1615 / 1803 / 2082 / 2375 / 2656 bei k = 0 / 2 / 4 / 6 / 8, die Methode bricht bei 10^8 in 1, bei 10^10 in 4 und bei 10^12 in 5 von 5 Instanzen durch Rundung ab; falsch (Fehler über 0.1 %) ist sie bei 10^12 in 3 von 5. Der Simplex bleibt bis 10^10 exakt und liefert bei 10^12 in 3 von 5 Fällen still einen falschen Wert (Fehler bis 87 %) |
| **Gleichungen** | Mischung 8 × 8 (Seed 35, eine Gleichung), als zwei Ungleichungen mit Aufweitung δ: erster zulässiger Mittelpunkt bei δ = 10^-2 / 10^-4 / 10^-6 / 10^-8 in Iteration **28 / 46 / 64 / 83** (ln(1/δ)); bei 10^-10 scheitert die Numerik nach etwa 80 bis 100 Iterationen (die Stelle hängt von der Plattform ab: Windows 84, Linux 97) |
| **Unzulässig, unbeschränkt** | Unzulässig: das Lehrbuch mit Mindestmenge x1 ≥ 6 wird in Iteration 1 erkannt (α = 1.11 ≥ 1: die Zeile schneidet die ganze Startkugel ab). Unbeschränkt: das Verfahren endet nach 34 Iterationen am Rand der künstlichen Startkugel (Zielwert 1999.998, Schranke 1000 je Dienst), also **ohne Beweis**; bei 4-facher Kugel bricht die Numerik ab |

## Vorab-Hypothesen

| Hypothese (vor der Messung) | Ergebnis |
|---|---|
| Iterationen ~ n²·ln(1/ε) | **Bestätigt, sehr sauber:** Iterationen / n² liegt für n = 2 bis 20 zwischen 23.8 und 25.5; je Zehnerpotenz Genauigkeit 79 bis 92 % der Theorie-Steigung |
| Das Ellipsoid braucht 10² bis 10⁴ mal mehr Operationen als der Simplex | **Bestätigt:** 163- bis 2104-fach auf Zufallsinstanzen (n ≤ 20); die Lücke wächst nicht monoton (n = 10: 765) |
| Tiefe Schnitte sparen 20 bis 50 % | **Teils widerlegt:** 12 bis 22 %, bei großem n eher 13 % |
| Die Theorie-Schranke ist weit über dem Gemessenen | **Bestätigt:** höchstens 13 %, bei n = 10 nur 1.6 % |
| P verliert die positive Definitheit ab ε = 10^-8 bei n ≥ 15 | **Widerlegt:** bei Zufallsinstanzen bis n = 40 gelingt das Zertifikat bis ε = 10^-12; darunter scheitert es in allen Instanzen gleich, unabhängig von n. Brüche gibt es bei Gleichungen mit sehr dünnem Inneren (10^-10), bei schlechter Skalierung (ab 10^8) und bei der 4-fachen Kugel der unbeschränkten Instanz |
| Der Simplex gewinnt überall | **Widerlegt, aber nur knapp und nur gegen die schlechte Regel:** auf dem Klee-Minty-Würfel braucht das Ellipsoid ab n = 13 (tief) bzw. 14 (zentral) weniger Operationen als der Simplex mit Dantzig-Regel; Steepest Edge und größter Zuwachs lösen den Würfel in einem Pivot (Stück 3) und schlagen das Ellipsoid dort um Größenordnungen |
| Die Wahl der Zeile ändert die Iterationen | **Widerlegt** (zentrale Schnitte): höchstens 2 %, mal besser, mal schlechter |

## Was die Demo zeigt

1. **Vier Schritte** (Schritt-Slider): **Die Ellipsen** (bei zwei Diensten das zulässige Vieleck, die Startkugel, die Ellipse der gewählten Iteration mit Mittelpunkt, Schnittgerade und bisherigen Mittelpunkten, Slider über die Iterationen, Ausschnitt oder ganze Kugel; sonst Tabelle und immer die Volumenkurve gegen die Garantie) → **Iterationen** (Zertifikatslücke und tatsächlicher Fehler über die Iterationen; auf Abruf Iterationen über n, ε, Startkugel und die Prüfung der Theorie-Schranke) → **Gegen den Simplex** (Operationen dieses Falls, auf Abruf über n und der Klee-Minty-Würfel mit Kreuzungspunkt) → **Grenzen** (kleinster Eigenwert von P; auf Abruf Schnittarten, schlechte Skalierung, Gleichungen).
2. **Regler:** Instanz (Lehrbuch, Zentrum, entartete Ecke, Zufall, Mischung, Klee-Minty-Würfel, Unzulässig, Unbeschränkt), Größe, Schnittart (zentral oder tief), Wahl der verletzten Zeile, Genauigkeit ε (10^-2 bis 10^-14), Startkugel (1- bis 64-fach), Skalierung der Spalten (bis 10^12).
3. **Zertifikat:** das Ergebnis wird als "Optimum" gemeldet, wenn die obere Schranke c·x + √(cᵀPc) um höchstens ε vom besten zulässigen Wert entfernt ist; unzulässig, wenn ein Schnitt die ganze Ellipse abschneidet, bevor ein zulässiger Punkt gefunden wurde; "Kugelrand" bei Verdacht auf Unbeschränktheit; "numerisch gescheitert", wenn Rundung die Invariante verletzt.

Presets (12): Lehrbuch: die ersten Ellipsen, Zentrum: Ellipsoid gegen Simplex, Iterationen wachsen mit n², Genauigkeit: Iterationen je Zehnerpotenz, Tiefe Schnitte, Die Theorie-Schranke ist weit weg, Klee-Minty-Würfel: hier gewinnt das Ellipsoid, Genauigkeitsgrenze: 10^-14 geht nicht, Schlechte Skalierung, Gleichungen: dünnes Inneres, Unzulässig: Ellipse ganz abgeschnitten, Unbeschränkt: Kugelrand.

## Modell und Verfahren

- **Instanz** (`ell_scenario.py`): die Auslastungsplanung der Vorgängerstücke (Lehrbuch, Zentrum, entartete Ecke, Zufall, Mischung, Unzulässig, Unbeschränkt) plus der Klee-Minty-Würfel in Chvátals Form (Stück 3, n ≤ 14: alle Zahlen exakt). `column_scaled` rechnet die Spalten mit Faktoren zwischen 1 und 10^k um (Optimalwert gleich).
- **Startkugel** (`ell_ellipsoid.start_ball`): Kasten [0, U] aus den Zeilen mit nichtnegativen Koeffizienten; Kugel um seine Mitte mit dem Radius "Faktor mal halbe Diagonale". Dienste ohne endliche Schranke (unbeschränkte Instanz) bekommen eine künstliche.
- **Ellipsoid** (`ell_ellipsoid.py`): je Iteration Mittelpunkt prüfen (alle Zeilen einschließlich x ≥ 0, Gleichungen als zwei Ungleichungen mit Aufweitung 10^-6), Machbarkeits- oder Objektivschnitt (c·z ≥ bester Wert), zentrales oder tiefes Update, Symmetrisierung von P, log-Volumen mitgeführt; Zertifikat und Fehlerstatus wie oben. Operationsmodell je Iteration 2m'n + 6n² + 4n.
- **Simplex** (`ell_algorithm.py`): der Zwei-Phasen-Tableau-Simplex der Stücke 1 bis 6 (Dantzig-Regel mit Bland-Notbremse), Operationsmodell 2(m+1)(Spalten+1) je Pivot.
- **Auswertung** (`ell_evaluation.py`): Einzellauf gegen den Simplex, Größen-, Genauigkeits- und Radiuskurven, Schnittarten, Theorie-Schranke (Inkugelradius über den Simplex), Skalierung, Gleichungen, Klee-Minty-Würfel.

## Was nicht funktioniert hat / Grenzen

- **Vorab-Hypothese "P bricht ab ε = 10^-8 bei großem n" – widerlegt.** Zufallsinstanzen halten bis ε = 10^-12 durch (n bis 40), egal wie groß n ist; die Bruchstelle liegt bei der Rundungsgenauigkeit der Zertifikatslücke, nicht bei der Größe. Meine erste Version hatte die "ganze Ellipse abgeschnitten"-Prüfung nur bei tiefen Schnitten und hat bei zentralen stillschweigend weitergerechnet; die Prüfung gilt jetzt für beide, und wenn sie nach einem zulässigen Punkt auslöst, ist das ein Rundungsfehler und wird als numerischer Abbruch gemeldet.
- **Der Ellipsoid gewinnt nur gegen eine schwache Simplex-Regel.** Der Kreuzungspunkt auf dem Klee-Minty-Würfel gilt für die Dantzig-Regel; mit Steepest Edge (ein Pivot, Stück 3) gibt es keinen. Der Würfel ist ein konstruierter Fall, und n ≤ 14 ist die Grenze, bis zu der alle Zahlen exakt bleiben (die Pivotzahl 2^n − 1 für n > 14 wird nicht mehr gemessen).
- **Kein Kreuzungspunkt auf Zufallsinstanzen.** Bis n = 20 liegt das Ellipsoid um den Faktor 160 bis 2100 hinter dem Simplex; größere Instanzen sind nicht gemessen.
- **Nur die Grundform.** Objektivschnitte statt der Optimierung über Bisektion des Zielwerts (Grötschel, Lovász, Schrijver), keine exakte Arithmetik und keine Rundung nach Khachiyan (Gitter- und Präzisionsargument nur genannt), keine Surrogatschnitte oder Schnitte in mehreren Richtungen.
- **Gleichungen sind aufgeweitet** (10^-6 in der App); je dünner die zulässige Menge, desto später der erste zulässige Punkt (Iteration 28 bis 83 von 10^-2 bis 10^-8), bei 10^-10 bricht die Numerik ab. Transportprobleme (nur Gleichungen) sind daher nicht dabei.
- **Unbeschränkte Instanzen liefern keinen Beweis**, nur den Rand der künstlichen Kugel; die Demo meldet das ausdrücklich. Bei 4-facher Kugel bricht die Numerik ab (P wird extrem gestreckt).
- **Operationsmodell, keine Wandzeit.** Speicherzugriffe und Bibliothekseffekte (numpy) sind nicht modelliert; das Ellipsoid ist im Modell nicht günstiger gerechnet als es ist.
- **Synthetische Instanzen, n ≤ 14 im Regler** (Größenkurve bis n = 20), Skalierungstest auf fünf Zufallsinstanzen; die Aussage über den Simplex bei schlechter Skalierung gilt für diesen Löser mit absoluter Toleranz 10^-9 und ohne Präsolve (das ist das Thema des Präsolve-Stücks).

## Verifikation

- `tests/test_algorithm.py`: **Volumenformel** (n = 2 bis 20) und Update-Formeln (Randpunkt der tiefen Schnitt-Ellipse, Stichprobe: Restmenge liegt in der neuen Ellipse); **Optimum in jeder Ellipse** (über 1000 Iterationen); Optimum gleich HiGHS und Simplex auf über 300 Läufen; Gleichungen bis 10^-6; unzulässig (Fixture und 60 konstruierte Instanzen, beide Schnittarten); unbeschränkt endet am Kugelrand; Theorie-Schranke für den ersten zulässigen Mittelpunkt (Würfel-Konstruktionen n = 2 bis 6); Iterationen wachsen mit der Genauigkeit und die Suche nach dem ersten zulässigen Punkt mit ln(1/Aufweitung); n = 1 als Bisektion; Buchführung des Operationsmodells; jeder Zweig (Machbarkeits-, Objektivschnitt, unzulässig, Kugelrand, Grenze, numerisch).
- `tests/test_scenario.py` (auch der Würfel gegen HiGHS), `test_evaluation.py`, `test_presets.py` (jede Zahl der Hilfetexte), `test_claims.py` (jede Zahl aus README und App über die echten `ev.*`-Funktionen; Iterationszahlen mit kleinem Band, weil Gleitkomma-Rundung sie plattformabhängig um wenige verschieben kann), `test_app.py` (Streamlit-AppTest: Voreinstellung, jedes Preset, jeder Schritt für jede Instanz und Schnittart, Iterations-Regler, Regler-Randwerte, Permalink-Grenzen, bedingte Regler, Berechnungen auf Abruf, Footer).
- Für die Prüfung genügt **pytest**; `scipy` dient nur als Gegenprobe (`requirements-dev.txt`), die App braucht nur numpy, pandas, plotly und streamlit.

## Lokal starten

```bash
python -m venv venv && venv/Scripts/activate  # Windows; Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Tests: `pip install -r requirements-dev.txt` und `python -m pytest tests/ -W error::SyntaxWarning`.

## Literatur

- Khachiyan, L. G. (1979). *A polynomial algorithm in linear programming.* Doklady Akademii Nauk SSSR 244(5), 1093–1096 (englisch: Soviet Mathematics Doklady 20, 191–194).
- Bland, R. G., Goldfarb, D., & Todd, M. J. (1981). *The ellipsoid method: a survey.* Operations Research 29(6), 1039–1091.
- Grötschel, M., Lovász, L., & Schrijver, A. (1988). *Geometric Algorithms and Combinatorial Optimization.* Springer.

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
