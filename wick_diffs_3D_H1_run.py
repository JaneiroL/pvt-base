#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test-Version für 3D→H1-WickDiffs:

- sucht die NEUESTE 3D-Pivots-Datei in: outputs/pivots/3D
- liest diese CSV
- nimmt einfach NUR die erste Zeile (z.B. CADJPY, 3D, short, ...)
- schreibt diese Zeile 1:1 in eine neue CSV in:
  outputs/wickdiffs/3D→H1/wick_diffs_3D_H1_<timestamp>.csv

Damit testen wir nur:
- Pfad-Finden
- CSV-Einlesen
- CSV-Schreiben in den richtigen Output-Ordner
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime
import sys
import pandas as pd


def find_latest_3d_pivots_csv() -> Path:
    """
    Sucht die neueste Datei vom Muster:
    outputs/pivots/3D/pivots_gap_ALL_3D_*.csv
    """
    base = Path("outputs") / "pivots" / "3D"
    if not base.exists():
        print(f"❌ Verzeichnis für 3D-Pivots existiert nicht: {base}")
        sys.exit(1)

    candidates = list(base.glob("pivots_gap_ALL_3D_*.csv"))
    if not candidates:
        print(f"❌ Keine Datei 'pivots_gap_ALL_3D_*.csv' in {base} gefunden.")
        sys.exit(1)

    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest


def main() -> None:
    # 1) Neueste Pivots-CSV bestimmen
    pivots_path = find_latest_3d_pivots_csv()
    print(f"🔄 Verwende 3D-Pivots CSV: {pivots_path}")

    # 2) CSV einlesen
    try:
        df = pd.read_csv(pivots_path)
    except Exception as e:
        print(f"❌ Konnte CSV nicht lesen: {pivots_path}\n   Fehler: {e}")
        sys.exit(1)

    if df.empty:
        print("⚠️ 3D-Pivots CSV ist leer – keine Zeilen vorhanden.")
        sys.exit(0)

    # 3) Einfach die erste Zeile nehmen (Index 0)
    first_row_df = df.iloc[[0]]  # DataFrame mit genau einer Zeile
    print("✅ Erste Zeile aus der 3D-Pivots-Datei:")
    print(first_row_df.to_string(index=False))

    # 4) Output-Ordner für WickDiffs: outputs/wickdiffs/3D→H1
    out_dir = Path("outputs") / "wickdiffs" / "3D→H1"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 5) Datei-Namen bauen und schreiben
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"wick_diffs_3D_H1_{stamp}.csv"

    try:
        first_row_df.to_csv(out_path, index=False)
    except Exception as e:
        print(f"❌ Konnte Output nicht schreiben: {out_path}\n   Fehler: {e}")
        sys.exit(1)

    print(f"\n💾 Test-Output gespeichert in:\n   {out_path.resolve()}")
    print("\n(Die Datei sollte GENAU diese eine Pivot-Zeile enthalten.)")


if __name__ == "__main__":
    main()
