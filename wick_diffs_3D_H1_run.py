#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse, re
from pathlib import Path
import pandas as pd
import numpy as np

# -------------------------
# Einstellungen
# -------------------------
MAX_REL_WIDTH = 0.19   # 19% Grenze

# -------------------------
# I/O: robuste CSV/Excel Leser (OHLC)
# -------------------------
CAND_TIME  = ["time", "timestamp", "date", "datetime", "unnamed: 0"]
CAND_OPEN  = ["open", "o"]
CAND_HIGH  = ["high", "h"]
CAND_LOW   = ["low", "l"]
CAND_CLOSE = ["close", "c"]


def _pick_col(df: pd.DataFrame, cands: list[str]) -> str:
    low = {str(c).strip().lower(): c for c in df.columns}
    # 1) exakte Matches
    for cand in cands:
        if cand in low:
            return low[cand]
    # 2) Teil-String-Matches
    for c in df.columns:
        lc = str(c).strip().lower()
        for cand in cands:
            if cand in lc:
                return c
    # 3) Fallback: "unnamed"
    for c in df.columns:
        if str(c).strip().lower().startswith("unnamed"):
            return c
    raise KeyError(f"Missing one of {cands}; got {list(df.columns)}")


def _normalize_dt_naive(series: pd.Series) -> pd.Series:
    """
    Datums-Spalte parsen und sicherstellen, dass sie TZ-NAIV ist.
    (Zeitzone wird entfernt, Werte bleiben als "lokale" Zeit.)
    """
    s = pd.to_datetime(series, errors="coerce", utc=False)
    try:
        # wenn tz-aware -> tz entfernen
        if getattr(s.dt, "tz", None) is not None:
            s = s.dt.tz_convert(None)
    except (TypeError, AttributeError):
        pass
    return s


def _normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    t = _pick_col(df, CAND_TIME)
    o = _pick_col(df, CAND_OPEN)
    h = _pick_col(df, CAND_HIGH)
    l = _pick_col(df, CAND_LOW)
    c = _pick_col(df, CAND_CLOSE)

    out = df.rename(
        columns={t: "time", o: "open", h: "high", l: "low", c: "close"}
    )[["time", "open", "high", "low", "close"]].copy()

    out["time"] = _normalize_dt_naive(out["time"])

    for col in ["open", "high", "low", "close"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = (
        out.dropna(subset=["time", "open", "high", "low", "close"])
           .sort_values("time")
           .reset_index(drop=True)
    )
    return out


def read_ohlc_file(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    if path.suffix.lower() == ".csv":
        return _normalize_ohlc(pd.read_csv(path))
    try:
        sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")
        for _, df in sheets.items():
            try:
                return _normalize_ohlc(df)
            except Exception:
                pass
    except Exception:
        return None
    return None


# -------------------------
# 3D-Pivots laden
# -------------------------
def _pick_col_generic(df: pd.DataFrame, name: str) -> str:
    low = {str(c).lower(): c for c in df.columns}
    n2 = name.lower()
    if n2 in low:
        return low[n2]
    spaced = n2.replace("_", " ")
    if spaced in low:
        return low[spaced]
    for c in df.columns:
        if n2 in str(c).lower():
            return c
    raise KeyError(name)


def load_3d_pivots(path: Path) -> pd.DataFrame:
    """
    Erwartet eine pivots_gap_ALL_3D_*.csv wie von pivots_multi.py erzeugt.
    Nutzt nur Reihen mit timeframe == '3D'.
    """
    df = pd.read_csv(path)

    out = pd.DataFrame({
        "pair": df[_pick_col_generic(df, "pair")].astype(str),
        "timeframe": df[_pick_col_generic(df, "timeframe")].astype(str),
        "pivot_type": df[_pick_col_generic(df, "pivot_type")].astype(str).str.lower(),
        "first_candle_time": _normalize_dt_naive(
            df[_pick_col_generic(df, "first_candle_time")]
        ),
        "second_candle_time": _normalize_dt_naive(
            df[_pick_col_generic(df, "second_candle_time")]
        ),
        "gap_low": pd.to_numeric(df[_pick_col_generic(df, "gap_low")], errors="coerce"),
        "gap_high": pd.to_numeric(df[_pick_col_generic(df, "gap_high")], errors="coerce"),
    })

    # optional: first_touch_time
    try:
        ft_colname = _pick_col_generic(df, "first_touch_time")
        out["first_touch_time"] = _normalize_dt_naive(df[ft_colname])
    except KeyError:
        out["first_touch_time"] = pd.NaT

    # Gap-Order normalisieren
    lo = out[["gap_low", "gap_high"]].min(axis=1)
    hi = out[["gap_low", "gap_high"]].max(axis=1)
    out["gap_low"], out["gap_high"] = lo, hi

    # Paarcode auf 6 Buchstaben extrahieren
    out["pair6"] = (
        out["pair"]
        .str.upper()
        .str.replace(r"[^A-Z]", "", regex=True)
        .str.extract(r"([A-Z]{6})", expand=False)
        .fillna(out["pair"].str.upper())
    )

    # Nur 3D-Pivots, sinnvolle Zeilen
    out = out[out["timeframe"].str.upper().str.contains("3D")]
    out = out.dropna(
        subset=["first_candle_time", "second_candle_time", "gap_low", "gap_high"]
    ).reset_index(drop=True)

    return out


# -------------------------
# Utils
# -------------------------
def any_touch_between(
    df: pd.DataFrame,
    low: float,
    high: float,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
) -> bool:
    """
    Gibt es IRGENDEINE Kerze, deren Range [low,high] schneidet,
    im Intervall (start_time, end_time] ?
    """
    if pd.isna(end_time):
        seg = df[df["time"] > start_time]
    else:
        seg = df[(df["time"] > start_time) & (df["time"] <= end_time)]
    if seg.empty:
        return False
    lo = min(low, high)
    hi = max(low, high)
    return bool(((seg["high"] >= lo) & (seg["low"] <= hi)).any())


def first_last_time(df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    return df["time"].iloc[0], df["time"].iloc[-1]


# -------------------------
# Kernlogik: pro 3D-Pivot im 1h-Chart scannen
# (exakt wie Weekly→H4-Goldstandard, nur 3D→H1)
# -------------------------
def scan_pair_for_3d_pivot_h1(
    h1: pd.DataFrame,
    pair: str,
    pivot_row: pd.Series
) -> list[dict]:
    ptype = pivot_row["pivot_type"]         # "long" / "short"
    t1    = pd.Timestamp(pivot_row["first_candle_time"])
    t2    = pd.Timestamp(pivot_row["second_candle_time"])
    d_touch = pivot_row.get("first_touch_time", pd.NaT)

    lo3  = float(pivot_row["gap_low"])
    hi3  = float(pivot_row["gap_high"])
    lo3, hi3 = (min(lo3, hi3), max(lo3, hi3))
    width3 = hi3 - lo3
    if width3 <= 0:
        return []

    # Zeitfenster: von Start der ersten 3D-Kerze bis Ende der 3 Tage der zweiten Kerze
    win_start = t1.normalize()  # 00:00 dieses Tages
    win_end   = t2 + pd.Timedelta(days=2, hours=23, minutes=59, seconds=59)

    dfw = h1[(h1["time"] >= win_start) & (h1["time"] <= win_end)].reset_index(drop=True)
    if dfw.shape[0] < 2:
        return []

    _, data_end = first_last_time(h1)
    results: list[dict] = []

    o = dfw["open"].to_numpy()
    h = dfw["high"].to_numpy()
    l = dfw["low"].to_numpy()
    c = dfw["close"].to_numpy()
    t = dfw["time"].to_numpy()

    # 1 = bull, -1 = bear, 0 = doji
    col = np.where(c > o, 1, np.where(c < o, -1, 0))

    # zusätzliche Mini-Debug-Infos je Pair
    raw_candidates = 0
    inside_gap     = 0
    width_ok       = 0
    unberuehrt     = 0

    for i in range(len(dfw) - 1):
        col1, col2 = col[i], col[i + 1]

        if ptype == "short":
            # 3D-Short → im 1h-Chart bull → bear
            if not (col1 == 1 and col2 == -1):
                continue
            z_lo = float(min(h[i], h[i + 1]))
            z_hi = float(max(h[i], h[i + 1]))
        elif ptype == "long":
            # 3D-Long → im 1h-Chart bear → bull
            if not (col1 == -1 and col2 == 1):
                continue
            z_lo = float(min(l[i], l[i + 1]))
            z_hi = float(max(l[i], l[i + 1]))
        else:
            continue

        raw_candidates += 1

        # komplett innerhalb des 3D-Gaps
        if not (z_lo >= lo3 and z_hi <= hi3):
            continue
        inside_gap += 1

        # Breite ≤ 19% der 3D-Range
        z_width = z_hi - z_lo
        if (z_width / width3) > MAX_REL_WIDTH:
            continue
        width_ok += 1

        t_a = pd.Timestamp(t[i])
        t_b = pd.Timestamp(t[i + 1])

        # --- UNBERÜHRT-REGEL ---
        end_check = pd.Timestamp(d_touch) if pd.notna(d_touch) else data_end
        touched = any_touch_between(h1, z_lo, z_hi, t_b, end_check)
        if touched:
            continue
        unberuehrt += 1

        results.append({
            "pair": pair,
            "pivot_type": ptype,
            "pivot_tf": "3D",
            "pivot_first_candle_time": t1,
            "pivot_second_candle_time": t2,
            "pivot_gap_low": lo3,
            "pivot_gap_high": hi3,
            "pivot_gap_width": width3,
            "ltf": "H1",
            "wd_first_candle_time": t_a,
            "wd_second_candle_time": t_b,
            "wd_zone_low": z_lo,
            "wd_zone_high": z_hi,
            "wd_zone_width": z_width,
            "wd_zone_pct_of_pivot": z_width / width3,
            "pivot_first_touch_time": (
                pd.Timestamp(d_touch) if pd.notna(d_touch) else pd.NaT
            ),
            "pending_until_pivot_touch": pd.isna(d_touch),
        })

    if raw_candidates > 0:
        print(
            f"   {pair}: "
            f"raw={raw_candidates}, in_gap={inside_gap}, "
            f"≤19%={width_ok}, unberührt={unberuehrt}"
        )

    return results


# -------------------------
# Pair-Dateien im H1-Ordner finden
# -------------------------
def pair_code_from_str(s: str) -> str:
    up = re.sub(r"[^A-Z]", "", str(s).upper())
    m = re.search(r"[A-Z]{6}", up)
    return m.group(0) if m else up[:6] or str(s)


def find_h1_files_map(h1_dir: Path) -> dict[str, Path]:
    mp: dict[str, Path] = {}
    if not h1_dir.exists():
        return mp
    for p in h1_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".csv", ".xlsx"}:
            code = pair_code_from_str(p.name)
            if len(code) == 6 and code not in mp:
                mp[code] = p
    return mp


# -------------------------
# Helpers für Standard-Pfade
# -------------------------
def find_latest_3d_pivots_file() -> Path | None:
    base = Path("outputs") / "pivots" / "3D"
    if not base.exists():
        return None
    candidates = list(base.glob("pivots_gap_ALL_3D_*.csv"))
    if not candidates:
        return None
    # neueste nach Änderungszeit
    return max(candidates, key=lambda p: p.stat().st_mtime)


def default_h1_dir() -> Path:
    # Genau dein Ordnername mit Leerzeichen:
    return Path("time frame data") / "1h data"


def default_output_dir() -> Path:
    return Path("outputs") / "wickdiffs" / "3D→H1"


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser(
        description=(
            "Finde 1h Wick-Differences innerhalb 3D-Pivot-Gaps "
            "(≤19% + UNBERÜHRT bis 3D-Touch / Datenende)."
        )
    )
    ap.add_argument(
        "--pivots3d",
        help="Optional: Pfad zur 3D-Pivots CSV (Standard: neueste in outputs/pivots/3D/)",
        default=None,
    )
    ap.add_argument(
        "--h1dir",
        help="Optional: Ordner mit 1h-CSV/Excel pro Pair (Standard: 'time frame data/1h data')",
        default=None,
    )
    ap.add_argument(
        "--only_pair",
        default=None,
        help="Optional: nur dieses 6-Letter Pair scannen, z.B. AUDCHF",
    )
    ap.add_argument(
        "--out",
        default=None,
        help=(
            "Ausgabe-CSV (voller Pfad, optional). "
            "Standard: outputs/wickdiffs/3D→H1/wick_diffs_3D_H1_<timestamp>.csv"
        ),
    )
    ap.add_argument(
        "--preview",
        type=int,
        default=20,
        help="Zeilen in der Terminal-Vorschau (Default 20)",
    )
    args = ap.parse_args()

    # --- 3D-Pivots-Datei bestimmen ---
    if args.pivots3d:
        pivots_path = Path(args.pivots3d)
    else:
        pivots_path = find_latest_3d_pivots_file()

    if pivots_path is None or not pivots_path.exists():
        print("❌ Keine 3D-Pivots CSV gefunden (erwartet in outputs/pivots/3D/).")
        return

    print(f"🔎 Verwende 3D-Pivots CSV: {pivots_path}")

    pivots3d = load_3d_pivots(pivots_path)
    if pivots3d.empty:
        print("⚠️ Keine 3D-Pivots in der angegebenen Datei gefunden.")
        return

    # Optionales Pair-Filter
    if args.only_pair:
        p6 = pair_code_from_str(args.only_pair)
        pivots3d = pivots3d[pivots3d["pair6"] == p6].reset_index(drop=True)
        print(f"🔎 Filter: nur Pair {p6}")
        if pivots3d.empty:
            print("⚠️ Für dieses Pair gibt es keine 3D-Pivots.")
            return

    # --- H1-Ordner bestimmen ---
    if args.h1dir:
        h1_dir = Path(args.h1dir)
    else:
        h1_dir = default_h1_dir()

    if not h1_dir.exists():
        print(f"❌ 1h-Ordner nicht gefunden: {h1_dir}")
        return
    print(f"📂 Verwende 1h-Ordner: {h1_dir}")

    h1_map = find_h1_files_map(h1_dir)

    all_rows: list[dict] = []
    skipped: list[str] = []

    for _, row in pivots3d.iterrows():
        pair6 = pair_code_from_str(row["pair"])
        h1_path = h1_map.get(pair6)
        if not h1_path:
            skipped.append(pair6)
            continue

        h1_df = read_ohlc_file(h1_path)
        if h1_df is None or h1_df.empty:
            skipped.append(pair6)
            continue

        print(f"\n▶ Scanne {pair6} im 1h-Chart für 3D-Pivot "
              f"{row['first_candle_time'].date()}–{row['second_candle_time'].date()}")

        all_rows += scan_pair_for_3d_pivot_h1(h1_df, pair6, row)

    if not all_rows:
        print(
            "\n⚠️ Keine gültigen 1h-Wick-Differences gefunden "
            "(Regeln: Richtung, komplett im 3D-Gap, ≤19%, UNBERÜHRT bis 3D-Touch/Datenende)."
        )
        if skipped:
            print("ℹ️ Übersprungen (keine/ungültige 1h-Datei):",
                  ", ".join(sorted(set(skipped))))
        return

    out = pd.DataFrame(all_rows)

    # Duplikate entfernen
    dedup_cols = [
        "pair",
        "pivot_type",
        "pivot_first_candle_time",
        "pivot_second_candle_time",
        "wd_first_candle_time",
        "wd_second_candle_time",
        "wd_zone_low",
        "wd_zone_high",
    ]
    out = out.drop_duplicates(subset=dedup_cols, keep="first").reset_index(drop=True)

    # Times formatieren
    time_cols = [
        "pivot_first_candle_time",
        "pivot_second_candle_time",
        "wd_first_candle_time",
        "wd_second_candle_time",
        "pivot_first_touch_time",
    ]
    for col in time_cols:
        out[col] = pd.to_datetime(out[col]).dt.strftime("%Y-%m-%d %H:%M")

    # Ausgabe-Pfad
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    if args.out:
        out_path = Path(args.out)
        out_dir = out_path.parent
    else:
        out_dir = default_output_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"wick_diffs_3D_H1_{stamp}.csv"

    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"\n✅ Gefundene (UNBERÜHRTE) 1h-Wick-Differences: {len(out)}")
    with pd.option_context("display.width", 220, "display.max_columns", None):
        print(out.head(args.preview).to_string(index=False))
    print(f"\n💾 Ergebnisse gespeichert in: {out_path.resolve()}")
    if skipped:
        print("ℹ️ Übersprungen (keine/ungültige 1h-Datei):",
              ", ".join(sorted(set(skipped))))


if __name__ == "__main__":
    main()
