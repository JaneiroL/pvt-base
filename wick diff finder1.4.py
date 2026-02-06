#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP 2 – Unified WickDiff Builder (INNER / OUTSIDE / ALL)

Ziel:
- WickDiffs auf LTF finden anhand HTF-Pivots (outputs/pivots/<TF>).
- Outputs werden IMMER eindeutig getrennt in 3 Ordner geschrieben:
    outputs/wickdiffs/<HTF>_<LTF>_INNER/
    outputs/wickdiffs/<HTF>_<LTF>_OUTSIDE/
    outputs/wickdiffs/<HTF>_<LTF>_ALL/

Varianten:
1 = INNER   : WD-Zone komplett innerhalb HTF-WickDiff-Zone (htf_wick_diff_low..high)
2 = OUTSIDE : WD-Zone innerhalb Pivot, aber komplett außerhalb HTF-WickDiff-Zone
3 = ALL     : WD-Zone innerhalb Pivot (gap_low..gap_high), egal wo zur HTF-WD-Zone
4 = ALL THREE (schreibt 3 Dateien pro TF)

Gemeinsam:
- Zeitfenster: komplette 1. + komplette 2. HTF-Candle (HARTE GRENZE am Ende von Candle #2)
- Unberührt (HARTE RULE, Pivot-CSV):
    * Ab wd_second_candle_time darf die WD-Zone NICHT mehr berührt werden bis pivot-first-touch (first_touch_time).
    * Wenn first_touch_time fehlt -> cutoff = Datenende, und WD muss bis Datenende unberührt bleiben.
    * WICHTIG: Touch exakt bei first_touch_time ist erlaubt (Entry-Touch), d.h. bei has_touch prüfen wir (wd_second, first_touch_time) ohne first_touch_time selbst.
- 19%-Regel: WD-Zonenbreite <= 19% der HTF Pivot-Range

Standard-Mapping (wenn du KEIN LTF angibst):
    3D-HTF -> H1-LTF
    W-HTF  -> H4-LTF
    2W-HTF -> D1-LTF
    M-HTF  -> 3D-LTF
"""

import argparse
import sys
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple

import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype

# -----------------------------
# Settings
# -----------------------------
MAX_REL_WIDTH = 0.19  # 19% Regel
PREVIEW_ROWS = 15

PAIRS_28 = {
    "AUDCAD","AUDCHF","AUDJPY","AUDNZD","AUDUSD",
    "CADCHF","CADJPY",
    "CHFJPY",
    "EURAUD","EURCAD","EURCHF","EURGBP","EURJPY","EURNZD","EURUSD",
    "GBPAUD","GBPCAD","GBPCHF","GBPJPY","GBPUSD","GBPNZD",
    "NZDCAD","NZDCHF","NZDJPY","NZDUSD",
    "USDCAD","USDCHF","USDJPY",
}

SPECIAL_PAIR_FIX = {"OANDAG": "GBPNZD"}

# Buffer wie in Step2-Ketten (BLEIBT definiert, aber wird NICHT mehr in der Unberührt-Logik verwendet)
BUFFER_HOURS_MAP = {"3D": 0, "W": 24, "2W": 48, "M": 96}

# HTF-Modes: Pivot-Ordner + Default-LTF-Key
MODE_SPECS = {
    "3D": {"pivot_subdir": "3D",      "default_ltf": "H1"},
    "W":  {"pivot_subdir": "W",       "default_ltf": "H4"},
    "2W": {"pivot_subdir": "2Weekly", "default_ltf": "D1"},
    "M":  {"pivot_subdir": "Monthly", "default_ltf": "3D"},
}

# Lower-Timeframe-Ordner
LTF_DIRS = {
    "H1":  Path("time frame data") / "1h data",
    "H4":  Path("time frame data") / "4h data",
    "H12": Path("time frame data") / "12h",
    "D1":  Path("time frame data") / "daily data",
    "3D":  Path("time frame data") / "3D",
}

# Higher-Timeframe-Ordner (für harte Zeitgrenzen)
HTF_DATA_DIRS = {
    "3D": Path("time frame data") / "3D",
    "W":  Path("time frame data") / "W",
    "2W": Path("time frame data") / "2Weekly",
    "M":  Path("time frame data") / "Monthly",
}

VARIANT_NAMES = {
    "1": "INNER",
    "2": "OUTSIDE",
    "3": "ALL",
}

# Spaltenkandidaten für OHLC
CAND_TIME  = ["time", "timestamp", "date", "datetime", "unnamed: 0"]
CAND_OPEN  = ["open", "o"]
CAND_HIGH  = ["high", "h"]
CAND_LOW   = ["low", "l"]
CAND_CLOSE = ["close", "c"]

# -----------------------------
# Utils – Zeit / OHLC
# -----------------------------
def to_dt(s):
    """
    Robust gegen gemischte Zeitzonen + numerische Unix-Timestamps.

    Verhalten:
    - Wenn die Eingabe numerisch ist (Series/Index oder Skalar):
        * |value| > 1e12  => Millisekunden-Timestamps
        * sonst           => Sekunden-Timestamps
    - Wenn die Eingabe Strings/Datumsobjekte sind:
        * Standard-pandas-Parsing mit utc=True
    - Immer:
        * Erst in UTC parsen, dann Zeitzone entfernen => tz-naiv
    """
    if isinstance(s, (pd.Series, pd.Index)):
        if is_numeric_dtype(s):
            vmax = pd.Series(s).astype(float).abs().max()
            unit = "ms" if vmax > 1e12 else "s"
            dt = pd.to_datetime(s, unit=unit, errors="coerce", utc=True)
            return dt.dt.tz_convert(None)
        dt = pd.to_datetime(s, errors="coerce", utc=True)
        return dt.dt.tz_convert(None)

    if isinstance(s, (int, float, np.integer, np.floating)):
        vmax = abs(float(s))
        unit = "ms" if vmax > 1e12 else "s"
        dt = pd.to_datetime([s], unit=unit, errors="coerce", utc=True)[0]
    else:
        dt = pd.to_datetime([s], errors="coerce", utc=True)[0]

    try:
        return dt.tz_convert(None)
    except Exception:
        return dt


def _std_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def _pick_col(df: pd.DataFrame, cands: List[str]) -> str:
    low = {str(c).strip().lower(): c for c in df.columns}
    for cand in cands:
        if cand in low:
            return low[cand]
    for c in df.columns:
        lc = str(c).strip().lower()
        for cand in cands:
            if cand in lc:
                return c
    for c in df.columns:
        if str(c).strip().lower().startswith("unnamed"):
            return c
    raise KeyError(f"Missing one of {cands}; got {list(df.columns)}")

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

def infer_pair_from_text(txt: str) -> Optional[str]:
    up = re.sub(r"[^A-Z]", "", str(txt).upper())
    for bad, real in SPECIAL_PAIR_FIX.items():
        if bad in up:
            return real
    for p in PAIRS_28:
        if p in up:
            return p
    m = re.search(r"([A-Z]{6})", up)
    return m.group(1) if m else None

def pair_code_from_str(s: str) -> str:
    p = infer_pair_from_text(s)
    code = (p or str(s)[:6]).upper()
    return SPECIAL_PAIR_FIX.get(code, code)

def read_ohlc_from_df(df: pd.DataFrame) -> pd.DataFrame:
    df = _std_cols(df)
    tcol = _pick_col(df, CAND_TIME)
    ocol = _pick_col(df, CAND_OPEN)
    hcol = _pick_col(df, CAND_HIGH)
    lcol = _pick_col(df, CAND_LOW)
    ccol = _pick_col(df, CAND_CLOSE)

    df = df.rename(columns={tcol: "time", ocol: "open", hcol: "high", lcol: "low", ccol: "close"})
    df["time"] = to_dt(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close", "time"]).reset_index(drop=True)
    return df[["time", "open", "high", "low", "close"]]

def try_read_ohlc_with_header_scan(df_raw: pd.DataFrame) -> pd.DataFrame:
    try:
        return read_ohlc_from_df(df_raw)
    except Exception:
        pass

    df = df_raw.copy()
    df.columns = [str(c) for c in df.columns]
    best_row = None

    for i in range(min(25, len(df))):
        row_vals = [str(x).strip().lower() for x in list(df.iloc[i].values)]
        ohlc_hits = sum(any(re.search(rf"\b{k}\b", v) for v in row_vals) for k in ["open", "high", "low", "close"])
        time_hits = any(re.search(r"\b(time|timestamp|date|datetime)\b", v) for v in row_vals)
        if ohlc_hits >= 3 and time_hits:
            best_row = i
            break

    if best_row is None:
        return read_ohlc_from_df(df)

    new_cols = [str(x).strip() for x in list(df.iloc[best_row].values)]
    df2 = df.iloc[best_row + 1 :].reset_index(drop=True)
    df2.columns = new_cols
    return read_ohlc_from_df(df2)

def read_ohlc_file(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        return try_read_ohlc_with_header_scan(df)
    if path.suffix.lower() == ".xlsx":
        try:
            sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl", header=0)
            for _, df in sheets.items():
                try:
                    return try_read_ohlc_with_header_scan(df)
                except Exception:
                    continue
        except Exception:
            return None
    return None

def find_ltf_files_map(ltf_dir: Path) -> Dict[str, Path]:
    mp: Dict[str, Path] = {}
    if not ltf_dir.exists():
        return mp
    for p in ltf_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".csv",".xlsx"}:
            code = pair_code_from_str(p.name)
            if len(code) == 6 and code in PAIRS_28 and code not in mp:
                mp[code] = p
    return mp

def candle_color(o: float, c: float) -> str:
    if c > o:
        return "bull"
    if c < o:
        return "bear"
    return "doji"

def any_touch_between(df: pd.DataFrame, low: float, high: float,
                      start_time: pd.Timestamp, end_time: pd.Timestamp) -> bool:
    # legacy helper: (start, end]  (inkl. end)
    seg = df[(df["time"] > start_time) & (df["time"] <= end_time)]
    if seg.empty:
        return False
    lo = min(low, high)
    hi = max(low, high)
    return bool(((seg["high"] >= lo) & (seg["low"] <= hi)).any())

def any_touch_between_end_exclusive(df: pd.DataFrame, low: float, high: float,
                                   start_time: pd.Timestamp, end_time: pd.Timestamp) -> bool:
    # NEW helper: (start, end)  (exkl. end)
    seg = df[(df["time"] > start_time) & (df["time"] < end_time)]
    if seg.empty:
        return False
    lo = min(low, high)
    hi = max(low, high)
    return bool(((seg["high"] >= lo) & (seg["low"] <= hi)).any())

def first_last_time(df: pd.DataFrame) -> Tuple[pd.Timestamp, pd.Timestamp]:
    return df["time"].iloc[0], df["time"].iloc[-1]

# -----------------------------
# Pivot loader
# -----------------------------
def latest_pivots_csv(pivots_dir: Path) -> Optional[Path]:
    if not pivots_dir.exists():
        return None
    cands = [p for p in pivots_dir.glob("*.csv")
             if ("pivots" in p.name.lower() and "gap" in p.name.lower())]
    if not cands:
        cands = [p for p in pivots_dir.glob("*.csv") if ("pivot" in p.name.lower())]
    if not cands:
        return None
    return max(cands, key=lambda x: x.stat().st_mtime)

def load_pivots_for_mode(base: Path, mode: str) -> pd.DataFrame:
    piv_subdir = MODE_SPECS[mode]["pivot_subdir"]
    piv_dir = base / "outputs" / "pivots" / piv_subdir
    p = latest_pivots_csv(piv_dir)
    if p is None:
        print(f"❌ Keine Pivot-CSV in {piv_dir} gefunden.")
        return pd.DataFrame()

    print(f"🔄 Verwende Pivots ({mode}): {p}")
    df = pd.read_csv(p)
    df.columns = [str(c) for c in df.columns]

    def pick_any(names: List[str]) -> str:
        for n in names:
            try:
                return _pick_col_generic(df, n)
            except KeyError:
                pass
        raise KeyError(f"None of these columns found: {names}")

    pair_col = pick_any(["pair","pair6","symbol","instrument"])
    ptype_col = pick_any(["pivot_type","direction","side"])
    gap_low_col = pick_any(["gap_low","pivot_low","htf_gap_low"])
    gap_high_col = pick_any(["gap_high","pivot_high","htf_gap_high"])
    fc_col = pick_any(["first_candle_time","pivot_first_time","htf_first_candle_time"])
    sc_col = pick_any(["second_candle_time","pivot_second_time","htf_second_candle_time"])

    # A) first_touch_time zwingend laden (wenn vorhanden)
    ft_col = None
    for cand in ["first_touch_time","pivot_touch_time","htf_first_touch_time"]:
        try:
            ft_col = _pick_col_generic(df, cand)
            break
        except KeyError:
            continue

    htf_wd_low_col = None
    htf_wd_high_col = None
    for cand in ["htf_wick_diff_low","wick_diff_low","htf_wd_low","wd_low_htf"]:
        try:
            htf_wd_low_col = _pick_col_generic(df, cand)
            break
        except KeyError:
            continue
    for cand in ["htf_wick_diff_high","wick_diff_high","htf_wd_high","wd_high_htf"]:
        try:
            htf_wd_high_col = _pick_col_generic(df, cand)
            break
        except KeyError:
            continue

    out = pd.DataFrame({
        "pair6": df[pair_col].astype(str).apply(pair_code_from_str),
        "pivot_type": df[ptype_col].astype(str).str.lower().str.strip(),
        "pivot_first_time": to_dt(df[fc_col]),
        "pivot_second_time": to_dt(df[sc_col]),
        "pivot_low": pd.to_numeric(df[gap_low_col], errors="coerce"),
        "pivot_high": pd.to_numeric(df[gap_high_col], errors="coerce"),

        # pivot_first_touch_time (Step1 Journaling) – kann NaT sein
        "pivot_first_touch_time": (to_dt(df[ft_col]) if ft_col else pd.NaT),

        "htf_wick_diff_low": (pd.to_numeric(df[htf_wd_low_col], errors="coerce")
                              if htf_wd_low_col else pd.NA),
        "htf_wick_diff_high": (pd.to_numeric(df[htf_wd_high_col], errors="coerce")
                               if htf_wd_high_col else pd.NA),
    })

    out = out.dropna(subset=[
        "pair6","pivot_type","pivot_first_time","pivot_second_time",
        "pivot_low","pivot_high"
    ]).reset_index(drop=True)
    out["pivot_low"], out["pivot_high"] = out[["pivot_low","pivot_high"]].min(axis=1), \
                                          out[["pivot_low","pivot_high"]].max(axis=1)

    mask = out["htf_wick_diff_low"].notna() & out["htf_wick_diff_high"].notna()
    if mask.any():
        lo = pd.to_numeric(out.loc[mask, "htf_wick_diff_low"], errors="coerce")
        hi = pd.to_numeric(out.loc[mask, "htf_wick_diff_high"], errors="coerce")
        out.loc[mask, "htf_wick_diff_low"] = pd.concat([lo, hi], axis=1).min(axis=1).values
        out.loc[mask, "htf_wick_diff_high"] = pd.concat([lo, hi], axis=1).max(axis=1).values

    return out

# -----------------------------
# HTF window end derivation (harte Grenze)
# -----------------------------
def _fallback_htf_end(mode: str, t2: pd.Timestamp) -> pd.Timestamp:
    t2 = pd.Timestamp(t2)
    if mode == "3D":
        return t2 + pd.Timedelta(days=3) - pd.Timedelta(nanoseconds=1)
    if mode == "W":
        return t2 + pd.Timedelta(days=7) - pd.Timedelta(nanoseconds=1)
    if mode == "2W":
        return t2 + pd.Timedelta(days=14) - pd.Timedelta(nanoseconds=1)
    if mode == "M":
        return (t2 + pd.offsets.MonthBegin(1)) - pd.Timedelta(nanoseconds=1)
    return t2 + pd.Timedelta(days=7) - pd.Timedelta(nanoseconds=1)

def _nearest_index_by_time(times: pd.Series, target: pd.Timestamp) -> Optional[int]:
    if times.empty:
        return None
    t = pd.Timestamp(target)
    arr = times.values
    pos = int(np.searchsorted(arr, np.datetime64(t)))
    candidates = []
    if 0 <= pos < len(arr):
        candidates.append(pos)
    if 0 <= pos-1 < len(arr):
        candidates.append(pos-1)
    if 0 <= pos+1 < len(arr):
        candidates.append(pos+1)
    if not candidates:
        return None
    best = min(candidates, key=lambda i: abs(pd.Timestamp(arr[i]) - t))
    return int(best)

def derive_window_end_from_htf(
    htf_df: pd.DataFrame,
    mode: str,
    t1: pd.Timestamp,
    t2: pd.Timestamp,
) -> Optional[pd.Timestamp]:
    """
    Liefert das Ende der 2. HTF-Candle:
    - Suche Candle bei t2 (oder nächstliegend)
    - Ende = Start der nächsten HTF-Candle - 1ns
    - Falls keine nächste Candle existiert: Ende = Datenende
    """
    if htf_df is None or htf_df.empty:
        return None

    df = htf_df.copy()
    df["time"] = to_dt(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    idx2 = _nearest_index_by_time(df["time"], pd.Timestamp(t2))
    if idx2 is None:
        return None

    if idx2 + 1 < len(df):
        next_time = pd.Timestamp(df.loc[idx2 + 1, "time"])
        return next_time - pd.Timedelta(nanoseconds=1)

    return pd.Timestamp(df["time"].iloc[-1])

# -----------------------------
# Wickdiff detection (LTF)
# -----------------------------
def detect_ltf_wickdiffs(
    ltf: pd.DataFrame,
    direction: str,
    t_start: pd.Timestamp,
    t_end: pd.Timestamp,
) -> List[Tuple[pd.Timestamp, pd.Timestamp, float, float]]:
    """
    Wickdiff Definition:
      long:  bear -> bull  => Zone über Lows
      short: bull -> bear  => Zone über Highs
    Nur Zeitfilter hier – Preisfilter macht die Variante-Logik.
    """
    if ltf.empty:
        return []
    df = ltf.copy()
    df["time"] = to_dt(df["time"])
    t_start = pd.Timestamp(t_start)
    t_end = pd.Timestamp(t_end)

    df = df[(df["time"] >= t_start) & (df["time"] <= t_end)].reset_index(drop=True)
    if len(df) < 2:
        return []

    out: List[Tuple[pd.Timestamp, pd.Timestamp, float, float]] = []
    for i in range(len(df) - 1):
        o1, h1, l1, c1, tt1 = df.loc[i,   ["open","high","low","close","time"]]
        o2, h2, l2, c2, tt2 = df.loc[i+1, ["open","high","low","close","time"]]
        col1, col2 = candle_color(o1, c1), candle_color(o2, c2)
        if col1 == "doji" or col2 == "doji":
            continue

        if direction == "long":
            if not (col1 == "bear" and col2 == "bull"):
                continue
            z_lo = float(min(l1, l2))
            z_hi = float(max(l1, l2))
        else:
            if not (col1 == "bull" and col2 == "bear"):
                continue
            z_lo = float(min(h1, h2))
            z_hi = float(max(h1, h2))

        out.append((pd.Timestamp(tt1), pd.Timestamp(tt2), z_lo, z_hi))

    return out

def zone_passes_variant(
    variant: str,
    z_lo: float, z_hi: float,
    pivot_low: float, pivot_high: float,
    htf_wd_low: Optional[float], htf_wd_high: Optional[float],
) -> bool:
    pl, ph = float(pivot_low), float(pivot_high)
    zl, zh = float(min(z_lo, z_hi)), float(max(z_lo, z_hi))

    if zl < pl or zh > ph:
        return False

    if variant == "ALL":
        return True

    if htf_wd_low is None or htf_wd_high is None:
        return False

    wl, wh = float(min(htf_wd_low, htf_wd_high)), float(max(htf_wd_low, htf_wd_high))

    if variant == "INNER":
        return (zl >= wl) and (zh <= wh)

    if variant == "OUTSIDE":
        return (zh <= wl) or (zl >= wh)

    return False

# -----------------------------
# Run per (HTF, LTF, Variant)
# -----------------------------
def run_mode_variant(base: Path, mode: str, ltf_key: str, variant: str) -> pd.DataFrame:
    piv = load_pivots_for_mode(base, mode)
    if piv.empty:
        return pd.DataFrame()

    ltf_dir_rel = LTF_DIRS[ltf_key]
    ltf_dir = base / ltf_dir_rel
    ltf_map = find_ltf_files_map(ltf_dir)
    if not ltf_map:
        print(f"❌ Keine LTF-Dateien in {ltf_dir} gefunden.")
        return pd.DataFrame()

    htf_dir = base / HTF_DATA_DIRS[mode]
    htf_map = find_ltf_files_map(htf_dir)

    rows: List[dict] = []
    skipped_pairs = set()
    warned_end = set()

    piv = piv.sort_values(["pair6","pivot_type","pivot_first_time","pivot_second_time"]).reset_index(drop=True)

    htf_cache: Dict[str, pd.DataFrame] = {}

    for _, r in piv.iterrows():
        pair6 = r["pair6"]
        direction = str(r["pivot_type"]).lower().strip()
        if direction not in {"long","short"}:
            continue

        ltf_path = ltf_map.get(pair6)
        if not ltf_path:
            skipped_pairs.add(pair6)
            continue

        ltf_df = read_ohlc_file(ltf_path)
        if ltf_df is None or ltf_df.empty:
            skipped_pairs.add(pair6)
            continue

        htf_t1 = pd.Timestamp(r["pivot_first_time"])
        htf_t2 = pd.Timestamp(r["pivot_second_time"])
        if htf_t2 <= htf_t1:
            continue

        # Window: Candle #1 start bis Candle #2 Ende (harte Grenze)
        win_start = htf_t1

        htf_df = None
        htf_path = htf_map.get(pair6)
        if htf_path:
            if pair6 in htf_cache:
                htf_df = htf_cache[pair6]
            else:
                htf_df = read_ohlc_file(htf_path)
                htf_cache[pair6] = (htf_df if htf_df is not None else pd.DataFrame())

        end2 = derive_window_end_from_htf(htf_df, mode, htf_t1, htf_t2) if htf_df is not None else None
        if end2 is None:
            if pair6 not in warned_end:
                print(
                    f"⚠️ Konnte HTF-Ende (2. Candle) nicht aus HTF-Daten ableiten für {mode}/{pair6}. "
                    f"Fallback auf feste TF-Länge/Monatswechsel."
                )
                warned_end.add(pair6)
            win_end = _fallback_htf_end(mode, htf_t2)
        else:
            win_end = pd.Timestamp(end2)

        p_low = float(r["pivot_low"])
        p_high = float(r["pivot_high"])
        width_w = p_high - p_low
        if width_w <= 0:
            continue

        htf_wd_low = None
        htf_wd_high = None
        try:
            if pd.notna(r.get("htf_wick_diff_low", pd.NA)) and pd.notna(r.get("htf_wick_diff_high", pd.NA)):
                htf_wd_low = float(r["htf_wick_diff_low"])
                htf_wd_high = float(r["htf_wick_diff_high"])
        except Exception:
            htf_wd_low = None
            htf_wd_high = None

        hits = detect_ltf_wickdiffs(
            ltf=ltf_df,
            direction=direction,
            t_start=win_start,
            t_end=win_end,
        )
        if not hits:
            continue

        # B) cutoff bestimmen NUR aus Pivot-CSV first_touch_time
        _, data_end = first_last_time(ltf_df)

        pivot_first_touch = r.get("pivot_first_touch_time", pd.NaT)
        has_touch = pd.notna(pivot_first_touch)

        if has_touch:
            touch_ts = pd.Timestamp(pivot_first_touch)
            cutoff = touch_ts
            pending = False
        else:
            cutoff = data_end
            pending = True

        for (wd_first, wd_second, z_lo, z_hi) in hits:
            z_width = float(max(z_lo, z_hi) - min(z_lo, z_hi))
            if z_width <= 0:
                continue

            if not zone_passes_variant(
                variant=variant,
                z_lo=z_lo, z_hi=z_hi,
                pivot_low=p_low, pivot_high=p_high,
                htf_wd_low=htf_wd_low, htf_wd_high=htf_wd_high,
            ):
                continue

            if (z_width / width_w) > MAX_REL_WIDTH:
                continue

            wd_second_ts = pd.Timestamp(wd_second)

            # Sonderfall: cutoff <= start => unsinniges Fenster
            if pd.Timestamp(cutoff) <= wd_second_ts:
                continue

            # B2) Unberührt-Check:
            # - bei has_touch: prüfe (wd_second, cutoff) EXKLUSIV cutoff (Touch exakt am cutoff ist erlaubt)
            # - bei pending:  prüfe (wd_second, cutoff] INKLUSIV cutoff (bis Datenende unberührt)
            if has_touch:
                if any_touch_between_end_exclusive(ltf_df, z_lo, z_hi, wd_second_ts, pd.Timestamp(cutoff)):
                    continue
            else:
                if any_touch_between(ltf_df, z_lo, z_hi, wd_second_ts, pd.Timestamp(cutoff)):
                    continue

            rows.append({
                "wd_variant": variant,
                "pair": pair6,
                "pivot_type": direction,
                "htf": mode,
                "ltf": ltf_key,

                "htf_first_candle_time": htf_t1,
                "htf_second_candle_time": htf_t2,
                "htf_gap_low": p_low,
                "htf_gap_high": p_high,
                "htf_gap_width": width_w,

                "htf_wick_diff_low": (htf_wd_low if htf_wd_low is not None else pd.NA),
                "htf_wick_diff_high": (htf_wd_high if htf_wd_high is not None else pd.NA),

                "wd_first_candle_time": pd.Timestamp(wd_first),
                "wd_second_candle_time": pd.Timestamp(wd_second),
                "wd_zone_low": float(min(z_lo, z_hi)),
                "wd_zone_high": float(max(z_lo, z_hi)),
                "wd_zone_width": z_width,
                "wd_zone_pct_of_htf_gap": z_width / width_w,

                # C) Output-Spalten: First-Touch nur aus Pivot-CSV
                "htf_first_touch_time": (pd.Timestamp(pivot_first_touch) if has_touch else pd.NaT),
                "pending_until_htf_touch": pending,

                # helpful debug (optional, harmless)
                "wd_search_window_start": win_start,
                "wd_search_window_end": win_end,
            })

    if skipped_pairs:
        print("ℹ️ Übersprungen (fehlende/ungültige LTF-Datei):", ", ".join(sorted(skipped_pairs)))

    return pd.DataFrame(rows)

# -----------------------------
# CLI / interactive
# -----------------------------
def parse_tf_input(s: str) -> List[str]:
    if s is None:
        raise ValueError("Ungültige Eingabe. Erlaubt: 3D, W, 2W, M, Both, All")
    raw = s.strip().lower()
    if not raw:
        raise ValueError("Ungültige Eingabe. Erlaubt: 3D, W, 2W, M, Both, All")

    if raw in {"both", "w+3d", "3d+w"}:
        wanted = {"3D","W"}
    elif raw in {"all", "everything"}:
        wanted = {"3D","W","2W","M"}
    else:
        parts = [p.strip() for p in re.split(r"[,\s+/|;]+", raw) if p.strip()]
        alias = {
            "3d":"3D","3":"3D","3tage":"3D","3-tage":"3D","3day":"3D",
            "w":"W","week":"W","weekly":"W","1w":"W",
            "2w":"2W","2week":"2W","2weekly":"2W","2-week":"2W","2wöchig":"2W",
            "m":"M","mon":"M","month":"M","monthly":"M","1m":"M",
            "both":"BOTH","all":"ALL",
        }
        wanted = set()
        for p in parts:
            if p not in alias:
                raise ValueError("Ungültige Eingabe. Erlaubt: 3D, W, 2W, M, Both, All")
            v = alias[p]
            if v == "BOTH":
                wanted |= {"3D","W"}
            elif v == "ALL":
                wanted |= {"3D","W","2W","M"}
            else:
                wanted.add(v)

    order = ["3D","W","2W","M"]
    return [x for x in order if x in wanted]

def parse_ltf_input(s: str) -> Optional[List[str]]:
    if s is None:
        return None
    raw = s.strip().lower()
    if not raw:
        return None

    if raw in {"all", "everything"}:
        return ["H1","H4","H12","D1"]

    parts = [p.strip() for p in re.split(r"[,\s+/|;]+", raw) if p.strip()]
    alias = {
        "h1":"H1","1h":"H1","60":"H1",
        "h4":"H4","4h":"H4","240":"H4",
        "h12":"H12","12h":"H12","720":"H12",
        "d1":"D1","1d":"D1","24h":"D1","daily":"D1",
    }
    wanted = []
    for p in parts:
        if p not in alias:
            raise ValueError("Ungültige LTF-Eingabe. Erlaubt: H1, H4, H12, D1, All oder Kombis.")
        v = alias[p]
        if v not in wanted:
            wanted.append(v)
    if not wanted:
        return None
    return wanted

def prompt_variant() -> str:
    print("\nWelche Variante willst du laufen lassen?")
    print("  1 = INNER   (nur innerhalb HTF-WickDiff-Zone)")
    print("  2 = OUTSIDE (innerhalb Pivot, aber außerhalb HTF-WickDiff-Zone)")
    print("  3 = ALL     (innerhalb Pivot, egal wo)")
    print("  4 = ALLE 3  (schreibt 3 Outputs)")
    v = input("Eingabe (1/2/3/4): ").strip()
    if v not in {"1","2","3","4"}:
        raise ValueError("Ungültige Variante. Erlaubt: 1,2,3,4")
    return v

def format_times(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    time_cols = [
        "htf_first_candle_time","htf_second_candle_time",
        "wd_first_candle_time","wd_second_candle_time","htf_first_touch_time",
        "wd_search_window_start","wd_search_window_end",
    ]
    for c in time_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
    return df

def main():
    ap = argparse.ArgumentParser(
        description="Step2 WickDiff Builder – getrennte Outputs: INNER / OUTSIDE / ALL."
    )
    ap.add_argument("--base", type=str, default=".",
                    help="Projekt-Root (wo outputs/ und time frame data/ liegen).")
    ap.add_argument("--tf", type=str, default="",
                    help="HTF: 3D | W | 2W | M | Both | All (oder Kombi: '3D,W').")
    ap.add_argument("--ltf", type=str, default="",
                    help="Unteres TF: H1,H4,H12,D1,All (oder Kombis: 'H1,H4').")
    ap.add_argument("--variant", type=str, default="",
                    help="1=INNER,2=OUTSIDE,3=ALL,4=ALLE (leer => Prompt).")
    args = ap.parse_args()

    base = Path(args.base).resolve()
    if not base.exists():
        print(f"❌ Base nicht gefunden: {base}")
        sys.exit(1)

    if not args.tf:
        print("Welche HTF-Pivots willst du nutzen?")
        print("  Optionen: 3D | W | 2W | M | Both | All (oder Kombis: '3D,W')")
        args.tf = input("HTF-Eingabe: ").strip()

    try:
        modes = parse_tf_input(args.tf)
    except Exception as e:
        print(f"❌ {e}")
        sys.exit(1)

    if not args.ltf:
        print("\nUnteres Timeframe (LTF) für WickDiff-Suche wählen.")
        print("  Optionen: H1, H4, H12, D1, All  (oder Kombis: 'H1,H4')")
        print("  Leer lassen = Standard pro HTF (3D→H1, W→H4, 2W→D1, M→3D)")
        args.ltf = input("LTF-Eingabe: ").strip()

    try:
        ltf_list = parse_ltf_input(args.ltf)
    except Exception as e:
        print(f"❌ {e}")
        sys.exit(1)

    mode_to_ltfs: Dict[str,List[str]] = {}
    if ltf_list is None:
        for m in modes:
            mode_to_ltfs[m] = [MODE_SPECS[m]["default_ltf"]]
    else:
        for m in modes:
            mode_to_ltfs[m] = ltf_list[:]

    variant_in = args.variant.strip()
    if not variant_in:
        try:
            variant_in = prompt_variant()
        except Exception as e:
            print(f"❌ {e}")
            sys.exit(1)
    if variant_in not in {"1","2","3","4"}:
        print("❌ Ungültige Variante. Erlaubt: 1,2,3,4")
        sys.exit(1)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = base / "outputs" / "wickdiffs"
    out_root.mkdir(parents=True, exist_ok=True)

    variants_to_run = ["1","2","3"] if variant_in == "4" else [variant_in]

    for mode in modes:
        for ltf_key in mode_to_ltfs[mode]:
            if ltf_key not in LTF_DIRS:
                print(f"⚠️ LTF {ltf_key} ist nicht konfiguriert, skip.")
                continue

            for v in variants_to_run:
                vname = VARIANT_NAMES[v]
                df = run_mode_variant(base, mode, ltf_key, vname)
                if df.empty:
                    print(f"⚠️ Keine Wickdiffs gefunden für {mode} | {ltf_key} | {vname}")
                    continue

                dedup_cols = [
                    "wd_variant","pair","pivot_type",
                    "htf_first_candle_time","htf_second_candle_time",
                    "wd_first_candle_time","wd_second_candle_time",
                    "wd_zone_low","wd_zone_high",
                ]
                df = df.drop_duplicates(subset=dedup_cols, keep="first").reset_index(drop=True)
                df = format_times(df)

                out_dir = out_root / f"{mode}_{ltf_key}_{vname}"
                out_dir.mkdir(parents=True, exist_ok=True)

                out_path = out_dir / f"wickdiffs_{mode}_{ltf_key}_{vname}_{stamp}.csv"
                df.to_csv(out_path, index=False)

                print(f"\n✅ Wickdiffs gespeichert: HTF={mode} | LTF={ltf_key} | Variant={vname} | n={len(df)}")
                with pd.option_context("display.width", 240, "display.max_columns", None):
                    print(df.head(PREVIEW_ROWS).to_string(index=False))
                print(f"💾 {out_path.resolve()}")

if __name__ == "__main__":
    main()
