#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP 2 – Unified WickDiff Builder (INNER / OUTSIDE / ALL)

Ziel:
- WickDiffs auf LTF finden anhand HTF-Pivots (outputs/pivots/<TF>).
- Outputs werden IMMER eindeutig getrennt in 3 Ordner geschrieben:
    outputs/wickdiffs/<TF_LTF>_INNER/
    outputs/wickdiffs/<TF_LTF>_OUTSIDE/
    outputs/wickdiffs/<TF_LTF>_ALL/

Varianten:
1 = INNER   : WD-Zone komplett innerhalb HTF-WickDiff-Zone (htf_wick_diff_low..high)
2 = OUTSIDE : WD-Zone innerhalb Pivot, aber komplett außerhalb HTF-WickDiff-Zone
3 = ALL     : WD-Zone innerhalb Pivot (gap_low..gap_high), egal wo zur HTF-WD-Zone
4 = ALL THREE (schreibt 3 Dateien pro TF)

Gemeinsam:
- Zeitfenster: komplette 1. + komplette 2. HTF-Candle
- Unberührt: nach WD-Entstehung keine Berührung bis HTF-touch (minus Buffer) bzw. Datenende
- 19%-Regel: WD-Zonenbreite <= 19% der HTF Pivot-Range
- Mapping:
    3D -> H1
    W  -> H4
    2W -> D1
    M  -> 3D
"""

import argparse
import sys
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple

import pandas as pd


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

# Buffer wie in euren Step2-Ketten
BUFFER_HOURS_MAP = {"3D": 0, "W": 24, "2W": 48, "M": 96}

# TF -> (Pivot-Subfolder, LTF-Folder, LTF-Label, OutputPrefix)
MODE_SPECS = {
    "3D": ("3D",      Path("time frame data") / "1h data",     "H1", "3D_H1"),
    "W":  ("W",       Path("time frame data") / "4h data",     "H4", "W_H4"),
    "2W": ("2Weekly", Path("time frame data") / "daily data",  "D1", "2W_D1"),
    "M":  ("Monthly", Path("time frame data") / "3D",          "3D", "M_3D"),
}

VARIANT_NAMES = {
    "1": "INNER",
    "2": "OUTSIDE",
    "3": "ALL",
}


# -----------------------------
# Utils
# -----------------------------
def to_dt(s) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=False)

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

CAND_TIME  = ["time", "timestamp", "date", "datetime", "unnamed: 0"]
CAND_OPEN  = ["open", "o"]
CAND_HIGH  = ["high", "h"]
CAND_LOW   = ["low", "l"]
CAND_CLOSE = ["close", "c"]

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

def normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    df = _std_cols(df)
    t = _pick_col(df, CAND_TIME)
    o = _pick_col(df, CAND_OPEN)
    h = _pick_col(df, CAND_HIGH)
    l = _pick_col(df, CAND_LOW)
    c = _pick_col(df, CAND_CLOSE)

    out = df.rename(columns={t:"time", o:"open", h:"high", l:"low", c:"close"})[
        ["time","open","high","low","close"]
    ].copy()

    if pd.api.types.is_numeric_dtype(out["time"]):
        vmax = pd.Series(out["time"]).astype(float).abs().max()
        unit = "ms" if vmax > 1e12 else "s"
        out["time"] = pd.to_datetime(out["time"], unit=unit, utc=False)
    else:
        out["time"] = to_dt(out["time"])

    for col in ["open","high","low","close"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["time","open","high","low","close"]).sort_values("time").reset_index(drop=True)
    return out

def read_ohlc_file(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    if path.suffix.lower() == ".csv":
        return normalize_ohlc(pd.read_csv(path))
    if path.suffix.lower() == ".xlsx":
        try:
            sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")
            for _, df in sheets.items():
                try:
                    return normalize_ohlc(df)
                except Exception:
                    pass
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

def any_touch_between(df: pd.DataFrame, low: float, high: float, start_time: pd.Timestamp, end_time: pd.Timestamp) -> bool:
    seg = df[(df["time"] > start_time) & (df["time"] <= end_time)]
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
    cands = [p for p in pivots_dir.glob("*.csv") if ("pivots" in p.name.lower() and "gap" in p.name.lower())]
    if not cands:
        # fallback: irgendwas mit pivots
        cands = [p for p in pivots_dir.glob("*.csv") if ("pivot" in p.name.lower())]
    if not cands:
        return None
    return max(cands, key=lambda x: x.stat().st_mtime)

def load_pivots_for_mode(base: Path, mode: str) -> pd.DataFrame:
    piv_subdir, _, _, _ = MODE_SPECS[mode]
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

    ft_col = None
    for cand in ["first_touch_time","pivot_touch_time","htf_first_touch_time"]:
        try:
            ft_col = _pick_col_generic(df, cand)
            break
        except KeyError:
            continue

    # optional HTF wickdiff bounds (für INNER/OUTSIDE)
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
        "pivot_touch_time": (to_dt(df[ft_col]) if ft_col else pd.NaT),
        "htf_wick_diff_low": (pd.to_numeric(df[htf_wd_low_col], errors="coerce") if htf_wd_low_col else pd.NA),
        "htf_wick_diff_high": (pd.to_numeric(df[htf_wd_high_col], errors="coerce") if htf_wd_high_col else pd.NA),
    })

    out = out.dropna(subset=["pair6","pivot_type","pivot_first_time","pivot_second_time","pivot_low","pivot_high"]).reset_index(drop=True)
    out["pivot_low"], out["pivot_high"] = out[["pivot_low","pivot_high"]].min(axis=1), out[["pivot_low","pivot_high"]].max(axis=1)

    # normalize htf wickdiff if available
    if "htf_wick_diff_low" in out.columns and "htf_wick_diff_high" in out.columns:
        # wenn beides vorhanden, min/max
        mask = out["htf_wick_diff_low"].notna() & out["htf_wick_diff_high"].notna()
        if mask.any():
            lo = pd.to_numeric(out.loc[mask, "htf_wick_diff_low"], errors="coerce")
            hi = pd.to_numeric(out.loc[mask, "htf_wick_diff_high"], errors="coerce")
            out.loc[mask, "htf_wick_diff_low"] = pd.concat([lo, hi], axis=1).min(axis=1).values
            out.loc[mask, "htf_wick_diff_high"] = pd.concat([lo, hi], axis=1).max(axis=1).values

    return out


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
    df = ltf[(ltf["time"] >= t_start) & (ltf["time"] <= t_end)].reset_index(drop=True)
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
    """
    variant: "INNER" | "OUTSIDE" | "ALL"
    """
    pl, ph = float(pivot_low), float(pivot_high)
    zl, zh = float(min(z_lo, z_hi)), float(max(z_lo, z_hi))

    # muss IMMER im Pivot liegen
    if zl < pl or zh > ph:
        return False

    if variant == "ALL":
        return True

    # INNER/OUTSIDE brauchen HTF wickdiff bounds
    if htf_wd_low is None or htf_wd_high is None:
        return False

    wl, wh = float(min(htf_wd_low, htf_wd_high)), float(max(htf_wd_low, htf_wd_high))

    if variant == "INNER":
        return (zl >= wl) and (zh <= wh)

    if variant == "OUTSIDE":
        return (zh <= wl) or (zl >= wh)

    return False


# -----------------------------
# Run per mode + variant
# -----------------------------
def run_mode_variant(base: Path, mode: str, variant: str) -> pd.DataFrame:
    piv = load_pivots_for_mode(base, mode)
    if piv.empty:
        return pd.DataFrame()

    piv_subdir, ltf_rel, ltf_label, _out_prefix = MODE_SPECS[mode]
    ltf_dir = base / ltf_rel
    ltf_map = find_ltf_files_map(ltf_dir)
    if not ltf_map:
        print(f"❌ Keine LTF-Dateien in {ltf_dir} gefunden.")
        return pd.DataFrame()

    buffer_hours = BUFFER_HOURS_MAP.get(mode, 0)

    rows: List[dict] = []
    skipped_pairs = set()

    piv = piv.sort_values(["pair6","pivot_type","pivot_first_time","pivot_second_time"]).reset_index(drop=True)

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

        # ZEITFENSTER: komplette 1. + komplette 2. HTF-Candle
        htf_t1 = pd.Timestamp(r["pivot_first_time"])
        htf_t2 = pd.Timestamp(r["pivot_second_time"])
        dur = htf_t2 - htf_t1
        if dur <= pd.Timedelta(0):
            continue
        win_start = htf_t1
        win_end = htf_t2 + dur

        # Pivot range
        p_low = float(r["pivot_low"])
        p_high = float(r["pivot_high"])
        width_w = p_high - p_low
        if width_w <= 0:
            continue

        # HTF wickdiff bounds (optional)
        htf_wd_low = None
        htf_wd_high = None
        try:
            if pd.notna(r.get("htf_wick_diff_low", pd.NA)) and pd.notna(r.get("htf_wick_diff_high", pd.NA)):
                htf_wd_low = float(r["htf_wick_diff_low"])
                htf_wd_high = float(r["htf_wick_diff_high"])
        except Exception:
            htf_wd_low = None
            htf_wd_high = None

        # LTF wickdiff candidates (zeit-gefiltert)
        hits = detect_ltf_wickdiffs(
            ltf=ltf_df,
            direction=direction,
            t_start=win_start,
            t_end=win_end,
        )
        if not hits:
            continue

        # Für "Unberührt"
        _, data_end = first_last_time(ltf_df)

        htf_touch = r.get("pivot_touch_time", pd.NaT)
        has_touch = pd.notna(htf_touch)
        if has_touch:
            touch_ts = pd.Timestamp(htf_touch)
            buffer_end = touch_ts - pd.Timedelta(hours=buffer_hours) if buffer_hours > 0 else touch_ts
        else:
            buffer_end = data_end

        for (wd_first, wd_second, z_lo, z_hi) in hits:
            z_width = float(max(z_lo, z_hi) - min(z_lo, z_hi))
            if z_width <= 0:
                continue

            # Variant-price filter
            if not zone_passes_variant(
                variant=variant,
                z_lo=z_lo, z_hi=z_hi,
                pivot_low=p_low, pivot_high=p_high,
                htf_wd_low=htf_wd_low, htf_wd_high=htf_wd_high,
            ):
                continue

            # 19%-Regel (bezogen auf Pivot-Range)
            if (z_width / width_w) > MAX_REL_WIDTH:
                continue

            # Unberührt: nach WD-Entstehung keine Touches bis buffer_end
            wd_second_ts = pd.Timestamp(wd_second)
            if buffer_end > wd_second_ts:
                if any_touch_between(ltf_df, z_lo, z_hi, wd_second_ts, buffer_end):
                    continue

            rows.append({
                "wd_variant": variant,  # <-- wichtig, damit alles später eindeutig ist

                "pair": pair6,
                "pivot_type": direction,
                "timeframe": mode,
                "ltf": ltf_label,

                "htf_first_candle_time": htf_t1,
                "htf_second_candle_time": htf_t2,
                "htf_gap_low": p_low,
                "htf_gap_high": p_high,
                "htf_gap_width": width_w,

                # HTF wickdiff (für INNER/OUTSIDE relevant; für ALL evtl. leer)
                "htf_wick_diff_low": (htf_wd_low if htf_wd_low is not None else pd.NA),
                "htf_wick_diff_high": (htf_wd_high if htf_wd_high is not None else pd.NA),

                "wd_first_candle_time": pd.Timestamp(wd_first),
                "wd_second_candle_time": pd.Timestamp(wd_second),
                "wd_zone_low": float(min(z_lo, z_hi)),
                "wd_zone_high": float(max(z_lo, z_hi)),
                "wd_zone_width": z_width,
                "wd_zone_pct_of_htf_gap": z_width / width_w,

                "htf_first_touch_time": (pd.Timestamp(htf_touch) if has_touch else pd.NaT),
                "pending_until_htf_touch": (not has_touch),
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
        parts = [p.strip() for p in re.split(r"[,\s+/|]+", raw) if p.strip()]
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
    ]
    for c in time_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
    return df


def main():
    ap = argparse.ArgumentParser(description="Step2 WickDiff Builder – getrennte Outputs: INNER / OUTSIDE / ALL.")
    ap.add_argument("--base", type=str, default=".", help="Projekt-Root (wo outputs/ und time frame data/ liegen).")
    ap.add_argument("--tf", type=str, default="All", help="3D | W | 2W | M | Both | All (oder Kombi: 'W,2W,M').")
    ap.add_argument("--variant", type=str, default="", help="1=INNER,2=OUTSIDE,3=ALL,4=ALLE (leer => Prompt).")
    args = ap.parse_args()

    base = Path(args.base).resolve()
    if not base.exists():
        print(f"❌ Base nicht gefunden: {base}")
        sys.exit(1)

    try:
        modes = parse_tf_input(args.tf)
    except Exception as e:
        print(f"❌ {e}")
        sys.exit(1)

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
        _piv_sub, _ltf_rel, _ltf_label, out_prefix = MODE_SPECS[mode]

        for v in variants_to_run:
            vname = VARIANT_NAMES[v]
            df = run_mode_variant(base, mode, vname)
            if df.empty:
                print(f"⚠️ Keine Wickdiffs gefunden für {mode} | {vname}")
                continue

            # dedup
            dedup_cols = [
                "wd_variant","pair","pivot_type",
                "htf_first_candle_time","htf_second_candle_time",
                "wd_first_candle_time","wd_second_candle_time",
                "wd_zone_low","wd_zone_high",
            ]
            df = df.drop_duplicates(subset=dedup_cols, keep="first").reset_index(drop=True)

            df = format_times(df)

            # <<< WICHTIG: Output-Ordner eindeutig pro Variante >>>
            out_dir = out_root / f"{out_prefix}_{vname}"
            out_dir.mkdir(parents=True, exist_ok=True)

            out_path = out_dir / f"wickdiffs_{mode}_{vname}_{stamp}.csv"
            df.to_csv(out_path, index=False)

            print(f"\n✅ Wickdiffs gespeichert: TF={mode} | Variant={vname} | n={len(df)}")
            with pd.option_context("display.width", 240, "display.max_columns", None):
                print(df.head(PREVIEW_ROWS).to_string(index=False))
            print(f"💾 {out_path.resolve()}")

if __name__ == "__main__":
    main()
