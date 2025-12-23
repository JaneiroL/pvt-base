#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP 2 (Variante 3):
- Nimmt die HTF-Pivots (großer Pivot / gap_low..gap_high) aus outputs/pivots/<TF>
- Sucht *innerhalb des gesamten großen Pivots* (ZEIT + PREIS) nach LTF-Wick-Differences
- Mapping (wie gewohnt):
    3D  -> H1
    W   -> H4
    2W  -> D1
    M   -> 3D
- Output bleibt im "Wickdiff-Format" (so wie Step2/Tradejournal es erwartet)
  und landet in:
    outputs/wickdiffs/3D→H1/
    outputs/wickdiffs/W→H4/
    outputs/wickdiffs/2W→1D/
    outputs/wickdiffs/M→3D/
"""

import argparse, sys, re
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple

import pandas as pd


# -----------------------------------
# Globale Einstellungen
# -----------------------------------
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

# Zeitfenster nach Pivot-Touch (wie bei dir in Step 3 / Tradejournal)
MAX_DAYS_MAP = {"3D": 6, "W": 14, "2W": 21, "M": 42}

# TF -> (Pivot-Ordner, LTF-Datenordner, Label, Wickdiff-Output-Subfolder)
MODE_SPECS = {
    "3D": ("3D",      ("time frame data" / Path("1h data")),   "H1", "3D→H1"),
    "W":  ("W",       ("time frame data" / Path("4h data")),   "H4", "W→H4"),
    "2W": ("2Weekly", ("time frame data" / Path("daily data")), "D1", "2W→1D"),
    "M":  ("Monthly", ("time frame data" / Path("3D")),        "3D", "M→3D"),
}

# -----------------------------------
# Utils
# -----------------------------------
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

CAND_TIME  = ["time", "timestamp", "date", "datetime", "unnamed: 0"]
CAND_OPEN  = ["open", "o"]
CAND_HIGH  = ["high", "h"]
CAND_LOW   = ["low", "l"]
CAND_CLOSE = ["close", "c"]

def to_dt(s) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=False)

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
    return (p or str(s)[:6]).upper()

def normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    df = _std_cols(df)
    t = _pick_col(df, CAND_TIME)
    o = _pick_col(df, CAND_OPEN)
    h = _pick_col(df, CAND_HIGH)
    l = _pick_col(df, CAND_LOW)
    c = _pick_col(df, CAND_CLOSE)
    out = df.rename(columns={t:"time", o:"open", h:"high", l:"low", c:"close"})[["time","open","high","low","close"]].copy()

    # time parsing
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

# -----------------------------------
# Load latest pivots CSV for a TF
# -----------------------------------
def latest_pivots_csv(pivots_dir: Path) -> Optional[Path]:
    if not pivots_dir.exists():
        return None
    cands = []
    for p in pivots_dir.glob("*.csv"):
        name = p.name.lower()
        if "pivots" in name and "gap" in name:
            cands.append(p)
    if not cands:
        return None
    return max(cands, key=lambda x: x.stat().st_mtime)

def load_pivots_for_mode(base: Path, mode: str) -> pd.DataFrame:
    if mode not in MODE_SPECS:
        raise ValueError(f"Unknown mode {mode}")

    piv_subdir, _, _, _ = MODE_SPECS[mode]
    piv_dir = base / "outputs" / "pivots" / piv_subdir
    p = latest_pivots_csv(piv_dir)
    if p is None:
        print(f"❌ Keine Pivot-CSV in {piv_dir} gefunden.")
        return pd.DataFrame()

    print(f"🔄 Verwende Pivots ({mode}): {p}")
    df = pd.read_csv(p)
    df.columns = [str(c) for c in df.columns]

    # required columns (robust)
    def pick(name: str, alts: List[str]) -> str:
        low = {str(c).lower(): c for c in df.columns}
        for a in [name] + alts:
            a2 = a.lower()
            if a2 in low:
                return low[a2]
        # contains-match
        for c in df.columns:
            if name.lower() in str(c).lower():
                return c
        raise KeyError(name)

    pair_col = pick("pair", [])
    ptype_col = pick("pivot_type", [])
    gap_low_col = pick("gap_low", [])
    gap_high_col = pick("gap_high", [])
    fc_col = pick("first_candle_time", [])
    sc_col = pick("second_candle_time", [])
    ft_col = None
    for cand in ["first_touch_time"]:
        if cand.lower() in {str(c).lower() for c in df.columns}:
            ft_col = pick(cand, [])
            break

    out = pd.DataFrame({
        "pair": df[pair_col].astype(str),
        "pair6": df[pair_col].astype(str).apply(pair_code_from_str),
        "timeframe": mode,
        "pivot_type": df[ptype_col].astype(str).str.lower().str.strip(),
        "pivot_first_time": to_dt(df[fc_col]),
        "pivot_second_time": to_dt(df[sc_col]),
        "pivot_low": pd.to_numeric(df[gap_low_col], errors="coerce"),
        "pivot_high": pd.to_numeric(df[gap_high_col], errors="coerce"),
        "pivot_touch_time": (to_dt(df[ft_col]) if ft_col else pd.NaT),
    })

    out = out.dropna(subset=["pair6","pivot_type","pivot_first_time","pivot_second_time","pivot_low","pivot_high"]).reset_index(drop=True)
    out["pivot_low"], out["pivot_high"] = out[["pivot_low","pivot_high"]].min(axis=1), out[["pivot_low","pivot_high"]].max(axis=1)
    return out

# -----------------------------------
# LTF Wick-Diff detection inside a pivot (Variante 3)
# -----------------------------------
def detect_ltf_wickdiffs_inside_pivot(
    ltf: pd.DataFrame,
    direction: str,
    t_start: pd.Timestamp,
    t_end: pd.Timestamp,
    price_low: float,
    price_high: float,
) -> List[Tuple[pd.Timestamp, pd.Timestamp, float, float]]:
    """
    Findet LTF Wick-Differences (wie Pivot-Definition auf LTF):
      long:  bear -> bull  => WD-Zone = [min(low1,low2), max(low1,low2)]
      short: bull -> bear  => WD-Zone = [min(high1,high2), max(high1,high2)]
    und filtert:
      - Candle-Zeiten in [t_start, t_end]
      - WD-Zone komplett innerhalb [price_low, price_high]
    """
    if ltf.empty:
        return []

    df = ltf[(ltf["time"] >= t_start) & (ltf["time"] <= t_end)].reset_index(drop=True)
    if len(df) < 2:
        return []

    pl, ph = float(price_low), float(price_high)
    out = []

    for i in range(len(df) - 1):
        o1, h1, l1, c1, t1 = df.loc[i, ["open","high","low","close","time"]]
        o2, h2, l2, c2, t2 = df.loc[i+1, ["open","high","low","close","time"]]
        col1, col2 = candle_color(o1, c1), candle_color(o2, c2)

        if col1 == "doji" or col2 == "doji":
            continue

        if direction == "long":
            if not (col1 == "bear" and col2 == "bull"):
                continue
            wd_low = float(min(l1, l2))
            wd_high = float(max(l1, l2))
        else:
            if not (col1 == "bull" and col2 == "bear"):
                continue
            wd_low = float(min(h1, h2))
            wd_high = float(max(h1, h2))

        # vollständig innerhalb des großen Pivots (Variante 3)
        if wd_low < pl or wd_high > ph:
            continue

        out.append((pd.Timestamp(t1), pd.Timestamp(t2), wd_low, wd_high))

    return out

# -----------------------------------
# Run per mode
# -----------------------------------
def run_mode(base: Path, mode: str) -> pd.DataFrame:
    piv = load_pivots_for_mode(base, mode)
    if piv.empty:
        return pd.DataFrame()

    piv_subdir, ltf_rel, ltf_label, out_sub = MODE_SPECS[mode]
    ltf_dir = base / ltf_rel
    ltf_map = find_ltf_files_map(ltf_dir)
    if not ltf_map:
        print(f"❌ Keine LTF-Dateien in {ltf_dir} gefunden.")
        return pd.DataFrame()

    max_days = MAX_DAYS_MAP.get(mode, 14)

    rows = []
    skipped_pairs = set()

    # Sortierung: wie du es willst (3D->W->2W->M) passiert im main, hier egal.
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

        # Zeitfenster:
        # wir nutzen pivot_touch_time, sonst fallback: pivot_second_time
        t0 = r["pivot_touch_time"]
        if pd.isna(t0):
            t0 = r["pivot_second_time"]
        t0 = pd.Timestamp(t0)
        t1 = t0 + pd.Timedelta(days=max_days)

        p_low = float(r["pivot_low"])
        p_high = float(r["pivot_high"])

        # LTF wickdiffs im gesamten Pivot suchen (Variante 3)
        hits = detect_ltf_wickdiffs_inside_pivot(
            ltf=ltf_df,
            direction=direction,
            t_start=t0,
            t_end=t1,
            price_low=p_low,
            price_high=p_high,
        )

        for (wd_first, wd_second, wd_low, wd_high) in hits:
            rows.append({
                "pair": pair6,
                "timeframe": mode,
                "ltf": ltf_label,
                "pivot_type": direction,
                "first_candle_time": r["pivot_first_time"],
                "second_candle_time": r["pivot_second_time"],
                "gap_low": p_low,
                "gap_high": p_high,
                "first_touch_time": r["pivot_touch_time"],
                "wd_first_candle_time": wd_first,
                "wd_second_candle_time": wd_second,
                "wd_zone_low": wd_low,
                "wd_zone_high": wd_high,
            })

    if skipped_pairs:
        print("ℹ️ Übersprungen (fehlende/ungültige LTF-Datei):", ", ".join(sorted(skipped_pairs)))

    return pd.DataFrame(rows)

# -----------------------------------
# Main
# -----------------------------------
def parse_tf_input(s: str) -> List[str]:
    """
    Erlaubt: W, 3D, 2W, M, Both, All + Kombis mit Komma/Space/+.
    Rückgabe ist *sortiert* in der Reihenfolge: 3D -> W -> 2W -> M
    """
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

def main():
    ap = argparse.ArgumentParser(description="Step2 Wickdiff Finder (Variante 3) – LTF wickdiffs im gesamten großen Pivot.")
    ap.add_argument("--base", type=str, default=".", help="Projekt-Root (wo outputs/ und time frame data/ liegen).")
    ap.add_argument("--tf", type=str, default="All", help="3D | W | 2W | M | Both | All (oder Kombi: 'W,2W,M').")
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

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = base / "outputs" / "wickdiffs"
    out_root.mkdir(parents=True, exist_ok=True)

    # Reihenfolge: 3D -> W -> 2W -> M
    for mode in modes:
        _, _, _, out_sub = MODE_SPECS[mode]
        df = run_mode(base, mode)
        if df.empty:
            print(f"⚠️ Keine Wickdiffs gefunden für {mode}.")
            continue

        out_dir = out_root / out_sub
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"wickdiffs_{mode}_{stamp}.csv"
        df.to_csv(out_path, index=False)
        print(f"💾 Wickdiffs {mode} gespeichert in: {out_path.resolve()}")

if __name__ == "__main__":
    main()
