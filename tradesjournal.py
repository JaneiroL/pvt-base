from __future__ import annotations

from pathlib import Path
import re
from typing import Tuple, Optional, Dict, List

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64tz_dtype

# -------------------------
# Config
# -------------------------
MIN_RR = 0.95
MAX_RR = 1.49
MIN_SL_PIPS = 50           # absolute Pips, umgerechnet per pip_size()
MAX_DAYS_AFTER_PIVOT = 14  # Kalender-Tage nach Pivot-Touch, in denen ein Entry kommen darf
PREVIEW_ROWS = 30

# Set der 28 Paare
PAIRS_28 = {
    "AUDCAD","AUDCHF","AUDJPY","AUDNZD","AUDUSD",
    "CADCHF","CADJPY",
    "CHFJPY",
    "EURAUD","EURCAD","EURCHF","EURGBP","EURJPY","EURNZD","EURUSD",
    "GBPAUD","GBPCAD","GBPCHF","GBPJPY","GBPUSD","GBPNZD",
    "NZDCAD","NZDCHF","NZDJPY","NZDUSD",
    "USDCAD","USDCHF","USDJPY",
}

# Falls Dateinamen komische Kürzel haben
SPECIAL_PAIR_FIX = {
    "OANDAG": "GBPNZD",
}

# -------------------------
# Helpers: Zeit / OHLC
# -------------------------
def to_naive_datetime(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce", utc=True)
    if is_datetime64tz_dtype(dt.dtype):
        dt = dt.dt.tz_convert(None)
    return dt

CAND_TIME  = ["time", "timestamp", "date", "datetime", "unnamed: 0"]
CAND_OPEN  = ["open", "o"]
CAND_HIGH  = ["high", "h"]
CAND_LOW   = ["low", "l"]
CAND_CLOSE = ["close", "c"]

def _pick_col(df: pd.DataFrame, cands) -> str:
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

def _normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
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
        out["time"] = to_naive_datetime(out["time"])

    for col in ["open","high","low","close"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = (
        out.dropna(subset=["time","open","high","low","close"])
           .sort_values("time")
           .reset_index(drop=True)
    )
    return out

def read_ohlc_file(path: Path) -> Optional[pd.DataFrame]:
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
                continue
    except Exception:
        return None
    return None

# -------------------------
# Generic column finder
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

# -------------------------
# Pair helpers
# -------------------------
def pair_code_from_str(s: str) -> str:
    txt = str(s)
    m = re.search(r"OANDA_([A-Z]{6})", txt.upper().replace(" ", ""))
    if m:
        code = m.group(1)
    else:
        up = re.sub(r"[^A-Z]", "", txt.upper())
        for bad, real in SPECIAL_PAIR_FIX.items():
            if bad in up:
                return real
        for p in PAIRS_28:
            if p in up:
                return p
        m2 = re.search(r"([A-Z]{6})", up)
        code = m2.group(1) if m2 else up[:6] or txt
    return SPECIAL_PAIR_FIX.get(code, code)

def find_ltf_files_map(ltf_dir: Path) -> Dict[str, Path]:
    mp: Dict[str, Path] = {}
    if not ltf_dir.exists():
        return mp
    for p in ltf_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".csv",".xlsx"}:
            code = pair_code_from_str(p.name)
            if len(code) == 6 and code.upper() in PAIRS_28 and code not in mp:
                mp[code] = p
    return mp

def pip_size_for_pair(pair: str) -> float:
    pair = pair.upper()
    if pair.endswith("JPY"):
        return 0.01
    return 0.0001

# -------------------------
# Wick-diff Loader
# -------------------------
def extract_pair6(x: str) -> str:
    up = re.sub(r"[^A-Z]", "", str(x).upper())
    for bad, real in SPECIAL_PAIR_FIX.items():
        if bad in up:
            return real
    for p in PAIRS_28:
        if p in up:
            return p
    m = re.search(r"([A-Z]{6})", up)
    return m.group(1) if m else up[:6] or str(x)

def load_wickdiffs(path: Path, mode: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if mode == "W":
        prefix = "weekly"
        pending_guess = "pending_until_weekly_touch"
    else:
        prefix = "3day"
        pending_guess = "pending_until_3day_touch"

    out = pd.DataFrame({
        "pair_raw": df[_pick_col_generic(df, "pair")].astype(str),
        "pivot_type": df[_pick_col_generic(df, "pivot_type")].astype(str).str.lower(),
        "ltf_label": df[_pick_col_generic(df, "ltf")].astype(str),
        "pivot_first_time": to_naive_datetime(
            df[_pick_col_generic(df, f"{prefix}_first_candle_time")]
        ),
        "pivot_second_time": to_naive_datetime(
            df[_pick_col_generic(df, f"{prefix}_second_candle_time")]
        ),
        "pivot_low": pd.to_numeric(df[_pick_col_generic(df, f"{prefix}_gap_low")],
                                   errors="coerce"),
        "pivot_high": pd.to_numeric(df[_pick_col_generic(df, f"{prefix}_gap_high")],
                                    errors="coerce"),
        "pivot_width": pd.to_numeric(df[_pick_col_generic(df, f"{prefix}_gap_width")],
                                     errors="coerce"),
        "pivot_first_touch_time": to_naive_datetime(
            df[_pick_col_generic(df, f"{prefix}_first_touch_time")]
        ),
        "zone_first_time": to_naive_datetime(df[_pick_col_generic(df, "wd_first_candle_time")]),
        "zone_second_time": to_naive_datetime(df[_pick_col_generic(df, "wd_second_candle_time")]),
        "zone_low": pd.to_numeric(df[_pick_col_generic(df, "wd_zone_low")], errors="coerce"),
        "zone_high": pd.to_numeric(df[_pick_col_generic(df, "wd_zone_high")], errors="coerce"),
        "zone_width": pd.to_numeric(df[_pick_col_generic(df, "wd_zone_width")], errors="coerce"),
        "zone_pct_of_pivot": pd.to_numeric(
            df[_pick_col_generic(df, "wd_zone_pct_of_weekly")], errors="coerce"
        ),
    })

    # pending-Flag (kann bei älteren Dateien fehlen)
    try:
        pending_col = _pick_col_generic(df, pending_guess)
        out["pending"] = df[pending_col].astype(bool)
    except KeyError:
        out["pending"] = False

    # Pivot-High/Low sauber sortieren
    lo = out[["pivot_low","pivot_high"]].min(axis=1)
    hi = out[["pivot_low","pivot_high"]].max(axis=1)
    out["pivot_low"], out["pivot_high"] = lo, hi
    out["pivot_width"] = out["pivot_high"] - out["pivot_low"]

    zlo = out[["zone_low","zone_high"]].min(axis=1)
    zhi = out[["zone_low","zone_high"]].max(axis=1)
    out["zone_low"], out["zone_high"] = zlo, zhi

    out["pair6"] = out["pair_raw"].apply(extract_pair6)

    out = out.dropna(
        subset=[
            "pivot_first_time","pivot_second_time",
            "pivot_low","pivot_high","pivot_width",
            "zone_low","zone_high","zone_width"
        ]
    ).reset_index(drop=True)

    return out

# -------------------------
# Entry Detection in LTF
# -------------------------
def find_entry_for_zone(
    ltf: pd.DataFrame,
    side: str,
    zone_low: float,
    zone_high: float,
    pivot_touch_time: pd.Timestamp,
    max_days: int
) -> Tuple[Optional[pd.Timestamp], Optional[float], Optional[str]]:
    """Suche erste gültige Entry-Kerze oder liefere Reason zurück."""
    if pd.isna(pivot_touch_time):
        return None, None, "no_pivot_touch"

    start = pd.Timestamp(pivot_touch_time)
    end = start + pd.Timedelta(days=max_days)

    seg = ltf[(ltf["time"] >= start) & (ltf["time"] <= end)].reset_index(drop=True)
    if seg.empty:
        return None, None, "no_ltf_data_in_window"

    for _, row in seg.iterrows():
        o = float(row["open"])
        h = float(row["high"])
        l = float(row["low"])
        c = float(row["close"])

        # Body-through -> Zone sofort tot
        if side == "long":
            body_kill = (o > zone_high and c < zone_low)
        else:  # short
            body_kill = (o < zone_low and c > zone_high)
        if body_kill:
            return None, None, "body_through_zone"

        touched = (h >= zone_low) and (l <= zone_high)

        if side == "long":
            entry_ok = touched and (c > zone_high)
        else:
            entry_ok = touched and (c < zone_low)

        if entry_ok:
            return pd.Timestamp(row["time"]), c, None

    return None, None, "no_valid_touch_within_window"

# -------------------------
# SL / TP / RR
# -------------------------
def compute_sl_tp_rr(
    side: str,
    pair: str,
    pivot_low: float,
    pivot_high: float,
    entry_price: float,
) -> Tuple[bool, str, Optional[float], Optional[float], Optional[float], Optional[float]]:
    lo, hi = float(pivot_low), float(pivot_high)
    if not np.isfinite(lo) or not np.isfinite(hi):
        return False, "invalid_pivot_prices", None, None, None, None
    if hi <= lo:
        return False, "pivot_width_non_positive", None, None, None, None

    width = hi - lo
    pip = pip_size_for_pair(pair)
    min_sl_dist = MIN_SL_PIPS * pip

    if side == "long":
        tp_level = hi + width                 # Spiegelung über dem Pivot
        base_sl_level = lo - 0.10 * width     # 110%-Level
        tp_dist = tp_level - entry_price
        base_sl_dist = entry_price - base_sl_level
    else:
        tp_level = lo - width
        base_sl_level = hi + 0.10 * width
        tp_dist = entry_price - tp_level
        base_sl_dist = base_sl_level - entry_price

    if tp_dist <= 0:
        return False, "entry_past_tp", None, None, None, None
    if base_sl_dist <= 0:
        return False, "entry_past_base_sl", None, None, None, None

    rr_base = tp_dist / base_sl_dist
    if rr_base < MIN_RR:
        return False, "rr_below_min_at_base_sl", None, None, None, None

    # minimale Distanz, die wir haben dürfen (Base-SL + min. 50 Pips)
    min_dist_possible = max(base_sl_dist, min_sl_dist)
    rr_min_dist = tp_dist / min_dist_possible
    if rr_min_dist < MIN_RR:
        return False, "rr_below_min_after_min_sl", None, None, None, None

    # wir brauchen Distanz >= min_dist_possible UND RR <= MAX_RR
    lower_bound = max(min_dist_possible, tp_dist / MAX_RR)
    upper_bound = tp_dist / MIN_RR  # max. Distanz, bevor RR < MIN_RR

    if lower_bound > upper_bound:
        return False, "no_distance_satisfies_rr_bounds", None, None, None, None

    final_dist = lower_bound
    final_rr = tp_dist / final_dist

    if side == "long":
        sl_level = entry_price - final_dist
    else:
        sl_level = entry_price + final_dist

    return True, "", sl_level, tp_level, final_dist, final_rr

# -------------------------
# Run-Logik pro Mode
# -------------------------
def run_mode(
    base: Path,
    mode: str,
    wick_dir: Path,
    pattern: str,
    ltf_dir: Path,
    out_label: str,
) -> None:
    if not wick_dir.exists():
        print(f"❌ Wickdiff-Ordner nicht gefunden: {wick_dir}")
        return
    files = sorted(wick_dir.glob(pattern))
    if not files:
        print(f"❌ Keine wick_diffs-CSV in {wick_dir} gefunden.")
        return
    wick_path = files[-1]
    print(f"🔄 Verwende Wick-Diffs ({mode}): {wick_path}")

    wd = load_wickdiffs(wick_path, mode)
    # nur Pivots mit Touch und nicht pending
    wd = wd[(~wd["pending"]) & wd["pivot_first_touch_time"].notna()].reset_index(drop=True)
    if wd.empty:
        print(f"⚠️ Keine reifen Wick-Differences mit Pivot-Touch gefunden für {mode}.")
        return

    ltf_map = find_ltf_files_map(ltf_dir)
    if not ltf_map:
        print(f"❌ Keine LTF-Dateien in {ltf_dir} gefunden.")
        return

    all_trades: List[dict] = []
    rejections: List[dict] = []
    used_pivots = set()

    for _, row in wd.iterrows():
        pair = row["pair6"]
        side = row["pivot_type"].lower()
        if side not in {"long","short"}:
            continue

        pivot_key = (pair, side, row["pivot_first_time"], row["pivot_second_time"])
        if pivot_key in used_pivots:
            rejections.append({
                "pair": pair,
                "mode": mode,
                "reason": "pivot_already_traded",
            })
            continue

        ltf_path = ltf_map.get(pair)
        if not ltf_path:
            rejections.append({
                "pair": pair,
                "mode": mode,
                "reason": "no_ltf_file",
            })
            continue

        ltf_df = read_ohlc_file(ltf_path)
        if ltf_df is None or ltf_df.empty:
            rejections.append({
                "pair": pair,
                "mode": mode,
                "reason": "ltf_empty_or_unreadable",
            })
            continue

        entry_time, entry_price, entry_fail = find_entry_for_zone(
            ltf=ltf_df,
            side=side,
            zone_low=float(row["zone_low"]),
            zone_high=float(row["zone_high"]),
            pivot_touch_time=row["pivot_first_touch_time"],
            max_days=MAX_DAYS_AFTER_PIVOT,
        )

        if entry_fail is not None:
            rejections.append({
                "pair": pair,
                "mode": mode,
                "reason": entry_fail,
            })
            continue

        ok, rr_reason, sl_level, tp_level, sl_dist, rr = compute_sl_tp_rr(
            side=side,
            pair=pair,
            pivot_low=float(row["pivot_low"]),
            pivot_high=float(row["pivot_high"]),
            entry_price=float(entry_price),
        )
        if not ok:
            rejections.append({
                "pair": pair,
                "mode": mode,
                "reason": rr_reason,
            })
            continue

        trade = {
            "pair": pair,
            "mode": mode,
            "ltf": row["ltf_label"],
            "pivot_type": side,
            "pivot_first_time": row["pivot_first_time"],
            "pivot_second_time": row["pivot_second_time"],
            "pivot_low": row["pivot_low"],
            "pivot_high": row["pivot_high"],
            "pivot_width": row["pivot_width"],
            "pivot_first_touch_time": row["pivot_first_touch_time"],
            "zone_first_time": row["zone_first_time"],
            "zone_second_time": row["zone_second_time"],
            "zone_low": row["zone_low"],
            "zone_high": row["zone_high"],
            "zone_width": row["zone_width"],
            "zone_pct_of_pivot": row["zone_pct_of_pivot"],
            "entry_time": entry_time,
            "entry_price": entry_price,
            "sl_level": sl_level,
            "tp_level": tp_level,
            "sl_distance": sl_dist,
            "tp_distance": (tp_level - entry_price) if side == "long" else (entry_price - tp_level),
            "rr": rr,
        }
        all_trades.append(trade)
        used_pivots.add(pivot_key)

    out_dir = base / "outputs" / "trades"
    out_dir.mkdir(parents=True, exist_ok=True)

    if all_trades:
        out_df = pd.DataFrame(all_trades)
        stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"trades_{out_label}_{stamp}.csv"
        out_df.to_csv(out_path, index=False)
        print(f"✅ {len(out_df)} Trades gespeichert in: {out_path.resolve()}")
        with pd.option_context("display.width", 220, "display.max_columns", None):
            print(out_df.head(PREVIEW_ROWS).to_string(index=False))
    else:
        print(f"⚠️ Keine gültigen Trades gefunden für {mode}.")

    if rejections:
        rej_df = pd.DataFrame(rejections)
        stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        rej_path = out_dir / f"trade_rejections_{out_label}_{stamp}.csv"
        rej_df.to_csv(rej_path, index=False)
        print(f"ℹ️ {len(rejections)} Setups verworfen (Details: {rej_path.name})")

# -------------------------
# main
# -------------------------
def main() -> None:
    base = Path(__file__).resolve().parent

    # Weekly-Pivots → H4
    run_mode(
        base=base,
        mode="W",
        wick_dir=base / "outputs" / "wickdiffs" / "W\u2192H4",
        pattern="wick_diffs_H4_*.csv",
        ltf_dir=base / "time frame data" / "4h data",
        out_label="W_H4",
    )

    # 3D-Pivots → H1
    run_mode(
        base=base,
        mode="3D",
        wick_dir=base / "outputs" / "wickdiffs" / "3D\u2192H1",
        pattern="wick_diffs_H1_*.csv",
        ltf_dir=base / "time frame data" / "1h data",
        out_label="3D_H1",
    )

if __name__ == "__main__":
    main()
