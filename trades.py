from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, List, Tuple, Set
import re
from datetime import datetime

import pandas as pd
from pandas.api.types import is_datetime64tz_dtype


# -----------------------------------
# Globale Einstellungen
# -----------------------------------
RR_MIN = 0.95
RR_MAX = 1.49
MIN_SL_PIPS = 50

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

# Step2 Unified Output Prefix pro HTF-Mode
MODE_TO_WD_PREFIX = {
    "3D": "3D_H1",
    "W":  "W_H4",
    "2W": "2W_D1",
    "M":  "M_3D",
}

# Step3 LTF-Daten pro Mode (wie bei dir)
MODE_SPECS = {
    "3D": ("H1", Path("time frame data") / "1h data"),
    "W":  ("H4", Path("time frame data") / "4h data"),
    "2W": ("D1", Path("time frame data") / "daily data"),
    "M":  ("3D", Path("time frame data") / "3D"),
}

# Laufzeitfenster (wie vorher)
MAX_DAYS_MAP = {"3D": 6, "W": 14, "2W": 21, "M": 42}

VARIANT_MAP = {
    "1": "INNER",
    "2": "OUTSIDE",
    "3": "ALL",
}


# -----------------------------------
# Utils
# -----------------------------------
def to_naive_datetime(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce", utc=True)
    if is_datetime64tz_dtype(dt.dtype):
        dt = dt.dt.tz_convert(None)
    return dt

def pip_size(pair: str) -> float:
    pair = pair.upper()
    return 0.01 if pair.endswith("JPY") else 0.0001

def pair_code_from_str(s: str) -> str:
    txt = str(s)

    m = re.search(r"OANDA_([A-Z]{6})", txt.upper().replace(" ", ""))
    if m:
        code = m.group(1)
        return SPECIAL_PAIR_FIX.get(code, code)

    up = re.sub(r"[^A-Z]", "", txt.upper())
    for bad, real in SPECIAL_PAIR_FIX.items():
        if bad in up:
            return real

    for p in sorted(PAIRS_28):
        if p in up:
            return p

    m2 = re.search(r"([A-Z]{6})", up)
    code = m2.group(1) if m2 else up[:6] or txt
    return SPECIAL_PAIR_FIX.get(code, code)

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

def _latest_csv_in_dir(d: Path) -> Optional[Path]:
    if not d.exists():
        return None
    cands = [p for p in d.glob("*.csv") if p.is_file()]
    if not cands:
        return None
    return max(cands, key=lambda x: x.stat().st_mtime)


# -----------------------------------
# OHLC Reader
# -----------------------------------
CAND_TIME  = ["time", "timestamp", "date", "datetime", "unnamed: 0"]
CAND_OPEN  = ["open", "o"]
CAND_HIGH  = ["high", "h"]
CAND_LOW   = ["low", "l"]
CAND_CLOSE = ["close", "c"]

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

def _normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    t = _pick_col(df, CAND_TIME)
    o = _pick_col(df, CAND_OPEN)
    h = _pick_col(df, CAND_HIGH)
    l = _pick_col(df, CAND_LOW)
    c = _pick_col(df, CAND_CLOSE)

    out = df.rename(columns={t:"time", o:"open", h:"high", l:"low", c:"close"})[["time","open","high","low","close"]].copy()

    if pd.api.types.is_numeric_dtype(out["time"]):
        vmax = pd.Series(out["time"]).astype(float).abs().max()
        unit = "ms" if vmax > 1e12 else "s"
        out["time"] = pd.to_datetime(out["time"], unit=unit, utc=False)
    else:
        out["time"] = to_naive_datetime(out["time"])

    for col in ["open","high","low","close"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["time","open","high","low","close"]).sort_values("time").reset_index(drop=True)
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
            code = pair_code_from_str(p.name).upper()
            if len(code) == 6 and code in PAIRS_28 and code not in mp:
                mp[code] = p
    return mp


# -----------------------------------
# Wickdiff Loader – sauber interlinked (INNER/OUTSIDE/ALL)
# -----------------------------------
def load_wickdiffs(mode: str, variant: str, base: Path) -> pd.DataFrame:
    """
    Liest Step2 Unified Wickdiffs aus eindeutigem Ordner:
      outputs/wickdiffs/<PREFIX>_<VARIANT>/
    und normalisiert auf Standard-Spalten für Step3.

    Erwartete Step2 Unified Spalten (robust):
      pair, pivot_type, htf_gap_low/high, htf_first/second_candle_time,
      htf_first_touch_time (optional),
      wd_first_candle_time, wd_second_candle_time,
      wd_zone_low, wd_zone_high,
      (optional) htf_wick_diff_low/high
    """
    if mode not in MODE_TO_WD_PREFIX:
        raise ValueError(f"Unknown mode: {mode}")
    prefix = MODE_TO_WD_PREFIX[mode]
    v = variant.upper().strip()
    if v not in {"INNER","OUTSIDE","ALL"}:
        raise ValueError(f"Unknown variant: {variant}")

    wd_dir = base / "outputs" / "wickdiffs" / f"{prefix}_{v}"
    path = _latest_csv_in_dir(wd_dir)

    if path is None:
        print(f"❌ Keine WickDiff-CSV gefunden in: {wd_dir}")
        return pd.DataFrame()

    print(f"🔄 Verwende WickDiffs: mode={mode} | variant={v} | file={path.name}")
    df_raw = pd.read_csv(path)
    df_raw.columns = [str(c) for c in df_raw.columns]

    def pick_any(names: List[str]) -> str:
        for n in names:
            try:
                return _pick_col_generic(df_raw, n)
            except KeyError:
                pass
        raise KeyError(f"None of these columns found: {names}")

    pair_col  = pick_any(["pair", "pair6", "symbol", "instrument"])
    ptype_col = pick_any(["pivot_type", "direction", "side"])

    # Pivot range (htf_gap_low/high sind eure neuen Standardnamen)
    low_col  = pick_any(["htf_gap_low","gap_low","pivot_low"])
    high_col = pick_any(["htf_gap_high","gap_high","pivot_high"])

    # Pivot times
    first_candle_col  = pick_any(["htf_first_candle_time","pivot_first_time","first_candle_time"])
    second_candle_col = pick_any(["htf_second_candle_time","pivot_second_time","second_candle_time"])

    # Pivot touch (FirstTouch) – kann NaT sein
    touch_col = None
    for cand in ["htf_first_touch_time","pivot_touch_time","first_touch_time"]:
        try:
            touch_col = _pick_col_generic(df_raw, cand)
            break
        except KeyError:
            continue

    # WD times + Zone
    wd_first_col  = pick_any(["wd_first_candle_time","wd_first_time","wd_first"])
    wd_second_col = pick_any(["wd_second_candle_time","wd_second_time","wd_second"])
    wd_low_col    = pick_any(["wd_zone_low","wd_low","wick_diff_low","wdiff_low"])
    wd_high_col   = pick_any(["wd_zone_high","wd_high","wick_diff_high","wdiff_high"])

    # Optional HTF wickdiff bounds (für OUTSIDE debug – kann leer sein)
    htf_wd_low_col = None
    htf_wd_high_col = None
    for cand in ["htf_wick_diff_low","wick_diff_low"]:
        try:
            htf_wd_low_col = _pick_col_generic(df_raw, cand)
            break
        except KeyError:
            continue
    for cand in ["htf_wick_diff_high","wick_diff_high"]:
        try:
            htf_wd_high_col = _pick_col_generic(df_raw, cand)
            break
        except KeyError:
            continue

    df = pd.DataFrame({
        "wd_variant": v,
        "pair_raw": df_raw[pair_col].astype(str),
        "pivot_type": df_raw[ptype_col].astype(str).str.lower().str.strip(),

        "pivot_low": pd.to_numeric(df_raw[low_col], errors="coerce"),
        "pivot_high": pd.to_numeric(df_raw[high_col], errors="coerce"),

        "pivot_first_time": to_naive_datetime(df_raw[first_candle_col]),
        "pivot_second_time": to_naive_datetime(df_raw[second_candle_col]),
        "pivot_touch_time": (to_naive_datetime(df_raw[touch_col]) if touch_col else pd.NaT),

        "wd_first_time": to_naive_datetime(df_raw[wd_first_col]),
        "wd_second_time": to_naive_datetime(df_raw[wd_second_col]),
        "wd_low": pd.to_numeric(df_raw[wd_low_col], errors="coerce"),
        "wd_high": pd.to_numeric(df_raw[wd_high_col], errors="coerce"),

        "htf_wick_diff_low": (pd.to_numeric(df_raw[htf_wd_low_col], errors="coerce") if htf_wd_low_col else pd.NA),
        "htf_wick_diff_high": (pd.to_numeric(df_raw[htf_wd_high_col], errors="coerce") if htf_wd_high_col else pd.NA),
    })

    df["pair6"] = df["pair_raw"].apply(pair_code_from_str).str.upper()

    # normalize ranges
    df["pivot_low"], df["pivot_high"] = df[["pivot_low","pivot_high"]].min(axis=1), df[["pivot_low","pivot_high"]].max(axis=1)
    df["wd_low"], df["wd_high"] = df[["wd_low","wd_high"]].min(axis=1), df[["wd_low","wd_high"]].max(axis=1)

    # keep only valid
    df = df.dropna(subset=[
        "pair6","pivot_type",
        "pivot_low","pivot_high","pivot_first_time","pivot_second_time",
        "wd_low","wd_high","wd_first_time","wd_second_time",
    ]).reset_index(drop=True)

    return df


# -----------------------------------
# Pivot-TP-Invalidation
# -----------------------------------
def compute_pivot_tp_level(pivot_low: float, pivot_high: float, direction: str) -> Optional[float]:
    pivot_low = float(pivot_low)
    pivot_high = float(pivot_high)
    if pivot_high <= pivot_low:
        return None
    rng = pivot_high - pivot_low
    return (pivot_high + rng) if direction == "long" else (pivot_low - rng)

def pivot_invalidated_by_tp_before_wd(
    ltf: pd.DataFrame,
    pivot_touch_time: pd.Timestamp,
    end_time: Optional[pd.Timestamp],
    zone_low: float,
    zone_high: float,
    direction: str,
    pivot_low: float,
    pivot_high: float,
) -> bool:
    if pd.isna(pivot_touch_time):
        return False

    tp_level = compute_pivot_tp_level(pivot_low, pivot_high, direction)
    if tp_level is None:
        return False

    df = ltf[ltf["time"] >= pivot_touch_time]
    if end_time is not None:
        df = df[df["time"] <= end_time]
    if df.empty:
        return False

    lo, hi = float(min(zone_low, zone_high)), float(max(zone_low, zone_high))

    for _, row in df.iterrows():
        h = float(row["high"])
        l = float(row["low"])

        touched_wd = (h >= lo) and (l <= hi)

        if direction == "long":
            hit_tp = h >= tp_level
        else:
            hit_tp = l <= tp_level

        if touched_wd:
            return False
        if hit_tp:
            return True

    return False


# -----------------------------------
# Entry / TP SL / Simulation
# -----------------------------------
def find_entry_candle(
    ltf: pd.DataFrame,
    zone_low: float,
    zone_high: float,
    direction: str,
    start_time: pd.Timestamp,
    end_time: Optional[pd.Timestamp] = None,
) -> Optional[int]:
    if pd.isna(start_time):
        return None

    df = ltf[ltf["time"] >= start_time]
    if end_time is not None:
        df = df[df["time"] <= end_time]
    df = df.reset_index()
    if df.empty:
        return None

    lo, hi = float(min(zone_low, zone_high)), float(max(zone_low, zone_high))

    for _, row in df.iterrows():
        idx_ltf = int(row["index"])
        o = float(row["open"])
        h = float(row["high"])
        l = float(row["low"])
        c = float(row["close"])

        # Kill-Kerze -> killt nur diese WD
        if direction == "long":
            if (o > hi and c < lo) or (c > hi and o < lo):
                return None
        else:
            if (o < lo and c > hi) or (c < lo and o > hi):
                return None

        touched = (h >= lo) and (l <= hi)
        body_low = min(o, c)
        body_high = max(o, c)

        if direction == "long":
            if touched and (c > hi) and (body_low >= hi):
                return idx_ltf
        else:
            if touched and (c < lo) and (body_high <= lo):
                return idx_ltf

    return None

def compute_tp_sl(
    entry_price: float,
    pivot_low: float,
    pivot_high: float,
    direction: str,
    pair6: str,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    pivot_low = float(pivot_low)
    pivot_high = float(pivot_high)
    if pivot_high <= pivot_low:
        return None, None, None

    rng = pivot_high - pivot_low
    if direction == "long":
        tp_price = pivot_high + rng
        baseline_sl = pivot_low - 0.1 * rng
        sl_dist_baseline = entry_price - baseline_sl
        tp_dist = tp_price - entry_price
    else:
        tp_price = pivot_low - rng
        baseline_sl = pivot_high + 0.1 * rng
        sl_dist_baseline = baseline_sl - entry_price
        tp_dist = entry_price - tp_price

    if sl_dist_baseline <= 0 or tp_dist <= 0:
        return None, None, None

    pip = pip_size(pair6)
    min_sl_dist = MIN_SL_PIPS * pip

    sl_dist = max(sl_dist_baseline, min_sl_dist, tp_dist / RR_MAX)
    rr = tp_dist / sl_dist
    if rr < RR_MIN or rr > RR_MAX:
        return None, None, None

    sl_price = (entry_price - sl_dist) if direction == "long" else (entry_price + sl_dist)
    return tp_price, sl_price, rr

def simulate_trade(
    ltf: pd.DataFrame,
    entry_idx: int,
    direction: str,
    entry_price: float,
    tp_price: float,
    sl_price: float,
    rr_pos: float,
) -> Optional[dict]:
    for j in range(entry_idx + 1, len(ltf)):
        row = ltf.iloc[j]
        h = float(row["high"])
        l = float(row["low"])
        t = row["time"]

        if direction == "long":
            hit_sl = l <= sl_price
            hit_tp = h >= tp_price
        else:
            hit_sl = h >= sl_price
            hit_tp = l <= tp_price

        # konservativ: SL gewinnt, wenn beides
        if hit_sl and hit_tp:
            return {"entry_time": ltf.iloc[entry_idx]["time"], "entry_price": entry_price,
                    "exit_time": t, "exit_price": sl_price, "result": "loss", "rr_signed": -1.0}
        if hit_sl:
            return {"entry_time": ltf.iloc[entry_idx]["time"], "entry_price": entry_price,
                    "exit_time": t, "exit_price": sl_price, "result": "loss", "rr_signed": -1.0}
        if hit_tp:
            return {"entry_time": ltf.iloc[entry_idx]["time"], "entry_price": entry_price,
                    "exit_time": t, "exit_price": tp_price, "result": "win", "rr_signed": float(rr_pos)}

    return None


# -----------------------------------
# Run-Mode – Pivot deaktiviert nach 1 Entry
# -----------------------------------
def run_mode(
    base: Path,
    mode: str,
    variant: str,
    ltf_dir: Path,
    ltf_label: str,
    pair_filter: Optional[Set[str]] = None,
    pivot_start: Optional[pd.Timestamp] = None,
    pivot_end: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    wd = load_wickdiffs(mode=mode, variant=variant, base=base)
    if wd.empty:
        return pd.DataFrame()

    if pair_filter is not None:
        wd = wd[wd["pair6"].isin({p.upper() for p in pair_filter})].copy()

    if pivot_start is not None:
        wd = wd[wd["pivot_first_time"] >= pivot_start].copy()
    if pivot_end is not None:
        wd = wd[wd["pivot_first_time"] < pivot_end].copy()

    if wd.empty:
        print(f"⚠️ Keine Wick-Pivots nach Filter für Modus {mode} | {variant}.")
        return pd.DataFrame()

    ltf_map = find_ltf_files_map(ltf_dir)
    if not ltf_map:
        print(f"❌ Keine LTF-Dateien in {ltf_dir} gefunden.")
        return pd.DataFrame()

    max_days = MAX_DAYS_MAP.get(mode, 14)

    trades: List[dict] = []
    skipped_pairs: Set[str] = set()

    wd["pivot_type"] = wd["pivot_type"].astype(str).str.lower().str.strip()
    wd = wd[wd["pivot_type"].isin({"long","short"})].copy()
    wd = wd.sort_values(["pair6","pivot_type","pivot_first_time","pivot_second_time","wd_first_time"]).reset_index(drop=True)

    pivot_groups = wd.groupby(["pair6","pivot_type","pivot_first_time","pivot_second_time"], sort=False)

    for (pair6, direction, p1, p2), g in pivot_groups:
        pair6 = str(pair6).upper()
        direction = str(direction).lower()

        ltf_path = ltf_map.get(pair6)
        if not ltf_path:
            skipped_pairs.add(pair6)
            continue

        ltf_df = read_ohlc_file(ltf_path)
        if ltf_df is None or ltf_df.empty:
            skipped_pairs.add(pair6)
            continue

        pivot_touch = g["pivot_touch_time"].dropna()
        if pivot_touch.empty:
            # First-Touch-Variante: ohne Touch => keine Trades
            continue
        pivot_touch_time = pd.Timestamp(pivot_touch.iloc[0])
        end_time = pivot_touch_time + pd.Timedelta(days=max_days)

        pivot_low = float(g["pivot_low"].iloc[0])
        pivot_high = float(g["pivot_high"].iloc[0])

        pivot_traded = False

        for _, row in g.iterrows():
            if pivot_traded:
                break

            zone_low = float(row["wd_low"])
            zone_high = float(row["wd_high"])

            # invalidiert nur diese WD
            if pivot_invalidated_by_tp_before_wd(
                ltf=ltf_df,
                pivot_touch_time=pivot_touch_time,
                end_time=end_time,
                zone_low=zone_low,
                zone_high=zone_high,
                direction=direction,
                pivot_low=pivot_low,
                pivot_high=pivot_high,
            ):
                continue

            entry_idx = find_entry_candle(
                ltf=ltf_df,
                zone_low=zone_low,
                zone_high=zone_high,
                direction=direction,
                start_time=pivot_touch_time,
                end_time=end_time,
            )
            if entry_idx is None:
                continue

            entry_price = float(ltf_df.loc[entry_idx, "close"])
            tp_price, sl_price, rr_pos = compute_tp_sl(entry_price, pivot_low, pivot_high, direction, pair6)
            if tp_price is None:
                continue

            sim = simulate_trade(ltf_df, entry_idx, direction, entry_price, tp_price, sl_price, rr_pos)
            if sim is None:
                continue

            trades.append({
                "wd_variant": row.get("wd_variant", variant.upper()),
                "mode": mode,
                "ltf": ltf_label,
                "pair": pair6,
                "direction": direction,
                "pivot_first_time": p1,
                "pivot_second_time": p2,
                "pivot_low": pivot_low,
                "pivot_high": pivot_high,
                "pivot_range": float(pivot_high - pivot_low),
                "wd_first_time": row["wd_first_time"],
                "wd_second_time": row["wd_second_time"],
                "wd_low": zone_low,
                "wd_high": zone_high,
                "entry_time": sim["entry_time"],
                "entry_price": sim["entry_price"],
                "tp_price": tp_price,
                "sl_price": sl_price,
                "exit_time": sim["exit_time"],
                "exit_price": sim["exit_price"],
                "result": sim["result"],
                "rr_signed": sim["rr_signed"],
            })

            # Regel: Entry = Pivot deaktiviert
            pivot_traded = True

    if skipped_pairs:
        print("ℹ️ Übersprungen (fehlende/ungültige LTF-Datei):", ", ".join(sorted(skipped_pairs)))

    return pd.DataFrame(trades)


# -----------------------------------
# Summary
# -----------------------------------
def summarize_trades(df_trades: pd.DataFrame, mode: str, variant: str, out_dir: Path, stamp: str) -> None:
    if df_trades.empty:
        print(f"⚠️ Keine Trades für Modus {mode} | {variant}.")
        return

    df = df_trades.copy()
    df["is_win"] = df["result"] == "win"

    rows = []
    for pair, g in df.groupby("pair"):
        n = len(g)
        wins = int(g["is_win"].sum())
        losses = int(n - wins)
        avg_rr_all = float(g["rr_signed"].mean())
        win_rate = float(wins / n * 100.0) if n else 0.0
        avg_rr_win = float(g.loc[g["is_win"], "rr_signed"].mean()) if wins else float("nan")
        avg_rr_loss = float(g.loc[~g["is_win"], "rr_signed"].mean()) if losses else float("nan")
        rows.append({
            "pair": pair, "n_trades": n, "wins": wins, "losses": losses,
            "avg_rr": avg_rr_all, "win_rate_%": win_rate,
            "avg_rr_win": avg_rr_win, "avg_rr_loss": avg_rr_loss,
        })

    summary = pd.DataFrame(rows).sort_values("pair").reset_index(drop=True)

    n_tot = len(df)
    wins_tot = int(df["is_win"].sum())
    losses_tot = int(n_tot - wins_tot)
    win_rate_tot = float(wins_tot / n_tot * 100.0) if n_tot else 0.0
    avg_rr_tot = float(df["rr_signed"].mean()) if n_tot else float("nan")
    avg_rr_win_tot = float(df.loc[df["is_win"], "rr_signed"].mean()) if wins_tot else float("nan")
    avg_rr_loss_tot = float(df.loc[~df["is_win"], "rr_signed"].mean()) if losses_tot else float("nan")

    summary = pd.concat([summary, pd.DataFrame([{
        "pair": "ALL", "n_trades": n_tot, "wins": wins_tot, "losses": losses_tot,
        "avg_rr": avg_rr_tot, "win_rate_%": win_rate_tot,
        "avg_rr_win": avg_rr_win_tot, "avg_rr_loss": avg_rr_loss_tot
    }])], ignore_index=True)

    summary_path = out_dir / f"trades_{mode}_summary_{variant}_{stamp}.csv"
    summary.to_csv(summary_path, index=False)

    print(f"\n===== Summary Trades: {mode} | {variant} =====")
    print(f"Gesamt: {n_tot} | Wins: {wins_tot} | Losses: {losses_tot} | Win-Rate: {win_rate_tot:.1f}%")
    print(f"Ø RR Win: {avg_rr_win_tot:.3f} | Ø RR Loss: {avg_rr_loss_tot:.3f} | EV: {avg_rr_tot:.3f}")
    print(f"💾 {summary_path.resolve()}")


# -----------------------------------
# CLI Prompts
# -----------------------------------
def prompt_variant() -> str:
    print("\nWelche Variante?")
    print("  1 = INNER")
    print("  2 = OUTSIDE")
    print("  3 = ALL")
    print("  4 = ALLE 3 (schreibt 3 Outputs)")
    v = input("Eingabe (1/2/3/4): ").strip()
    if v not in {"1","2","3","4"}:
        raise ValueError("Ungültig. Nur 1/2/3/4.")
    return v

def prompt_modes() -> List[str]:
    print("\nWelche Timeframes/Modes? (Multi-Auswahl möglich)")
    print("  1 = 3D")
    print("  2 = W")
    print("  3 = 2W")
    print("  4 = M")
    print("  5 = ALL (3D,W,2W,M)")
    s = input("Eingabe (z.B. 1,2 oder 13 oder all): ").strip().lower()

    if not s:
        raise ValueError("Keine Eingabe.")

    # Allow "all" or "5"
    if s in {"5", "all"}:
        return ["3D", "W", "2W", "M"]

    # Normalize separators -> tokens
    # Accept "1,2", "1 2", "1;2", "1/2"
    for sep in [",", ";", "/", "|"]:
        s = s.replace(sep, " ")
    s = s.replace("\t", " ")

    tokens = [t for t in s.split(" ") if t]

    # If user typed "12" or "134" etc. (no spaces/commas), treat as digits
    if len(tokens) == 1 and tokens[0].isdigit() and len(tokens[0]) > 1:
        tokens = list(tokens[0])

    mapping = {"1": "3D", "2": "W", "3": "2W", "4": "M"}

    modes = []
    for t in tokens:
        if t in {"5", "all"}:
            return ["3D", "W", "2W", "M"]
        if t not in mapping:
            raise ValueError(f"Ungültige Auswahl: '{t}'. Erlaubt: 1,2,3,4 oder 5/all.")
        modes.append(mapping[t])

    # De-dup preserving order
    seen = set()
    out = []
    for m in modes:
        if m not in seen:
            seen.add(m)
            out.append(m)

    return out



# -----------------------------------
# Main – Trades Outputs sauber pro Variante getrennt
# -----------------------------------
def main() -> None:
    base = Path(__file__).resolve().parent

    try:
        vsel = prompt_variant()
        modes = prompt_modes()
    except Exception as e:
        print(f"❌ {e}")
        return

    variants = [VARIANT_MAP[vsel]] if vsel in {"1","2","3"} else ["INNER","OUTSIDE","ALL"]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for variant in variants:
        out_base = base / "outputs" / "trades" / variant
        out_base.mkdir(parents=True, exist_ok=True)

        for mode in modes:
            if mode not in MODE_SPECS:
                continue

            ltf_label, ltf_rel = MODE_SPECS[mode]
            ltf_dir = base / ltf_rel

            trades = run_mode(
                base=base,
                mode=mode,
                variant=variant,
                ltf_dir=ltf_dir,
                ltf_label=ltf_label,
            )
            if trades.empty:
                continue

            path = out_base / f"trades_{mode}_{variant}_{stamp}.csv"
            trades.to_csv(path, index=False)
            print(f"\n💾 Trades gespeichert: {path.resolve()}")

            summarize_trades(trades, mode, variant, out_base, stamp)

if __name__ == "__main__":
    main()
