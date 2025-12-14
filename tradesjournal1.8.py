from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, List, Tuple, Set
import re

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64tz_dtype

# -----------------------------------
# Globale Einstellungen
# -----------------------------------
RR_MIN = 0.95
RR_MAX = 1.49
MIN_SL_PIPS = 50

# Default-"Timer" nach Pivot-Touch bis Entry-Logik stoppt
# (dein aktueller Stand: 3D=6, W=14; neu: 2W=21, M=42)
MAX_DAYS_BY_MODE = {
    "3D": 6,
    "W": 14,
    "2W": 21,
    "M": 42,
}

# 28er-Universum (nur FX, inkl. GBPNZD)
PAIRS_28 = {
    "AUDCAD","AUDCHF","AUDJPY","AUDNZD","AUDUSD",
    "CADCHF","CADJPY",
    "CHFJPY",
    "EURAUD","EURCAD","EURCHF","EURGBP","EURJPY","EURNZD","EURUSD",
    "GBPAUD","GBPCAD","GBPCHF","GBPJPY","GBPUSD","GBPNZD",
    "NZDCAD","NZDCHF","NZDJPY","NZDUSD",
    "USDCAD","USDCHF","USDJPY",
}

# Sonderfall: alte kaputte Codes (falls sie auftauchen)
SPECIAL_PAIR_FIX = {
    "OANDAG": "GBPNZD",
}

# -----------------------------------
# Hilfsfunktionen
# -----------------------------------
def to_naive_datetime(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce", utc=True)
    if is_datetime64tz_dtype(dt.dtype):
        dt = dt.dt.tz_convert(None)
    return dt


def pip_size(pair: str) -> float:
    pair = pair.upper()
    # JPY-Pairs: 0.01 statt 0.0001
    if pair.endswith("JPY"):
        return 0.01
    return 0.0001


def pair_code_from_str(s: str) -> str:
    """
    Extrahiert Paircode robust aus Text/Dateiname.
    Wichtig: keine "GBP*"-Verwechslung → wir matchen erst exakt PAIRS_28.
    """
    txt = str(s)

    # 1) Explizit OANDA_{PAIR}
    m = re.search(r"OANDA_([A-Z]{6})", txt.upper().replace(" ", ""))
    if m:
        code = m.group(1)
        return SPECIAL_PAIR_FIX.get(code, code)

    # 2) Buchstaben filtern
    up = re.sub(r"[^A-Z]", "", txt.upper())

    # Sonderfix zuerst
    for bad, real in SPECIAL_PAIR_FIX.items():
        if bad in up:
            return real

    # Exakt eines der 28 Paare suchen (wichtig für GBPxxy-Verwechslungen)
    for p in PAIRS_28:
        if p in up:
            return p

    # Fallback: irgendein 6er Block
    m2 = re.search(r"([A-Z]{6})", up)
    code = m2.group(1) if m2 else up[:6] or txt
    return SPECIAL_PAIR_FIX.get(code, code)


# -----------------------------------
# OHLC Reader (wie vorher)
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

    out = df.rename(
        columns={t: "time", o: "open", h: "high", l: "low", c: "close"}
    )[["time", "open", "high", "low", "close"]].copy()

    if pd.api.types.is_numeric_dtype(out["time"]):
        vmax = pd.Series(out["time"]).astype(float).abs().max()
        unit = "ms" if vmax > 1e12 else "s"
        out["time"] = pd.to_datetime(out["time"], unit=unit, utc=False)
    else:
        out["time"] = to_naive_datetime(out["time"])

    for col in ["open", "high", "low", "close"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = (
        out.dropna(subset=["time", "open", "high", "low", "close"])
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
                pass
    except Exception:
        return None
    return None


def find_ltf_files_map(ltf_dir: Path) -> Dict[str, Path]:
    """
    Mapping 'EURUSD' -> Pfad. Erkennt Pair aus Dateiname.
    """
    mp: Dict[str, Path] = {}
    if not ltf_dir.exists():
        return mp
    for p in ltf_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".csv", ".xlsx"}:
            code = pair_code_from_str(p.name).upper()
            if len(code) == 6 and code in PAIRS_28 and code not in mp:
                mp[code] = p
    return mp


# -----------------------------------
# Spaltenrobustes Laden der Wickdiffs
# -----------------------------------
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


def _pick_first_existing(df: pd.DataFrame, candidates: List[str]) -> str:
    last_err = None
    for cand in candidates:
        try:
            return _pick_col_generic(df, cand)
        except Exception as e:
            last_err = e
            continue
    raise KeyError(f"None of these columns found: {candidates} (last_err={last_err})")


def load_wickdiffs(mode: str, base: Path) -> pd.DataFrame:
    """
    Lädt die neueste Wickdiff-CSV je Modus und normalisiert auf ein einheitliches Schema:
      pair6, pivot_type, pivot_low/high, pivot_first/second_time, pivot_touch_time,
      wd_low/high, wd_first/second_time
    """
    mode = mode.upper().strip()

    if mode == "W":
        wd_dir = base / "outputs" / "wickdiffs" / "W→H4"
        pattern = "wick_diffs_*.csv"
    elif mode == "3D":
        wd_dir = base / "outputs" / "wickdiffs" / "3D→H1"
        pattern = "wick_diffs_*.csv"
    elif mode == "2W":
        wd_dir = base / "outputs" / "wickdiffs" / "2W→1D"
        pattern = "wick_diffs_*.csv"
    elif mode == "M":
        wd_dir = base / "outputs" / "wickdiffs" / "M→3D"
        pattern = "wick_diffs_*.csv"
    else:
        print(f"❌ Unbekannter Modus: {mode}")
        return pd.DataFrame()

    if not wd_dir.exists():
        print(f"❌ Wickdiff-Ordner nicht gefunden: {wd_dir}")
        return pd.DataFrame()

    files = sorted(wd_dir.glob(pattern))
    if not files:
        print(f"❌ Keine Wick-Diffs in {wd_dir} gefunden.")
        return pd.DataFrame()

    path = files[-1]
    print(f"🔄 Verwende Wick-Diffs ({mode}): {path}")
    df_raw = pd.read_csv(path)
    df_raw.columns = [str(c) for c in df_raw.columns]

    # Pflichtspalten: pair + pivot_type
    pair_col  = _pick_first_existing(df_raw, ["pair", "pair6", "pair_raw"])
    ptype_col = _pick_first_existing(df_raw, ["pivot_type", "direction"])

    # Pivot Gap low/high (kann weekly_gap_low etc heißen)
    gap_low_col  = _pick_first_existing(df_raw, ["gap_low", "weekly_gap_low", "3day_gap_low", "2w_gap_low", "m_gap_low"])
    gap_high_col = _pick_first_existing(df_raw, ["gap_high", "weekly_gap_high", "3day_gap_high", "2w_gap_high", "m_gap_high"])

    # Pivot times (first/second candle time)
    pivot_first_col  = _pick_first_existing(df_raw, ["first_candle_time", "weekly_first_candle_time", "3day_first_candle_time", "2w_first_candle_time", "m_first_candle_time"])
    pivot_second_col = _pick_first_existing(df_raw, ["second_candle_time", "weekly_second_candle_time", "3day_second_candle_time", "2w_second_candle_time", "m_second_candle_time"])

    # WD times + zone low/high
    wd_first_col  = _pick_first_existing(df_raw, ["wd_first_candle_time"])
    wd_second_col = _pick_first_existing(df_raw, ["wd_second_candle_time"])
    wd_low_col    = _pick_first_existing(df_raw, ["wd_zone_low", "wd_low"])
    wd_high_col   = _pick_first_existing(df_raw, ["wd_zone_high", "wd_high"])

    # Optional: first touch time (verschiedene Namensvarianten)
    pivot_touch_col = None
    for cand in [
        "first_touch_time",
        "weekly_first_touch_time",
        "3day_first_touch_time",
        "2w_first_touch_time",
        "m_first_touch_time",
        "pivot_touch_time",
    ]:
        try:
            pivot_touch_col = _pick_col_generic(df_raw, cand)
            break
        except KeyError:
            continue

    df = pd.DataFrame({
        "pair_raw": df_raw[pair_col].astype(str),
        "pivot_type": df_raw[ptype_col].astype(str).str.lower().str.strip(),
        "pivot_low": pd.to_numeric(df_raw[gap_low_col], errors="coerce"),
        "pivot_high": pd.to_numeric(df_raw[gap_high_col], errors="coerce"),
        "pivot_first_time": to_naive_datetime(df_raw[pivot_first_col]),
        "pivot_second_time": to_naive_datetime(df_raw[pivot_second_col]),
        "pivot_touch_time": (
            to_naive_datetime(df_raw[pivot_touch_col]) if pivot_touch_col else pd.NaT
        ),
        "wd_first_time": to_naive_datetime(df_raw[wd_first_col]),
        "wd_second_time": to_naive_datetime(df_raw[wd_second_col]),
        "wd_low": pd.to_numeric(df_raw[wd_low_col], errors="coerce"),
        "wd_high": pd.to_numeric(df_raw[wd_high_col], errors="coerce"),
    })

    df["pair6"] = df["pair_raw"].apply(pair_code_from_str).str.upper()

    # Order normalisieren
    pl = df[["pivot_low", "pivot_high"]].min(axis=1)
    ph = df[["pivot_low", "pivot_high"]].max(axis=1)
    df["pivot_low"], df["pivot_high"] = pl, ph

    wl = df[["wd_low", "wd_high"]].min(axis=1)
    wh = df[["wd_low", "wd_high"]].max(axis=1)
    df["wd_low"], df["wd_high"] = wl, wh

    df = df.dropna(
        subset=[
            "pair6","pivot_type",
            "pivot_low","pivot_high",
            "pivot_first_time","pivot_second_time",
            "wd_low","wd_high","wd_first_time","wd_second_time",
        ]
    ).reset_index(drop=True)

    # Nur gültige Paare
    df = df[df["pair6"].isin(PAIRS_28)].reset_index(drop=True)

    return df


# -----------------------------------
# Pivot-TP-Invalidation (deine Regel)
# -----------------------------------
def compute_pivot_tp_level(
    pivot_low: float,
    pivot_high: float,
    direction: str,
) -> Optional[float]:
    """
    1:1 Pivot-Extension:
      Long:  pivot_high + Range
      Short: pivot_low  - Range
    """
    pivot_low = float(pivot_low)
    pivot_high = float(pivot_high)
    if pivot_high <= pivot_low:
        return None
    rng = pivot_high - pivot_low
    if direction == "long":
        return pivot_high + rng
    return pivot_low - rng


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
    """
    Ab pivot_touch_time bis end_time:
      - Wenn TP-Level zuerst erreicht wird, bevor WD-Zone berührt wurde → invalid
      - Wenn WD-Zone zuerst berührt wird → ok
    """
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
# Entry-Findung & Trade-Simulation
# -----------------------------------
def find_entry_candle(
    ltf: pd.DataFrame,
    zone_low: float,
    zone_high: float,
    direction: str,
    start_time: pd.Timestamp,
    end_time: Optional[pd.Timestamp] = None,
) -> Optional[int]:
    """
    Entry-Kerze innerhalb WD-Zone:
    - Kill-Kerze: Körper komplett durch Zone → tot
    - Long: touched + close > hi + body komplett über hi
    - Short: touched + close < lo + body komplett unter lo
    """
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

        # Kill-Kerze
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
    """
    TP/SL:
    - TP = 1:1 Pivot-Extension (pivot_tp_level)
    - SL mind.:
      * Pivot-1.1 baseline (0.1*rng beyond pivot)
      * 50 Pips
      * groß genug, damit RR_MAX nicht überschritten wird
    - RR muss in [RR_MIN, RR_MAX]
    """
    pivot_low = float(pivot_low)
    pivot_high = float(pivot_high)
    if pivot_high <= pivot_low:
        return None, None, None

    rng = pivot_high - pivot_low

    if direction == "long":
        tp_price = pivot_high + rng
        baseline_sl_price = pivot_low - 0.1 * rng
        sl_dist_baseline = entry_price - baseline_sl_price
        tp_dist = tp_price - entry_price
    else:
        tp_price = pivot_low - rng
        baseline_sl_price = pivot_high + 0.1 * rng
        sl_dist_baseline = baseline_sl_price - entry_price
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
    """
    Sim:
    - Kerzen nach Entry
    - wenn TP & SL in derselben Kerze → konservativ SL zuerst
    """
    n = len(ltf)
    for j in range(entry_idx + 1, n):
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

        if hit_sl and hit_tp:
            return {
                "entry_time": ltf.iloc[entry_idx]["time"],
                "entry_price": entry_price,
                "exit_time": t,
                "exit_price": sl_price,
                "result": "loss",
                "rr_signed": -1.0,
            }
        if hit_sl:
            return {
                "entry_time": ltf.iloc[entry_idx]["time"],
                "entry_price": entry_price,
                "exit_time": t,
                "exit_price": sl_price,
                "result": "loss",
                "rr_signed": -1.0,
            }
        if hit_tp:
            return {
                "entry_time": ltf.iloc[entry_idx]["time"],
                "entry_price": entry_price,
                "exit_time": t,
                "exit_price": tp_price,
                "result": "win",
                "rr_signed": float(rr_pos),
            }

    return None


# -----------------------------------
# Run-Mode (W / 3D / 2W / M)
# -----------------------------------
def run_mode(
    base: Path,
    mode: str,
    ltf_dir: Path,
    ltf_label: str,
    pair_filter: Optional[Set[str]] = None,
    pivot_start: Optional[pd.Timestamp] = None,
    pivot_end: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    mode = mode.upper().strip()

    wd = load_wickdiffs(mode, base)
    if wd.empty:
        return pd.DataFrame()

    if pair_filter is not None:
        wd = wd[wd["pair6"].isin(set(x.upper() for x in pair_filter))].copy()

    if pivot_start is not None:
        wd = wd[wd["pivot_first_time"] >= pivot_start].copy()
    if pivot_end is not None:
        wd = wd[wd["pivot_first_time"] < pivot_end].copy()

    if wd.empty:
        print(f"⚠️ Keine Wick-Pivots nach Filter für Modus {mode}.")
        return pd.DataFrame()

    ltf_map = find_ltf_files_map(ltf_dir)
    if not ltf_map:
        print(f"❌ Keine LTF-Dateien in {ltf_dir} gefunden.")
        return pd.DataFrame()

    trades: List[dict] = []
    used_pivots: Set[Tuple] = set()
    skipped_pairs: Set[str] = set()

    max_days = MAX_DAYS_BY_MODE.get(mode, 14)

    for _, row in wd.iterrows():
        pair6 = str(row["pair6"]).upper()
        direction = str(row["pivot_type"]).lower()
        if direction not in {"long", "short"}:
            continue

        pivot_id = (
            pair6,
            direction,
            row["pivot_first_time"],
            row["pivot_second_time"],
            # WD-times mit rein, damit nicht versehentlich “ein Pivot” mehrere WD-Varianten blockiert
            row["wd_first_time"],
            row["wd_second_time"],
        )
        if pivot_id in used_pivots:
            continue

        ltf_path = ltf_map.get(pair6)
        if not ltf_path:
            skipped_pairs.add(pair6)
            continue

        ltf_df = read_ohlc_file(ltf_path)
        if ltf_df is None or ltf_df.empty:
            skipped_pairs.add(pair6)
            continue

        pivot_touch_time = row["pivot_touch_time"]
        if pd.isna(pivot_touch_time):
            continue

        end_time = pivot_touch_time + pd.Timedelta(days=max_days)

        # 1) Pivot invalid, wenn nach Touch TP zuerst kommt, bevor WD berührt wird
        if pivot_invalidated_by_tp_before_wd(
            ltf=ltf_df,
            pivot_touch_time=pivot_touch_time,
            end_time=end_time,
            zone_low=row["wd_low"],
            zone_high=row["wd_high"],
            direction=direction,
            pivot_low=row["pivot_low"],
            pivot_high=row["pivot_high"],
        ):
            continue

        # 2) Entry in WD-Zone im Zeitfenster
        entry_idx = find_entry_candle(
            ltf=ltf_df,
            zone_low=row["wd_low"],
            zone_high=row["wd_high"],
            direction=direction,
            start_time=pivot_touch_time,
            end_time=end_time,
        )
        if entry_idx is None:
            continue

        entry_price = float(ltf_df.loc[entry_idx, "close"])
        tp_price, sl_price, rr_pos = compute_tp_sl(
            entry_price, row["pivot_low"], row["pivot_high"], direction, pair6
        )
        if tp_price is None:
            continue

        sim = simulate_trade(
            ltf=ltf_df,
            entry_idx=entry_idx,
            direction=direction,
            entry_price=entry_price,
            tp_price=tp_price,
            sl_price=sl_price,
            rr_pos=rr_pos,
        )
        if sim is None:
            continue

        trades.append({
            "mode": mode,
            "ltf": ltf_label,
            "pair": pair6,
            "direction": direction,
            "pivot_first_time": row["pivot_first_time"],
            "pivot_second_time": row["pivot_second_time"],
            "pivot_low": row["pivot_low"],
            "pivot_high": row["pivot_high"],
            "pivot_range": float(row["pivot_high"] - row["pivot_low"]),
            "pivot_touch_time": row["pivot_touch_time"],
            "wd_first_time": row["wd_first_time"],
            "wd_second_time": row["wd_second_time"],
            "wd_low": row["wd_low"],
            "wd_high": row["wd_high"],
            "entry_time": sim["entry_time"],
            "entry_price": sim["entry_price"],
            "tp_price": tp_price,
            "sl_price": sl_price,
            "exit_time": sim["exit_time"],
            "exit_price": sim["exit_price"],
            "result": sim["result"],
            "rr_signed": sim["rr_signed"],
        })

        used_pivots.add(pivot_id)

    if skipped_pairs:
        print("ℹ️ Übersprungen (fehlende/ungültige LTF-Datei):", ", ".join(sorted(skipped_pairs)))

    return pd.DataFrame(trades)


# -----------------------------------
# Zusammenfassung
# -----------------------------------
def summarize_trades(
    df_trades: pd.DataFrame, mode: str, out_dir: Path, stamp: str
) -> None:
    if df_trades.empty:
        print(f"⚠️ Keine Trades für Modus {mode}.")
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
            "pair": pair,
            "n_trades": n,
            "wins": wins,
            "losses": losses,
            "avg_rr": avg_rr_all,
            "win_rate_%": win_rate,
            "avg_rr_win": avg_rr_win,
            "avg_rr_loss": avg_rr_loss,
        })

    summary = pd.DataFrame(rows).sort_values("pair").reset_index(drop=True)

    n_tot = len(df)
    wins_tot = int(df["is_win"].sum())
    losses_tot = int(n_tot - wins_tot)
    win_rate_tot = float(wins_tot / n_tot * 100.0) if n_tot else 0.0
    avg_rr_tot = float(df["rr_signed"].mean()) if n_tot else float("nan")
    avg_rr_win_tot = float(df.loc[df["is_win"], "rr_signed"].mean()) if wins_tot else float("nan")
    avg_rr_loss_tot = float(df.loc[~df["is_win"], "rr_signed"].mean()) if losses_tot else float("nan")

    all_row = pd.DataFrame([{
        "pair": "ALL",
        "n_trades": n_tot,
        "wins": wins_tot,
        "losses": losses_tot,
        "avg_rr": avg_rr_tot,
        "win_rate_%": win_rate_tot,
        "avg_rr_win": avg_rr_win_tot,
        "avg_rr_loss": avg_rr_loss_tot,
    }])

    summary = pd.concat([summary, all_row], ignore_index=True)

    summary_path = out_dir / f"trades_{mode}_summary_{stamp}.csv"
    summary.to_csv(summary_path, index=False)

    print(f"\n===== Zusammenfassung {mode}-Trades =====")
    print(f"Gesamt: {n_tot} Trades | Wins: {wins_tot} | Losses: {losses_tot} | Win-Rate: {win_rate_tot:.1f}%")
    print(f"Ø RR Gewinn: {avg_rr_win_tot:.3f} | Ø RR Verlust: {avg_rr_loss_tot:.3f} | EV/Trade: {avg_rr_tot:.3f}")
    print(f"💾 Summary gespeichert in: {summary_path.resolve()}")


# -----------------------------------
# Main – kompletter Backtest (alle Modis)
# -----------------------------------
def main_allpairs() -> None:
    base = Path(__file__).resolve().parent

    out_base = base / "outputs" / "trades"
    out_base.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    # Modus-Konfig: welches LTF wird für Validierung/Entry/Trade genutzt?
    MODES = [
        ("W",  base / "time frame data" / "4h data",   "H4"),
        ("3D", base / "time frame data" / "1h data",   "H1"),
        ("2W", base / "time frame data" / "daily data","D1"),
        ("M",  base / "time frame data" / "3D",        "3D"),
    ]

    for mode, ltf_dir, ltf_label in MODES:
        trades = run_mode(
            base=base,
            mode=mode,
            ltf_dir=ltf_dir,
            ltf_label=ltf_label,
        )
        if trades.empty:
            continue

        path = out_base / f"trades_{mode}_{stamp}.csv"
        trades.to_csv(path, index=False)
        print(f"\n💾 Trades {mode} gespeichert in: {path.resolve()}")
        summarize_trades(trades, mode, out_base, stamp)


if __name__ == "__main__":
    main_allpairs()
