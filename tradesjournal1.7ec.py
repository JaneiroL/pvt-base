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

# Sonderfall: OANDAG → GBPNZD
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
    if pair.endswith("JPY"):
        return 0.01
    return 0.0001


def pair_code_from_str(s: str) -> str:
    """
    Extrahiert einen 6-Buchstaben-Paircode aus Dateinamen / Text wie:
    - 'OANDA_GBPAUD_H1.csv' → GBPAUD
    - 'OANDAGBPNZD' → GBPNZD (Spezialmapping)
    - 'EURUSD' → EURUSD
    """
    txt = str(s)
    # 1) Klassischer OANDA_FX-Name
    m = re.search(r"OANDA_([A-Z]{6})", txt.upper().replace(" ", ""))
    if m:
        code = m.group(1)
    else:
        # 2) Alles auf A–Z filtern und Paar darin suchen
        up = re.sub(r"[^A-Z]", "", txt.upper())
        # Erst Spezialfix prüfen
        for bad, real in SPECIAL_PAIR_FIX.items():
            if bad in up:
                return real
        # Dann alle bekannten 28 Paare versuchen
        for p in PAIRS_28:
            if p in up:
                return p
        # Fallback: irgendein 6er Block
        m2 = re.search(r"([A-Z]{6})", up)
        code = m2.group(1) if m2 else up[:6] or txt
    return SPECIAL_PAIR_FIX.get(code, code)


# -----------------------------------
# OHLC-Reader
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
    Baut ein Mapping:
        'EURUSD' -> Pfad zur 1h / 4h Datei
    Durchsucht rekursiv ltf_dir und erkennt Paare über den Dateinamen.
    """
    mp: Dict[str, Path] = {}
    if not ltf_dir.exists():
        return mp
    for p in ltf_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".csv", ".xlsx"}:
            code = pair_code_from_str(p.name)
            code_u = code.upper()
            if len(code_u) == 6 and code_u in PAIRS_28 and code_u not in mp:
                mp[code_u] = p
    return mp


# -----------------------------------
# Wick-Diffs laden
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


def load_wickdiffs(mode: str, base: Path) -> pd.DataFrame:
    if mode == "W":
        wd_dir = base / "outputs" / "wickdiffs" / "W→H4"
        pattern = "wick_diffs_H4_*.csv"
    else:
        wd_dir = base / "outputs" / "wickdiffs" / "3D→H1"
        pattern = "wick_diffs_H1_*.csv"

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

    pair_col = _pick_col_generic(df_raw, "pair")
    ptype_col = _pick_col_generic(df_raw, "pivot_type")
    gap_low_col = _pick_col_generic(df_raw, "gap_low")
    gap_high_col = _pick_col_generic(df_raw, "gap_high")
    first_candle_col = _pick_col_generic(df_raw, "first_candle_time")
    second_candle_col = _pick_col_generic(df_raw, "second_candle_time")
    wd_first_col = _pick_col_generic(df_raw, "wd_first_candle_time")
    wd_second_col = _pick_col_generic(df_raw, "wd_second_candle_time")
    wd_low_col = _pick_col_generic(df_raw, "wd_zone_low")
    wd_high_col = _pick_col_generic(df_raw, "wd_zone_high")

    first_touch_col = None
    for cand in ["first_touch_time", "weekly_first_touch_time", "3day_first_touch_time"]:
        try:
            first_touch_col = _pick_col_generic(df_raw, cand)
            break
        except KeyError:
            continue

    df = pd.DataFrame({
        "pair_raw": df_raw[pair_col].astype(str),
        "pivot_type": df_raw[ptype_col].astype(str).str.lower().str.strip(),
        "pivot_low": pd.to_numeric(df_raw[gap_low_col], errors="coerce"),
        "pivot_high": pd.to_numeric(df_raw[gap_high_col], errors="coerce"),
        "pivot_first_time": to_naive_datetime(df_raw[first_candle_col]),
        "pivot_second_time": to_naive_datetime(df_raw[second_candle_col]),
        "pivot_touch_time": (
            to_naive_datetime(df_raw[first_touch_col]) if first_touch_col else pd.NaT
        ),
        "wd_first_time": to_naive_datetime(df_raw[wd_first_col]),
        "wd_second_time": to_naive_datetime(df_raw[wd_second_col]),
        "wd_low": pd.to_numeric(df_raw[wd_low_col], errors="coerce"),
        "wd_high": pd.to_numeric(df_raw[wd_high_col], errors="coerce"),
    })
    df["pair6"] = df["pair_raw"].apply(pair_code_from_str).str.upper()

    df = df.dropna(
        subset=[
            "pivot_low",
            "pivot_high",
            "pivot_first_time",
            "pivot_second_time",
            "wd_low",
            "wd_high",
            "wd_first_time",
            "wd_second_time",
        ]
    ).reset_index(drop=True)
    return df


# -----------------------------------
# Pivot-TP-Niveau & Invalidation
# -----------------------------------
def compute_pivot_tp_level(pivot_low: float, pivot_high: float, direction: str) -> Optional[float]:
    """
    TP-Level = 1:1-Extension der Pivot-Breite:
      Long  → pivot_high + (pivot_high - pivot_low)
      Short → pivot_low  - (pivot_high - pivot_low)
    """
    pivot_low = float(pivot_low)
    pivot_high = float(pivot_high)
    if pivot_high <= pivot_low:
        return None
    rng = pivot_high - pivot_low
    if direction == "long":
        return pivot_high + rng
    else:
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
    Neue Regel:
    Nach dem ersten Pivot-Touch (pivot_touch_time) schauen wir im LTF:

    - Wenn zuerst das 1:1-Pivot-TP-Level erreicht wird
    - BEVOR die WD-Zone (zone_low/zone_high) jemals berührt wurde,
      dann ist der gesamte große Pivot INVALID und es darf NIE ein Entry aus
      diesem Pivot entstehen.
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

        # Wenn wir zuerst WD sehen → keine Invalidation
        if touched_wd:
            return False

        # Wenn wir zuerst TP-Level sehen → Pivot invalid
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
    Findet die erste gültige Entry-Kerze nach start_time (optional begrenzt durch end_time),
    mit folgenden Regeln:

    - Kill-Kerze: Körper geht komplett durch die WD-Zone → KEIN Trade (Pivot tot)
    - Gültiger Entry:
        Long:  Kerze berührt die Zone + schließt darüber, kompletter Körper über Zone
        Short: Kerze berührt die Zone + schließt darunter, kompletter Körper unter Zone
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

        # Kill-Kerze: Körper vollständig durch die Zone
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
            closed_above = c > hi
            body_above = body_low >= hi
            if touched and closed_above and body_above:
                return idx_ltf
        else:
            closed_below = c < lo
            body_below = body_high <= lo
            if touched and closed_below and body_below:
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
    SL/TP-Berechnung:

    - TP immer 1:1 zur Pivot-Breite (gleiche Logik wie bei Invalidation)
    - SL-Basis = Pivot-1.1 (also etwas unter/über dem Pivot)
    - MIN_SL_PIPS (z.B. 50 Pips)
    - RR in [RR_MIN, RR_MAX] (z.B. 0.95–1.49), sonst kein Trade
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

    # Mindestens: Pivot-1.1, 50 Pips, sowie groß genug für RR_MAX
    sl_dist = max(sl_dist_baseline, min_sl_dist, tp_dist / RR_MAX)

    rr = tp_dist / sl_dist
    if rr < RR_MIN or rr > RR_MAX:
        return None, None, None

    if direction == "long":
        sl_price = entry_price - sl_dist
    else:
        sl_price = entry_price + sl_dist

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
    Simuliert den Trade auf Candle-Lows/Highs:

    - Wenn SL und TP in derselben Kerze getroffen → konservativ SL
    - Sonst erster Treffer entscheidet
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
            # konservativ: SL zuerst
            result = "loss"
            exit_price = sl_price
            rr_signed = -1.0
            exit_time = t
            break
        elif hit_sl:
            result = "loss"
            exit_price = sl_price
            rr_signed = -1.0
            exit_time = t
            break
        elif hit_tp:
            result = "win"
            exit_price = tp_price
            rr_signed = float(rr_pos)
            exit_time = t
            break
    else:
        # Weder SL noch TP jemals getroffen → wir werten Trade nicht
        return None

    return {
        "entry_time": ltf.iloc[entry_idx]["time"],
        "entry_price": entry_price,
        "exit_time": exit_time,
        "exit_price": exit_price,
        "result": result,
        "rr_signed": rr_signed,
    }


# -----------------------------------
# Run-Mode (hier nur 3D → H1, aber generisch gelassen)
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
    wd = load_wickdiffs(mode, base)
    if wd.empty:
        return pd.DataFrame()

    # Nur gewünschte Paare (für diesen Run später: {"EURCHF"})
    if pair_filter is not None:
        wd = wd[wd["pair6"].isin(pair_filter)].copy()

    # Zeitraum-Filter auf Pivot-Entstehung (erste Pivot-Kerze)
    if pivot_start is not None:
        wd = wd[wd["pivot_first_time"] >= pivot_start].copy()
    if pivot_end is not None:
        wd = wd[wd["pivot_first_time"] <= pivot_end].copy()

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

    # 6 Tage für 3D, 14 Tage für Weekly
    max_days = 14 if mode == "W" else 6

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
        )
        if pivot_id in used_pivots:
            continue  # Nur ein Trade pro großem Pivot

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

        # Gültigkeitsfenster des Pivots nach Pivot-Touch
        end_time = pivot_touch_time + pd.Timedelta(days=max_days)

        # 1) Neue Invalidation: Bounce von Pivot → 1:1-TP-Level, ohne WD-Touch
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
            # Pivot komplett invalid → kein Trade
            continue

        # 2) Entry suchen innerhalb des 6/14-Tage-Fensters
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

        # 3) SL/TP berechnen (inkl. RR-Regeln und min. SL)
        entry_price = float(ltf_df.loc[entry_idx, "close"])
        tp_price, sl_price, rr_pos = compute_tp_sl(
            entry_price, row["pivot_low"], row["pivot_high"], direction, pair6
        )
        if tp_price is None:
            continue

        # 4) Trade simulieren
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

        trades.append(
            {
                "mode": mode,
                "ltf": ltf_label,
                "pair": pair6,
                "direction": direction,
                "pivot_first_time": row["pivot_first_time"],
                "pivot_second_time": row["pivot_second_time"],
                "pivot_low": row["pivot_low"],
                "pivot_high": row["pivot_high"],
                "pivot_range": float(row["pivot_high"] - row["pivot_low"]),
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
            }
        )
        used_pivots.add(pivot_id)

    if skipped_pairs:
        print(
            "ℹ️ Übersprungen (fehlende/ungültige LTF-Datei):",
            ", ".join(sorted(skipped_pairs)),
        )

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
        rows.append(
            {
                "pair": pair,
                "n_trades": n,
                "wins": wins,
                "losses": losses,
                "avg_rr": avg_rr_all,
                "win_rate_%": win_rate,
                "avg_rr_win": avg_rr_win,
                "avg_rr_loss": avg_rr_loss,
            }
        )

    summary = pd.DataFrame(rows).sort_values("pair").reset_index(drop=True)

    # Gesamt-Übersicht (ALL-Zeile)
    n_tot = len(df)
    wins_tot = int(df["is_win"].sum())
    losses_tot = int(n_tot - wins_tot)
    win_rate_tot = float(wins_tot / n_tot * 100.0) if n_tot else 0.0
    avg_rr_tot = float(df["rr_signed"].mean()) if n_tot else float("nan")
    avg_rr_win_tot = (
        float(df.loc[df["is_win"], "rr_signed"].mean()) if wins_tot else float("nan")
    )
    avg_rr_loss_tot = (
        float(df.loc[~df["is_win"], "rr_signed"].mean()) if losses_tot else float("nan")
    )

    all_row = pd.DataFrame(
        [
            {
                "pair": "ALL",
                "n_trades": n_tot,
                "wins": wins_tot,
                "losses": losses_tot,
                "avg_rr": avg_rr_tot,
                "win_rate_%": win_rate_tot,
                "avg_rr_win": avg_rr_win_tot,
                "avg_rr_loss": avg_rr_loss_tot,
            }
        ]
    )
    summary = pd.concat([summary, all_row], ignore_index=True)

    summary_path = out_dir / f"trades_{mode}_summary_{stamp}.csv"
    summary.to_csv(summary_path, index=False)

    # Menschlich lesbarer Konsolen-Output
    print(f"\n===== Zusammenfassung {mode}-Trades =====")
    print(
        f"Gesamt: {n_tot} Trades | Wins: {wins_tot} | Losses: {losses_tot} "
        f"| Win-Rate: {win_rate_tot:.1f}%"
    )
    print(
        f"Ø RRR Gewinn: {avg_rr_win_tot:.3f} | Ø RRR Verlust: {avg_rr_loss_tot:.3f} "
        f"| Erwartungswert pro Trade: {avg_rr_tot:.3f}"
    )
    print(f"💾 Summary gespeichert in: {summary_path.resolve()}")


# -----------------------------------
# Main – EURCHF, 3D, 2015-01-01 bis 2015-08-31
# -----------------------------------
def main() -> None:
    base = Path(__file__).resolve().parent

    out_base = base / "outputs" / "trades"
    out_base.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    # Zeitraum: 01.01.2015 bis inkl. 31.08.2015 (per <= Filter)
    start = pd.Timestamp(2015, 1, 1)
    end   = pd.Timestamp(2015, 8, 31, 23, 59, 59)

    trades_3d = run_mode(
        base=base,
        mode="3D",
        ltf_dir=base / "time frame data" / "1h data",
        ltf_label="H1",
        pair_filter={"EURCHF"},
        pivot_start=start,
        pivot_end=end,
    )
    if not trades_3d.empty:
        path_3d = out_base / f"trades_3D_EURCHF_20150101_20150831_{stamp}.csv"
        trades_3d.to_csv(path_3d, index=False)
        print(f"\n💾 Trades 3D EURCHF 2015 gespeichert in: {path_3d.resolve()}")
        summarize_trades(trades_3d, "3D_EURCHF_2015_01_01_2015_08_31", out_base, stamp)
    else:
        print("⚠️ Keine Trades für EURCHF im Zeitraum 2015-01-01 bis 2015-08-31 gefunden.")


if __name__ == "__main__":
    main()
