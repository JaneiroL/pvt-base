from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# ------------------------------------------------------------
# Grund-Settings
# ------------------------------------------------------------

PAIRS_28 = {
    "AUDCAD", "AUDCHF", "AUDJPY", "AUDNZD", "AUDUSD",
    "CADJPY", "CHFJPY",
    "EURAUD", "EURCAD", "EURCHF", "EURGBP", "EURJPY", "EURNZD", "EURUSD",
    "GBPAUD", "GBPCAD", "GBPCHF", "GBPJPY", "GBPNZD", "GBPUSD",
    "NZDCAD", "NZDCHF", "NZDJPY", "NZDUSD",
    "USDCAD", "USDCHF", "USDJPY",
}

MIN_SL_PIPS = 50            # Mindest-SL-Größe
MIN_RR = 0.95               # kleinste erlaubte RR
MAX_RR = 1.49               # größte erlaubte RR
INVALID_WINDOW_DAYS = 14    # 14 echte Tage nach Pivot-Touch


# ------------------------------------------------------------
# Hilfsfunktionen
# ------------------------------------------------------------

def pip_size_for_pair(pair: str) -> float:
    """Grobe Pip-Size: JPY-Paare 0.01, Rest 0.0001."""
    pair = pair.upper()
    if pair.endswith("JPY"):
        return 0.01
    return 0.0001


def to_naive_ts(val) -> Optional[pd.Timestamp]:
    """String/Datum zu naive pandas.Timestamp (oder None)."""
    if pd.isna(val):
        return None
    ts = pd.to_datetime(val)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts


def load_ohlc(path: Path) -> pd.DataFrame:
    """OHLC-CSV laden und auf Standardspalten bringen."""
    df = pd.read_csv(path)
    # Spaltennamen vereinheitlichen
    cols = {c.lower(): c for c in df.columns}
    time_col = None
    for cand in ["time", "timestamp", "date"]:
        if cand in cols:
            time_col = cols[cand]
            break
    if time_col is None:
        raise ValueError(f"Keine Zeitspalte in {path}")

    df.rename(columns={time_col: "time"}, inplace=True)

    rename_map = {}
    for std, cands in {
        "open": ["open", "o"],
        "high": ["high", "h"],
        "low": ["low", "l"],
        "close": ["close", "c"],
    }.items():
        for cand in cands:
            if cand in cols:
                rename_map[cols[cand]] = std
                break
    df.rename(columns=rename_map, inplace=True)

    needed = {"time", "open", "high", "low", "close"}
    if not needed.issubset(df.columns):
        raise ValueError(f"Fehlende OHLC-Spalten in {path}: {needed - set(df.columns)}")

    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
    df.sort_values("time", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def find_ltf_files(ltf_dir: Path) -> Dict[str, Path]:
    """
    Map von Pair -> OHLC-CSV im gegebenen Timeframe-Ordner.
    Erlaubt Dateinamen wie 'OANDA_EURCHF_merged.csv' oder 'OANDA_EURCHF, 240.csv'.
    """
    mapping: Dict[str, Path] = {}
    if not ltf_dir.exists():
        print(f"⚠️ LTF-Ordner nicht gefunden: {ltf_dir}")
        return mapping

    for f in ltf_dir.glob("*.csv"):
        name = f.name.upper()
        # suche 6er-PAAR im Dateinamen
        for pair in PAIRS_28:
            if pair in name:
                mapping[pair] = f
                break
    return mapping


# ------------------------------------------------------------
# COT-Filter
# ------------------------------------------------------------

@dataclass
class COTWindow:
    pair: str
    direction: str   # "long" oder "short"
    start: pd.Timestamp
    end: pd.Timestamp


def find_cot_excel(base: Path) -> Optional[Path]:
    """Sucht nach einer COT-Excel irgendwo unterhalb von 'cot data' oder 'cot_data'."""
    for folder_name in ["cot data", "cot_data", "cot"]:
        folder = base / folder_name
        if not folder.exists():
            continue
        xlsx_files = sorted(folder.glob("*.xlsx"))
        if xlsx_files:
            # nimm die erste gefundene Datei
            return xlsx_files[0]
    return None


def load_cot_windows(base: Path) -> List[COTWindow]:
    """
    Erwartet eine Excel mit Spalten (irgendwie benannt):
    - pair (z.B. 'EURUSD')
    - direction / bias ('long' / 'short' oder 'L' / 'S')
    - start (Datum)
    - end (Datum)
    Falls Format unerwartet -> gibt leere Liste zurück und es wird ohne COT-Filter gearbeitet.
    """
    xlsx = find_cot_excel(base)
    if xlsx is None:
        print("⚠️ Keine COT-Excel gefunden – Trades werden OHNE COT-Filter berechnet.")
        return []

    print(f"📄 Lade COT-Excel: {xlsx}")
    try:
        sheets = pd.read_excel(xlsx, sheet_name=None)
    except Exception as exc:
        print(f"⚠️ Konnte COT-Excel nicht lesen ({exc}) – COT-Filter deaktiviert.")
        return []

    windows: List[COTWindow] = []

    for _, df in sheets.items():
        if df.empty:
            continue
        cols_l = {c.lower().strip(): c for c in df.columns}

        def pick_col(candidates: List[str]) -> Optional[str]:
            for cand in candidates:
                if cand in cols_l:
                    return cols_l[cand]
            return None

        col_pair = pick_col(["pair", "symbol", "paar"])
        col_dir = pick_col(["direction", "bias", "richtung", "longshort"])
        col_start = pick_col(["start", "from", "beginn", "von", "datefrom", "startdatum"])
        col_end = pick_col(["end", "bis", "to", "until", "dateto", "enddatum"])

        if not all([col_pair, col_dir, col_start, col_end]):
            continue

        for _, row in df.iterrows():
            pair = str(row[col_pair]).upper().strip()
            if pair not in PAIRS_28:
                continue

            direction = str(row[col_dir]).strip().lower()
            if direction.startswith("l"):
                direction = "long"
            elif direction.startswith("s"):
                direction = "short"
            else:
                continue

            start = to_naive_ts(row[col_start])
            end = to_naive_ts(row[col_end])
            if not start or not end:
                continue

            windows.append(COTWindow(pair=pair, direction=direction, start=start, end=end))

    if not windows:
        print("⚠️ Keine verwertbaren COT-Zeilen gefunden – COT-Filter deaktiviert.")
    else:
        print(f"✅ COT-Fenster geladen: {len(windows)} Einträge")

    return windows


def cot_allows_trade(
    windows: List[COTWindow],
    pair: str,
    direction: str,
    entry_time: pd.Timestamp,
) -> bool:
    """True, wenn entweder kein COT aktiv ist oder Entry in passendes Fenster fällt."""
    if not windows:
        # kein COT-Filter aktiv
        return True

    pair = pair.upper()
    d = direction.lower()
    for w in windows:
        if w.pair != pair or w.direction != d:
            continue
        if w.start <= entry_time <= w.end:
            return True
    return False


# ------------------------------------------------------------
# Wick-Diff & Pivot Daten normalisieren
# ------------------------------------------------------------

@dataclass
class WickDiffRow:
    mode: str              # "3D" oder "W"
    ltf: str               # "H1" oder "H4"
    pair: str
    direction: str         # "long" / "short"
    pivot_first_time: pd.Timestamp
    pivot_second_time: pd.Timestamp
    pivot_low: float
    pivot_high: float
    pivot_range: float
    pivot_touch_time: Optional[pd.Timestamp]
    wd_first_time: pd.Timestamp
    wd_second_time: pd.Timestamp
    wd_low: float
    wd_high: float


def normalize_wickdiff_row(sr: pd.Series, mode: str) -> WickDiffRow:
    """
    WickDiff-CSV-Zeile in ein einheitliches Objekt übersetzen.
    Erwartet Spalten (je nach Mode):
    - pair
    - pivot_type
      und dann:
      W:  weekly_first_candle_time, weekly_second_candle_time,
          weekly_gap_low, weekly_gap_high, weekly_first_touch_time
      3D: 3day_first_candle_time, 3day_second_candle_time,
          3day_gap_low, 3day_gap_high, 3day_first_touch_time
    - wd_first_candle_time, wd_second_candle_time, wd_zone_low, wd_zone_high
    - ltf / ltf_label (H1 / H4)
    """
    cols = {c.lower(): c for c in sr.index}

    def pick_col(candidates: List[str]) -> Optional[str]:
        for cand in candidates:
            if cand in cols:
                return cols[cand]
        return None

    pair_col = pick_col(["pair"])
    pivot_type_col = pick_col(["pivot_type", "type", "direction"])

    if mode == "W":
        pf_col = pick_col(["weekly_first_candle_time"])
        ps_col = pick_col(["weekly_second_candle_time"])
        pl_col = pick_col(["weekly_gap_low"])
        ph_col = pick_col(["weekly_gap_high"])
        pt_col = pick_col(["weekly_first_touch_time"])
    else:  # "3D"
        pf_col = pick_col(["3day_first_candle_time"])
        ps_col = pick_col(["3day_second_candle_time"])
        pl_col = pick_col(["3day_gap_low"])
        ph_col = pick_col(["3day_gap_high"])
        pt_col = pick_col(["3day_first_touch_time"])

    w1_col = pick_col(["wd_first_candle_time"])
    w2_col = pick_col(["wd_second_candle_time"])
    wl_col = pick_col(["wd_zone_low"])
    wh_col = pick_col(["wd_zone_high"])
    ltf_col = pick_col(["ltf", "ltf_label"])

    if not all([pair_col, pivot_type_col, pf_col, ps_col, pl_col, ph_col, w1_col, w2_col, wl_col, wh_col, ltf_col]):
        raise ValueError(f"Fehlende Spalten in WickDiff-Zeile: {sr.to_dict()}")

    pair = str(sr[pair_col]).upper()
    direction_raw = str(sr[pivot_type_col]).lower()
    direction = "long" if direction_raw.startswith("l") else "short"

    pivot_first_time = to_naive_ts(sr[pf_col])
    pivot_second_time = to_naive_ts(sr[ps_col])
    pivot_low = float(sr[pl_col])
    pivot_high = float(sr[ph_col])
    if pivot_low > pivot_high:
        pivot_low, pivot_high = pivot_high, pivot_low
    pivot_range = pivot_high - pivot_low

    pivot_touch_time = to_naive_ts(sr[pt_col]) if pt_col is not None else None

    wd_first_time = to_naive_ts(sr[w1_col])
    wd_second_time = to_naive_ts(sr[w2_col])
    wd_low = float(sr[wl_col])
    wd_high = float(sr[wh_col])
    if wd_low > wd_high:
        wd_low, wd_high = wd_high, wd_low

    ltf = str(sr[ltf_col]).upper()

    return WickDiffRow(
        mode=mode,
        ltf=ltf,
        pair=pair,
        direction=direction,
        pivot_first_time=pivot_first_time,
        pivot_second_time=pivot_second_time,
        pivot_low=pivot_low,
        pivot_high=pivot_high,
        pivot_range=pivot_range,
        pivot_touch_time=pivot_touch_time,
        wd_first_time=wd_first_time,
        wd_second_time=wd_second_time,
        wd_low=wd_low,
        wd_high=wd_high,
    )


# ------------------------------------------------------------
# SL / TP Berechnung unter allen Constraints
# ------------------------------------------------------------

@dataclass
class TradeSetup:
    tp_price: float
    sl_price: float
    rr: float


def compute_tp_sl_for_entry(
    pair: str,
    direction: str,
    entry_price: float,
    pivot_low: float,
    pivot_high: float,
) -> Optional[TradeSetup]:
    """
    Berechnet TP & SL mit folgenden Regeln:
    - TP = Pivot-Breite gespiegelt (long: pivot_high + range, short: pivot_low - range)
    - SL muss außerhalb des Pivots bei 1.1 * Range liegen
    - SL mind. MIN_SL_PIPS entfernt
    - RR zwischen MIN_RR und MAX_RR
    Gibt None zurück, wenn keine Konfiguration möglich ist.
    """
    pivot_low, pivot_high = sorted((pivot_low, pivot_high))
    pivot_range = pivot_high - pivot_low
    if pivot_range <= 0:
        return None

    pip = pip_size_for_pair(pair)
    min_sl_dist_price = MIN_SL_PIPS * pip

    direction = direction.lower()
    if direction == "long":
        tp_price = pivot_high + pivot_range
        baseline_sl_price = pivot_low - 0.1 * pivot_range

        reward_dist = tp_price - entry_price
        if reward_dist <= 0:
            return None

        baseline_min_dist = entry_price - baseline_sl_price
        # Minimal notwendige Risikodistanz (größer gleich):
        risk_dist_min = max(
            min_sl_dist_price,
            baseline_min_dist,
            reward_dist / MAX_RR,
        )
        # Maximal erlaubte Risikodistanz:
        risk_dist_max = reward_dist / MIN_RR

        if risk_dist_min <= 0 or risk_dist_min > risk_dist_max:
            return None

        sl_price = entry_price - risk_dist_min
        rr = reward_dist / risk_dist_min

    else:  # short
        tp_price = pivot_low - pivot_range
        baseline_sl_price = pivot_high + 0.1 * pivot_range

        reward_dist = entry_price - tp_price
        if reward_dist <= 0:
            return None

        baseline_min_dist = baseline_sl_price - entry_price
        risk_dist_min = max(
            min_sl_dist_price,
            baseline_min_dist,
            reward_dist / MAX_RR,
        )
        risk_dist_max = reward_dist / MIN_RR

        if risk_dist_min <= 0 or risk_dist_min > risk_dist_max:
            return None

        sl_price = entry_price + risk_dist_min
        rr = reward_dist / risk_dist_min

    if not (MIN_RR <= rr <= MAX_RR):
        return None

    return TradeSetup(tp_price=tp_price, sl_price=sl_price, rr=rr)


# ------------------------------------------------------------
# Trade-Simulation inkl. neuer Invalidation (100%-Move vor Entry)
# ------------------------------------------------------------

@dataclass
class TradeResult:
    mode: str
    ltf: str
    pair: str
    direction: str
    pivot_first_time: pd.Timestamp
    pivot_second_time: pd.Timestamp
    pivot_low: float
    pivot_high: float
    pivot_range: float
    wd_first_time: pd.Timestamp
    wd_second_time: pd.Timestamp
    wd_low: float
    wd_high: float
    entry_time: pd.Timestamp
    entry_price: float
    tp_price: float
    sl_price: float
    exit_time: pd.Timestamp
    exit_price: float
    result: str      # "win" / "loss"
    rr_signed: float


def simulate_trade_for_wick(
    wd: WickDiffRow,
    ohlc: pd.DataFrame,
    cot_windows: List[COTWindow],
) -> Tuple[Optional[TradeResult], Optional[str]]:
    """
    Führt alle Schritte aus:
    - Fenster nach Pivot-Touch (14 Tage)
    - Neue Invalidation: 100%-Pivot-Move (theoretisches TP-Level) vor Entry
    - Wick-Diff-Bounce als Entry
    - SL/TP-Berechnung inkl. RR-Constraints und Mindest-Pips
    - COT-Filter
    - Lauf des Trades bis TP oder SL
    Gibt entweder TradeResult zurück oder None + Reject-Reason.
    """
    if wd.pivot_touch_time is None:
        return None, "no_pivot_touch_time"

    # Zeitfenster nach Pivot-Touch
    start = wd.pivot_touch_time
    end = wd.pivot_touch_time + pd.Timedelta(days=INVALID_WINDOW_DAYS)

    sub = ohlc[(ohlc["time"] >= start) & (ohlc["time"] <= end)].copy()
    if sub.empty:
        return None, "no_ltf_data_in_window"

    direction = wd.direction.lower()
    pair = wd.pair.upper()

    # Theoretisches TP-Level für die neue Invalidation:
    # "Wenn dieses Level erreicht wird, ist der Pivot aufgebraucht."
    if direction == "long":
        tp_level_for_invalidation = wd.pivot_high + wd.pivot_range
    else:
        tp_level_for_invalidation = wd.pivot_low - wd.pivot_range

    entry_time = None
    entry_price = None
    reject_reason = None

    for _, row in sub.iterrows():
        t = row["time"]
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])

        # ---------------------------------------------
        # 1) NEUE INVALIDATION: 100%-Pivot-Move vor Entry
        # ---------------------------------------------
        if direction == "long":
            if high >= tp_level_for_invalidation:
                # Kurs hat bereits das theoretische TP-Level erreicht,
                # bevor eine valide Wick-Diff-Berührung entstanden ist.
                return None, "pivot_used_by_full_move_before_entry"
        else:  # short
            if low <= tp_level_for_invalidation:
                return None, "pivot_used_by_full_move_before_entry"

        # ---------------------------------------------
        # 2) Wick-Diff-Bounce als Entry
        # ---------------------------------------------
        touched_zone = (high >= wd.wd_low) and (low <= wd.wd_high)
        if not touched_zone:
            continue

        if direction == "long":
            if close > wd.wd_high:
                entry_time = t
                entry_price = close
                break
            else:
                reject_reason = "wick_touched_without_valid_bounce"
                break
        else:  # short
            if close < wd.wd_low:
                entry_time = t
                entry_price = close
                break
            else:
                reject_reason = "wick_touched_without_valid_bounce"
                break

    if entry_time is None or entry_price is None:
        if reject_reason is None:
            reject_reason = "no_valid_entry_in_14d"
        return None, reject_reason

    # ---------------------------------------------
    # 3) SL/TP-Konfiguration
    # ---------------------------------------------
    setup = compute_tp_sl_for_entry(
        pair=pair,
        direction=direction,
        entry_price=entry_price,
        pivot_low=wd.pivot_low,
        pivot_high=wd.pivot_high,
    )
    if setup is None:
        return None, "no_valid_sl_tp_rr_window"

    tp_price = setup.tp_price
    sl_price = setup.sl_price
    rr = setup.rr

    # ---------------------------------------------
    # 4) COT-Filter
    # ---------------------------------------------
    if not cot_allows_trade(cot_windows, pair, direction, entry_time):
        return None, "outside_cot_window"

    # ---------------------------------------------
    # 5) Trade-Lauf bis TP oder SL
    # ---------------------------------------------
    after_entry = ohlc[ohlc["time"] >= entry_time].copy()
    if after_entry.empty:
        return None, "no_data_after_entry"

    exit_time = None
    exit_price = None
    result = None

    for _, row in after_entry.iterrows():
        t = row["time"]
        high = float(row["high"])
        low = float(row["low"])

        if direction == "long":
            # zuerst TP prüfen, dann SL (Annäherung)
            if high >= tp_price:
                exit_time = t
                exit_price = tp_price
                result = "win"
                break
            if low <= sl_price:
                exit_time = t
                exit_price = sl_price
                result = "loss"
                break
        else:  # short
            if low <= tp_price:
                exit_time = t
                exit_price = tp_price
                result = "win"
                break
            if high >= sl_price:
                exit_time = t
                exit_price = sl_price
                result = "loss"
                break

    if result is None:
        return None, "no_tp_or_sl_hit_until_data_end"

    # RR mit Vorzeichen
    if direction == "long":
        risk = entry_price - sl_price
        if result == "win":
            rr_signed = (tp_price - entry_price) / risk
        else:
            rr_signed = -1.0
    else:
        risk = sl_price - entry_price
        if result == "win":
            rr_signed = (entry_price - tp_price) / risk
        else:
            rr_signed = -1.0

    trade = TradeResult(
        mode=wd.mode,
        ltf=wd.ltf,
        pair=pair,
        direction=direction,
        pivot_first_time=wd.pivot_first_time,
        pivot_second_time=wd.pivot_second_time,
        pivot_low=wd.pivot_low,
        pivot_high=wd.pivot_high,
        pivot_range=wd.pivot_range,
        wd_first_time=wd.wd_first_time,
        wd_second_time=wd.wd_second_time,
        wd_low=wd.wd_low,
        wd_high=wd.wd_high,
        entry_time=entry_time,
        entry_price=entry_price,
        tp_price=tp_price,
        sl_price=sl_price,
        exit_time=exit_time,
        exit_price=exit_price,
        result=result,
        rr_signed=rr_signed,
    )
    return trade, None


# ------------------------------------------------------------
# Zusammenfassung / Summary
# ------------------------------------------------------------

def build_summary(trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    Baut eine gut lesbare Summary:
    - Erste Zeile: OVERALL (alle Paare)
    - Danach je Paar eine Zeile.
    Spalten:
    scope, pair, n_trades, wins, losses, win_rate_%,
    avg_rr_all, avg_rr_win, avg_rr_loss, total_rr
    """
    rows = []

    def summary_for_subset(df: pd.DataFrame, pair: str, scope: str):
        if df.empty:
            return {
                "scope": scope,
                "pair": pair,
                "n_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate_%": 0.0,
                "avg_rr_all": 0.0,
                "avg_rr_win": 0.0,
                "avg_rr_loss": 0.0,
                "total_rr": 0.0,
            }
        wins = (df["result"] == "win").sum()
        losses = (df["result"] == "loss").sum()
        n_trades = len(df)
        win_rate = (wins / n_trades * 100.0) if n_trades > 0 else 0.0
        total_rr = df["rr_signed"].sum()
        avg_rr_all = df["rr_signed"].mean()

        wins_df = df[df["result"] == "win"]
        losses_df = df[df["result"] == "loss"]

        avg_rr_win = float(wins_df["rr_signed"].mean()) if not wins_df.empty else 0.0
        avg_rr_loss = float(losses_df["rr_signed"].mean()) if not losses_df.empty else 0.0

        return {
            "scope": scope,
            "pair": pair,
            "n_trades": int(n_trades),
            "wins": int(wins),
            "losses": int(losses),
            "win_rate_%": win_rate,
            "avg_rr_all": avg_rr_all,
            "avg_rr_win": avg_rr_win,
            "avg_rr_loss": avg_rr_loss,
            "total_rr": total_rr,
        }

    # Overall
    rows.append(summary_for_subset(trades_df, "ALL", "OVERALL"))

    # je Pair
    for pair, df_pair in trades_df.groupby("pair"):
        rows.append(summary_for_subset(df_pair, pair, "PAIR"))

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# Hauptlauf für einen Mode (3D/H1 oder W/H4)
# ------------------------------------------------------------

def run_for_mode(
    base: Path,
    mode: str,          # "3D" oder "W"
    ltf_label: str,     # "H1" oder "H4"
    wick_dir_name: str, # z.B. "3D→H1"
    cot_windows: List[COTWindow],
) -> None:
    # WickDiff-Datei finden (neueste)
    wick_dir = base / "outputs" / "wickdiffs" / wick_dir_name
    if not wick_dir.exists():
        print(f"⚠️ WickDiff-Ordner fehlt: {wick_dir}")
        return

    pattern = f"wick_diffs_{ltf_label}_*.csv"
    wick_files = sorted(wick_dir.glob(pattern))
    if not wick_files:
        print(f"⚠️ Keine WickDiff-CSV gefunden in {wick_dir} ({pattern})")
        return

    wick_path = wick_files[-1]
    print(f"📥 Verwende WickDiff-Datei ({mode}): {wick_path.name}")

    df_wd = pd.read_csv(wick_path)

    # LTF-Daten-Dateien finden
    if ltf_label == "H1":
        ltf_dir = base / "time frame data" / "1h data"
    else:
        ltf_dir = base / "time frame data" / "4h data"

    ltf_files = find_ltf_files(ltf_dir)
    ltf_cache: Dict[str, pd.DataFrame] = {}

    trades: List[TradeResult] = []
    rejections: List[Dict[str, object]] = []

    for _, sr in df_wd.iterrows():
        try:
            wd = normalize_wickdiff_row(sr, mode=mode)
        except Exception as exc:
            rejections.append({
                "mode": mode,
                "ltf": ltf_label,
                "pair": sr.get("pair", ""),
                "direction": sr.get("pivot_type", ""),
                "reason": f"normalize_error: {exc}",
            })
            continue

        pair = wd.pair.upper()
        if pair not in ltf_files:
            rejections.append({
                "mode": mode,
                "ltf": ltf_label,
                "pair": pair,
                "direction": wd.direction,
                "reason": "no_ltf_file",
            })
            continue

        if pair not in ltf_cache:
            try:
                ltf_cache[pair] = load_ohlc(ltf_files[pair])
            except Exception as exc:
                rejections.append({
                    "mode": mode,
                    "ltf": ltf_label,
                    "pair": pair,
                    "direction": wd.direction,
                    "reason": f"load_ohlc_error: {exc}",
                })
                continue

        ohlc = ltf_cache[pair]

        trade, reason = simulate_trade_for_wick(wd, ohlc, cot_windows)
        if trade is not None:
            trades.append(trade)
        else:
            rejections.append({
                "mode": mode,
                "ltf": ltf_label,
                "pair": pair,
                "direction": wd.direction,
                "pivot_first_time": wd.pivot_first_time,
                "pivot_second_time": wd.pivot_second_time,
                "pivot_low": wd.pivot_low,
                "pivot_high": wd.pivot_high,
                "wd_first_time": wd.wd_first_time,
                "wd_second_time": wd.wd_second_time,
                "reason": reason,
            })

    # ---------------------------------------------
    # Outputs schreiben
    # ---------------------------------------------
    trades_dir = base / "outputs" / "trades"
    trades_dir.mkdir(parents=True, exist_ok=True)

    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    trades_csv = trades_dir / f"trades_{mode}_{timestamp}.csv"
    rejections_csv = trades_dir / f"trade_rejections_{mode}_{ltf_label}_{timestamp}.csv"
    summary_csv = trades_dir / f"trades_{mode}_summary_{timestamp}.csv"

    if trades:
        df_trades = pd.DataFrame([t.__dict__ for t in trades])
        df_trades.to_csv(trades_csv, index=False)
        print(f"✅ Trades ({mode}) gespeichert: {trades_csv.name}")

        df_summary = build_summary(df_trades)
        df_summary.to_csv(summary_csv, index=False)
        print(f"📊 Summary ({mode}) gespeichert: {summary_csv.name}")
    else:
        print(f"⚠️ Keine validen Trades für Mode {mode}.")

    if rejections:
        df_rej = pd.DataFrame(rejections)
        df_rej.to_csv(rejections_csv, index=False)
        print(f"🧾 Rejections ({mode}) gespeichert: {rejections_csv.name}")


# ------------------------------------------------------------
# main()
# ------------------------------------------------------------

def main() -> None:
    base = Path(__file__).resolve().parent

    print("🚀 Starte Trades-Journal Backtest ...")

    # COT-Fenster laden (falls vorhanden)
    cot_windows = load_cot_windows(base)

    # 3D/H1
    run_for_mode(
        base=base,
        mode="3D",
        ltf_label="H1",
        wick_dir_name="3D→H1",
        cot_windows=cot_windows,
    )

    # Weekly/H4
    run_for_mode(
        base=base,
        mode="W",
        ltf_label="H4",
        wick_dir_name="W→H4",
        cot_windows=cot_windows,
    )

    print("🏁 Backtest fertig.")


if __name__ == "__main__":
    main()
