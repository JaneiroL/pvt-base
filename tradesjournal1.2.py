from __future__ import annotations

from pathlib import Path
import re
from typing import Dict, List, Tuple, Optional

import pandas as pd
from pandas.api.types import is_datetime64tz_dtype

# --------------------------------------------------
# Konfiguration
# --------------------------------------------------
MIN_RR = 0.95          # minimale Chance-Risiko-Einheit
MAX_RR = 1.49          # maximale Chance-Risiko-Einheit
MIN_SL_PIPS = 50       # Stop-Loss immer mindestens 50 Pips
ENTRY_WINDOW_DAYS = 14 # nach Pivot-Touch: max. 14 Kalendertage für Entry

# 28er-Universum
PAIRS_28 = {
    "AUDCAD","AUDCHF","AUDJPY","AUDNZD","AUDUSD",
    "CADCHF","CADJPY",
    "CHFJPY",
    "EURAUD","EURCAD","EURCHF","EURGBP","EURJPY","EURNZD","EURUSD",
    "GBPAUD","GBPCAD","GBPCHF","GBPJPY","GBPUSD","GBPNZD",
    "NZDCAD","NZDCHF","NZDJPY","NZDUSD",
    "USDCAD","USDCHF","USDJPY",
}

# Sonderfall-Map, wenn Dateinamen komisch sind
SPECIAL_PAIR_FIX = {
    "OANDAG": "GBPNZD",
}


# --------------------------------------------------
# Basis-Helfer
# --------------------------------------------------
def to_naive_datetime(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce", utc=True)
    if is_datetime64tz_dtype(dt.dtype):
        dt = dt.dt.tz_convert(None)
    return dt


def pip_size_for(pair: str) -> float:
    pair = pair.upper()
    if pair.endswith("JPY"):
        return 0.01
    return 0.0001


def pair_code_from_str(s: str) -> str:
    """Versucht aus Dateiname / Text einen 6-stelligen FX-Code zu extrahieren."""
    txt = str(s)
    # z.B. OANDA_EURCHF
    m = re.search(r"OANDA_([A-Z]{6})", txt.upper().replace(" ", ""))
    if m:
        code = m.group(1)
    else:
        up = re.sub(r"[^A-Z]", "", txt.upper())
        # Sonderfälle zuerst
        for bad, real in SPECIAL_PAIR_FIX.items():
            if bad in up:
                return real
        # Dann Standard-28er
        for p in PAIRS_28:
            if p in up:
                return p
        # Fallback: erste 6 Großbuchstaben
        m2 = re.search(r"([A-Z]{6})", up)
        code = m2.group(1) if m2 else up[:6] or txt
    return SPECIAL_PAIR_FIX.get(code, code)


def read_ohlc_file(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")
        df = next(iter(sheets.values()))

    cols = {str(c).strip().lower(): c for c in df.columns}

    def pick(cands):
        for cand in cands:
            if cand in cols:
                return cols[cand]
        for c in df.columns:
            name = str(c).strip().lower()
            for cand in cands:
                if cand in name:
                    return c
        raise KeyError(cands)

    t = pick(["time", "timestamp", "date", "datetime", "unnamed: 0"])
    o = pick(["open", "o"])
    h = pick(["high", "h"])
    l = pick(["low", "l"])
    c = pick(["close", "c"])

    out = df.rename(columns={t: "time", o: "open", h: "high", l: "low", c: "close"})[
        ["time", "open", "high", "low", "close"]
    ].copy()

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


def find_ltf_files_map(ltf_dir: Path) -> Dict[str, Path]:
    """Sucht in einem TF-Ordner (1h / 4h) nach Dateien und mappt sie auf 6er-Paircodes."""
    mp: Dict[str, Path] = {}
    if not ltf_dir.exists():
        return mp

    for p in ltf_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".csv", ".xlsx"}:
            code = pair_code_from_str(p.name)
            if len(code) == 6 and code.upper() in PAIRS_28 and code not in mp:
                mp[code] = p
    return mp


# --------------------------------------------------
# COT-Signale
# --------------------------------------------------
def _pick_col_generic(df: pd.DataFrame, name: str) -> str:
    """Sucht eine Spalte, deren Name `name` enthält (case-insensitive)."""
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


def load_cot_signals(path: Path) -> Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp, str]]]:
    """
    Liest die COT-Signale ein.

    Rückgabe:
        { 'EURUSD': [(start, end, 'long'), (start2, end2, 'short'), ...], ... }
    """
    sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")
    records: Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp, str]]] = {}

    for sheet_name, df in sheets.items():
        if df is None or df.empty:
            continue
        df2 = df.dropna(how="all").copy()
        if df2.empty:
            continue
        df2.columns = [str(c).strip().lower() for c in df2.columns]

        has_pair = any("pair" in c or "symbol" in c or "instrument" in c for c in df2.columns)
        dir_cols = ["direction", "bias", "signal", "richtung", "type", "typ", "longshort", "long_short"]
        start_cols = ["start", "begin", "von", "from", "start_date"]
        end_cols = ["end", "bis", "to", "end_date"]

        if has_pair:
            # Variante: eine Tabelle mit Spalte "pair"
            try:
                pair_col = _pick_col_generic(df2, "pair")
            except KeyError:
                pair_col = None
                for cand in ["symbol", "instrument"]:
                    try:
                        pair_col = _pick_col_generic(df2, cand)
                        break
                    except KeyError:
                        pair_col = None
                if pair_col is None:
                    continue

            # Richtung
            try:
                dir_col = _pick_col_generic(df2, dir_cols[0])
            except KeyError:
                found = None
                for cand in dir_cols[1:]:
                    try:
                        found = _pick_col_generic(df2, cand)
                        break
                    except KeyError:
                        continue
                if found is None:
                    continue
                dir_col = found

            # Start/Ende
            try:
                start_col = _pick_col_generic(df2, start_cols[0])
            except KeyError:
                found = None
                for cand in start_cols[1:]:
                    try:
                        found = _pick_col_generic(df2, cand)
                        break
                    except KeyError:
                        continue
                if found is None:
                    continue
                start_col = found

            try:
                end_col = _pick_col_generic(df2, end_cols[0])
            except KeyError:
                found = None
                for cand in end_cols[1:]:
                    try:
                        found = _pick_col_generic(df2, cand)
                        break
                    except KeyError:
                        continue
                if found is None:
                    continue
                end_col = found

            for _, row in df2.iterrows():
                pair_raw = str(row[pair_col])
                if not pair_raw or pair_raw.lower().startswith("nan"):
                    continue
                pair = pair_code_from_str(pair_raw).upper()
                if pair not in PAIRS_28:
                    continue

                bias_raw = str(row[dir_col]).strip().lower()
                if not bias_raw:
                    continue
                if bias_raw.startswith("l"):
                    bias = "long"
                elif bias_raw.startswith("s"):
                    bias = "short"
                else:
                    continue

                start = pd.to_datetime(row[start_col], errors="coerce")
                end = pd.to_datetime(row[end_col], errors="coerce")
                if pd.isna(start) or pd.isna(end):
                    continue
                start = start.normalize()
                end = end.normalize()
                records.setdefault(pair, []).append((start, end, bias))

        else:
            # Variante: ein Sheet pro Pair
            pair = pair_code_from_str(sheet_name).upper()
            if pair not in PAIRS_28:
                continue

            try:
                dir_col = _pick_col_generic(df2, dir_cols[0])
            except KeyError:
                found = None
                for cand in dir_cols[1:]:
                    try:
                        found = _pick_col_generic(df2, cand)
                        break
                    except KeyError:
                        continue
                if found is None:
                    continue
                dir_col = found

            try:
                start_col = _pick_col_generic(df2, start_cols[0])
            except KeyError:
                found = None
                for cand in start_cols[1:]:
                    try:
                        found = _pick_col_generic(df2, cand)
                        break
                    except KeyError:
                        continue
                if found is None:
                    continue
                start_col = found

            try:
                end_col = _pick_col_generic(df2, end_cols[0])
            except KeyError:
                found = None
                for cand in end_cols[1:]:
                    try:
                        found = _pick_col_generic(df2, cand)
                        break
                    except KeyError:
                        continue
                if found is None:
                    continue
                end_col = found

            for _, row in df2.iterrows():
                bias_raw = str(row[dir_col]).strip().lower()
                if not bias_raw:
                    continue
                if bias_raw.startswith("l"):
                    bias = "long"
                elif bias_raw.startswith("s"):
                    bias = "short"
                else:
                    continue

                start = pd.to_datetime(row[start_col], errors="coerce")
                end = pd.to_datetime(row[end_col], errors="coerce")
                if pd.isna(start) or pd.isna(end):
                    continue
                start = start.normalize()
                end = end.normalize()
                records.setdefault(pair, []).append((start, end, bias))

    # Sortierung pro Pair
    for pair, lst in records.items():
        lst.sort(key=lambda x: x[0])
    return records


def cot_allows_trade(
    cot_map: Optional[Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp, str]]]],
    pair: str,
    bias: str,
    entry_time: pd.Timestamp,
) -> bool:
    """Prüft, ob für pair/bias zum Entry-Zeitpunkt ein passendes COT-Signal existiert."""
    if not cot_map:
        # Kein COT-Filter aktiv
        return True

    pair = pair.upper()
    bias = bias.lower()
    if pair not in cot_map:
        return False

    d = entry_time.normalize()
    for start, end, b in cot_map[pair]:
        if b == bias and start <= d <= end:
            return True
    return False


# --------------------------------------------------
# wick_diffs laden (W→H4 / 3D→H1)
# --------------------------------------------------
def load_latest_wickdiffs(base: Path, mode: str) -> pd.DataFrame:
    """
    mode: 'W' oder '3D'.
    Sucht sich die neueste wick_diffs-CSV für den Mode und normalisiert
    die Spalten auf ein gemeinsames Schema.
    """
    if mode == "W":
        sub = "W\u2192H4"
        pattern = "wick_diffs_H4_*.csv"
    else:
        sub = "3D\u2192H1"
        pattern = "wick_diffs_H1_*.csv"

    wdir = base / "outputs" / "wickdiffs" / sub
    files = sorted(wdir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"Keine wick_diffs-CSV für Mode {mode} in {wdir} gefunden.")
    path = files[-1]

    df = pd.read_csv(path)

    # Pivot-Spalten generisch identifizieren (alles, was NICHT mit 'wd_' anfängt)
    def pick_pivot(col_key: str) -> str:
        for c in df.columns:
            name = str(c).lower()
            if col_key in name and not name.startswith("wd_"):
                return c
        raise KeyError(col_key)

    first_col = pick_pivot("first_candle_time")
    second_col = pick_pivot("second_candle_time")
    low_col = pick_pivot("gap_low")
    high_col = pick_pivot("gap_high")
    width_col = pick_pivot("gap_width")
    touch_col = pick_pivot("first_touch_time")

    # Wick-Diff-Spalten
    wd_first = next(c for c in df.columns if str(c).lower().startswith("wd_first_candle_time"))
    wd_second = next(c for c in df.columns if str(c).lower().startswith("wd_second_candle_time"))
    zonelow_col = next(c for c in df.columns if str(c).lower().startswith("wd_zone_low"))
    zonehigh_col = next(c for c in df.columns if str(c).lower().startswith("wd_zone_high"))
    zonewidth_col = next(c for c in df.columns if str(c).lower().startswith("wd_zone_width"))

    pct_col = None
    for c in df.columns:
        if "wd_zone_pct" in str(c).lower():
            pct_col = c
            break

    # Zeiten nach Timestamp
    for col in [first_col, second_col, touch_col, wd_first, wd_second]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    out = pd.DataFrame({
        "pair": df["pair"].astype(str).str.upper(),
        "pivot_type": df["pivot_type"].astype(str).str.lower(),
        "pivot_first_candle_time": df[first_col],
        "pivot_second_candle_time": df[second_col],
        "pivot_gap_low": pd.to_numeric(df[low_col], errors="coerce"),
        "pivot_gap_high": pd.to_numeric(df[high_col], errors="coerce"),
        "pivot_gap_width": pd.to_numeric(df[width_col], errors="coerce"),
        "pivot_first_touch_time": df[touch_col],
        "wd_first_candle_time": df[wd_first],
        "wd_second_candle_time": df[wd_second],
        "wd_zone_low": pd.to_numeric(df[zonelow_col], errors="coerce"),
        "wd_zone_high": pd.to_numeric(df[zonehigh_col], errors="coerce"),
        "wd_zone_width": pd.to_numeric(df[zonewidth_col], errors="coerce"),
    })

    if pct_col is not None:
        out["wd_zone_rel_width"] = pd.to_numeric(df[pct_col], errors="coerce")
    else:
        out["wd_zone_rel_width"] = pd.NA

    out = out.dropna(
        subset=[
            "pivot_first_candle_time", "pivot_second_candle_time",
            "pivot_gap_low", "pivot_gap_high", "pivot_first_touch_time",
            "wd_zone_low", "wd_zone_high",
        ]
    ).reset_index(drop=True)

    out["mode"] = mode
    return out


# --------------------------------------------------
# Entry-Erkennung & SL/TP-Bestimmung
# --------------------------------------------------
def find_entry_for_zone(
    ltf: pd.DataFrame,
    row: pd.Series,
) -> Optional[Tuple[pd.Timestamp, float, float, float, float, float]]:
    """
    Sucht nach Pivot-Touch (max. ENTRY_WINDOW_DAYS) eine gültige
    Bounce-Kerze in der Wick-Diff-Zone.

    Rückgabe:
      (entry_time, entry_price, sl_price, tp_price, risk_pips, rr_ratio)
    oder None, falls kein gültiger Entry gefunden wird.
    """
    pair = str(row["pair"]).upper()
    pivot_type = str(row["pivot_type"]).lower()
    p_low = float(row["pivot_gap_low"])
    p_high = float(row["pivot_gap_high"])

    if p_high <= p_low:
        return None

    pivot_range_price = p_high - p_low
    pip = pip_size_for(pair)

    zone_low = float(row["wd_zone_low"])
    zone_high = float(row["wd_zone_high"])
    z_lo, z_hi = min(zone_low, zone_high), max(zone_low, zone_high)

    touch_time = row["pivot_first_touch_time"]
    if pd.isna(touch_time):
        return None
    touch_time = pd.Timestamp(touch_time)

    end_time = touch_time + pd.Timedelta(days=ENTRY_WINDOW_DAYS)

    df = ltf[(ltf["time"] > touch_time) & (ltf["time"] <= end_time)].reset_index(drop=True)
    if df.empty:
        return None

    for _, c in df.iterrows():
        t = c["time"]
        o = float(c["open"])
        h = float(c["high"])
        l = float(c["low"])
        cl = float(c["close"])

        touches = (h >= z_lo) and (l <= z_hi)

        if pivot_type == "long":
            # Invalidation: Kerze bohrt komplett durch die Zone (von oben nach unten)
            if touches and o >= z_hi and cl <= z_lo:
                return None

            # Gültiger Bounce:
            # Body komplett über Zone, nur unterer Docht in Zone
            if touches and o > z_hi and cl > z_hi and l <= z_hi:
                entry_time = pd.Timestamp(t)
                entry_price = cl

                # TP: Pivot-Range gespiegelt nach oben ab Pivot-High
                tp_price = p_high + pivot_range_price

                # Minimale SL-Positionen
                sl_pivot = p_low - 0.10 * pivot_range_price
                sl_50 = entry_price - MIN_SL_PIPS * pip
                sl_price = min(sl_pivot, sl_50)  # weiter unten = mehr Risiko

                risk_price = entry_price - sl_price
                if risk_price <= 0:
                    return None
                reward_price = tp_price - entry_price
                rr = reward_price / risk_price

                # Falls RR > MAX_RR: SL weiter weg schieben bis RR == MAX_RR
                if rr > MAX_RR:
                    desired_risk = reward_price / MAX_RR
                    sl_price = entry_price - desired_risk
                    risk_price = entry_price - sl_price
                    rr = reward_price / risk_price

                risk_pips = risk_price / pip
                # Mindestgröße 50 Pips erzwingen
                if risk_pips < MIN_SL_PIPS:
                    extra = (MIN_SL_PIPS - risk_pips) * pip
                    sl_price -= extra
                    risk_price = entry_price - sl_price
                    risk_pips = risk_price / pip
                    rr = reward_price / risk_price

                if rr < MIN_RR or rr > MAX_RR:
                    return None

                return entry_time, entry_price, sl_price, tp_price, risk_pips, rr

        elif pivot_type == "short":
            # Invalidation: von unten nach oben komplett durch die Zone
            if touches and o <= z_lo and cl >= z_hi:
                return None

            # Gültiger Bounce: Body komplett unter Zone, nur oberer Docht in Zone
            if touches and o < z_lo and cl < z_lo and h >= z_lo:
                entry_time = pd.Timestamp(t)
                entry_price = cl

                tp_price = p_low - pivot_range_price

                sl_pivot = p_high + 0.10 * pivot_range_price
                sl_50 = entry_price + MIN_SL_PIPS * pip
                sl_price = max(sl_pivot, sl_50)  # weiter oben = mehr Risiko

                risk_price = sl_price - entry_price
                if risk_price <= 0:
                    return None
                reward_price = entry_price - tp_price
                rr = reward_price / risk_price

                if rr > MAX_RR:
                    desired_risk = reward_price / MAX_RR
                    sl_price = entry_price + desired_risk
                    risk_price = sl_price - entry_price
                    rr = reward_price / risk_price

                risk_pips = risk_price / pip
                if risk_pips < MIN_SL_PIPS:
                    extra = (MIN_SL_PIPS - risk_pips) * pip
                    sl_price += extra
                    risk_price = sl_price - entry_price
                    risk_pips = risk_price / pip
                    rr = reward_price / risk_price

                if rr < MIN_RR or rr > MAX_RR:
                    return None

                return entry_time, entry_price, sl_price, tp_price, risk_pips, rr

    return None


# --------------------------------------------------
# Trade-Verlauf simulieren (TP/SL zuerst?)
# --------------------------------------------------
def simulate_trade_outcome(
    ltf: pd.DataFrame,
    entry_time: pd.Timestamp,
    sl_price: float,
    tp_price: float,
    direction: str,
) -> Tuple[str, Optional[pd.Timestamp]]:
    """
    Rückgabe:
        ("TP" | "SL" | "NONE", hit_time)
    SL hat Vorrang, falls TP und SL in derselben Kerze getriggert würden.
    """
    direction = direction.lower()
    df = ltf[ltf["time"] > entry_time].reset_index(drop=True)
    if df.empty:
        return "NONE", None

    for _, c in df.iterrows():
        t = c["time"]
        h = float(c["high"])
        l = float(c["low"])

        if direction == "long":
            sl_hit = l <= sl_price
            tp_hit = h >= tp_price
        else:
            sl_hit = h >= sl_price
            tp_hit = l <= tp_price

        if sl_hit and tp_hit:
            return "SL", pd.Timestamp(t)
        if sl_hit:
            return "SL", pd.Timestamp(t)
        if tp_hit:
            return "TP", pd.Timestamp(t)

    return "NONE", None


# --------------------------------------------------
# Hauptlogik pro Mode (W / 3D)
# --------------------------------------------------
def run_trades_for_mode(
    base: Path,
    mode: str,
    cot_map: Optional[Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp, str]]]],
) -> None:
    wick = load_latest_wickdiffs(base, mode)

    if mode == "W":
        ltf_dir = base / "time frame data" / "4h data"
        ltf_label = "H4"
    else:
        ltf_dir = base / "time frame data" / "1h data"
        ltf_label = "H1"

    ltf_map = find_ltf_files_map(ltf_dir)
    if not ltf_map:
        print(f"❌ Keine LTF-Dateien gefunden in {ltf_dir}")
        return

    # Cache für eingelesene OHLCs
    ohlc_cache: Dict[str, pd.DataFrame] = {}

    def get_ltf(pair: str) -> Optional[pd.DataFrame]:
        pair_u = pair.upper()
        if pair_u in ohlc_cache:
            return ohlc_cache[pair_u]
        path = ltf_map.get(pair_u)
        if not path:
            return None
        df = read_ohlc_file(path)
        if df is None or df.empty:
            return None
        ohlc_cache[pair_u] = df
        return df

    # Nur ein Trade pro Pivot
    used_pivots = set()
    trades: List[dict] = []

    wick = wick.sort_values(
        ["pair", "pivot_first_candle_time", "wd_first_candle_time"]
    ).reset_index(drop=True)

    for _, row in wick.iterrows():
        pair = str(row["pair"]).upper()
        pivot_type = str(row["pivot_type"]).lower()

        pivot_key = (
            pair,
            row["pivot_first_candle_time"],
            row["pivot_second_candle_time"],
        )
        if pivot_key in used_pivots:
            continue

        ltf = get_ltf(pair)
        if ltf is None:
            continue

        entry_info = find_entry_for_zone(ltf, row)
        if entry_info is None:
            continue

        entry_time, entry_price, sl_price, tp_price, risk_pips, rr = entry_info

        # COT-Filter GANZ zum Schluss der Validierung
        if not cot_allows_trade(cot_map, pair, pivot_type, entry_time):
            continue

        outcome, hit_time = simulate_trade_outcome(
            ltf, entry_time, sl_price, tp_price, pivot_type
        )

        pip = pip_size_for(pair)
        pivot_range_price = float(row["pivot_gap_high"]) - float(row["pivot_gap_low"])
        pivot_range_pips = pivot_range_price / pip

        if outcome == "TP":
            rr_result = rr
        elif outcome == "SL":
            rr_result = -1.0
        else:
            rr_result = 0.0  # offener Trade, nicht in Erwartungswert eingerechnet

        trades.append({
            "pair": pair,
            "mode": mode,
            "ltf": ltf_label,
            "pivot_type": pivot_type,
            "pivot_first_candle_time": row["pivot_first_candle_time"],
            "pivot_second_candle_time": row["pivot_second_candle_time"],
            "pivot_gap_low": float(row["pivot_gap_low"]),
            "pivot_gap_high": float(row["pivot_gap_high"]),
            "pivot_gap_width": pivot_range_price,
            "pivot_gap_pips": pivot_range_pips,
            "pivot_first_touch_time": row["pivot_first_touch_time"],
            "wd_first_candle_time": row["wd_first_candle_time"],
            "wd_second_candle_time": row["wd_second_candle_time"],
            "wd_zone_low": float(row["wd_zone_low"]),
            "wd_zone_high": float(row["wd_zone_high"]),
            "entry_time": entry_time,
            "entry_price": entry_price,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "risk_pips": risk_pips,
            "rr": rr,
            "outcome": outcome,
            "outcome_time": hit_time,
            "rr_result": rr_result,
        })

        used_pivots.add(pivot_key)

    if not trades:
        print(f"⚠️ Keine gültigen Trades für Mode {mode} gefunden.")
        return

    trades_df = pd.DataFrame(trades)

    # Erwartungswert & Winrate (nur abgeschlossene Trades)
    closed = trades_df[trades_df["outcome"].isin(["TP", "SL"])].copy()
    if not closed.empty:
        n_total = len(closed)
        n_win = (closed["outcome"] == "TP").sum()
        n_loss = (closed["outcome"] == "SL").sum()
        win_rate = n_win / n_total * 100
        exp_rr = closed["rr_result"].mean()
    else:
        n_total = n_win = n_loss = 0
        win_rate = float("nan")
        exp_rr = float("nan")

    out_dir = base / "outputs" / "trades"
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    trades_path = out_dir / f"trades_{mode}_{stamp}.csv"
    trades_df.to_csv(trades_path, index=False)

    # Pair-Statistik
    if not closed.empty:
        pair_stats = (
            closed.groupby("pair")
            .agg(
                n_trades=("outcome", "size"),
                wins=("outcome", lambda x: (x == "TP").sum()),
                losses=("outcome", lambda x: (x == "SL").sum()),
                avg_rr=("rr_result", "mean"),
            )
            .reset_index()
        )
        pair_stats["win_rate_%"] = pair_stats["wins"] / pair_stats["n_trades"] * 100
    else:
        pair_stats = pd.DataFrame(
            columns=["pair", "n_trades", "wins", "losses", "avg_rr", "win_rate_%"]
        )

    stats_path = out_dir / f"trades_{mode}_summary_{stamp}.csv"
    pair_stats.to_csv(stats_path, index=False)

    print(f"✅ Trades für Mode {mode}: {len(trades_df)} insgesamt, {n_total} geschlossen.")
    print(f"   Winrate (geschlossen): {win_rate:.1f}%  | Erwartungswert (RR): {exp_rr:.3f}")
    print(f"   Trades-CSV:   {trades_path}")
    print(f"   Summary-CSV:  {stats_path}")


# --------------------------------------------------
# main()
# --------------------------------------------------
def main() -> None:
    base = Path(__file__).resolve().parent

    # COT-Datei suchen
    cot_dir = base / "cot data"
    cot_path: Optional[Path] = None
    if cot_dir.exists():
        candidates = sorted(cot_dir.glob("COT*ignale*.xlsx"))
        if candidates:
            cot_path = candidates[-1]

    cot_map = None
    if cot_path and cot_path.exists():
        try:
            cot_map = load_cot_signals(cot_path)
            print(f"🔄 COT-Signale geladen aus: {cot_path}")
        except Exception as e:
            print(f"⚠️ Konnte COT-Signale nicht laden ({cot_path}): {e}")
            print("   → Trades werden OHNE COT-Filter berechnet.")
            cot_map = None
    else:
        print("⚠️ Keine COT-Excel gefunden – Trades werden OHNE COT-Filter berechnet.")

    # Weekly (W → H4)
    run_trades_for_mode(base, "W", cot_map)
    # 3D (3D → H1)
    run_trades_for_mode(base, "3D", cot_map)


if __name__ == "__main__":
    main()
