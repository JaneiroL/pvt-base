from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, List, Tuple, Set
import re
import warnings

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64tz_dtype

# -----------------------------------
# Globale Einstellungen
# -----------------------------------
RR_MIN = 0.90
RR_MAX = 1.49
MIN_SL_PIPS = 50

# Finaler Filter (nur planned_rr)
RR_FILTER_MIN = 0.90
RR_FILTER_MAX = 1.30

PAIRS_28 = {
    "AUDCAD","AUDCHF","AUDJPY","AUDNZD","AUDUSD",
    "CADCHF","CADJPY",
    "CHFJPY",
    "EURAUD","EURCAD","EURCHF","EURGBP","EURJPY","EURNZD","EURUSD",
    "GBPAUD","GBPCAD","GBPCHF","GBPJPY","GBPUSD","GBPNZD",
    "NZDCAD","NZDCHF","NZDJPY","NZDUSD",
    "USDCAD","USDCHF","USDJPY",
}

SPECIAL_PAIR_FIX = {
    "OANDAG": "GBPNZD",
}

warnings.filterwarnings(
    "ignore",
    message=".*is_datetime64tz_dtype is deprecated.*",
    category=FutureWarning,
)

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

def _parse_float_01(raw: str, default: float) -> float:
    if raw is None:
        return float(default)
    s = str(raw).strip().lower()
    if s == "":
        return float(default)
    if s in {"non", "none", "no", "off", "0"}:
        return 0.0
    s = s.replace(",", ".")
    try:
        v = float(s)
    except Exception:
        print(f"⚠️ Konnte '{raw}' nicht als Zahl lesen. Nutze Default={default}.")
        return float(default)

    if not np.isfinite(v) or v < 0:
        print(f"⚠️ Ungültig '{raw}'. Nutze Default={default}.")
        return float(default)

    v_rounded = round(v, 1)
    if abs(v_rounded - v) > 1e-9:
        print(f"ℹ️ Runde '{v}' auf 0.1-Schritte -> {v_rounded}.")
    return float(v_rounded)

def _parse_yes_no_default_yes(raw: str) -> bool:
    s = (raw or "").strip().lower()
    if s == "":
        return True
    if s in {"n", "no", "nein", "0", "off"}:
        return False
    if s in {"y", "yes", "ja", "1", "on"}:
        return True
    return True

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
        if p.is_file() and p.suffix.lower() in {".csv", ".xlsx"}:
            code = pair_code_from_str(p.name).upper()
            if len(code) == 6 and code in PAIRS_28 and code not in mp:
                mp[code] = p
    return mp

# -----------------------------------
# Generische Spaltenwahl
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

def _pick_any(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    for n in names:
        try:
            return _pick_col_generic(df, n)
        except KeyError:
            continue
    return None

def _latest_wickdiff_csv_in_dir(d: Path) -> Optional[Path]:
    if not d.exists():
        return None
    cands = []
    for p in d.glob("*.csv"):
        name = p.name.lower()
        if ("wick" in name and "diff" in name) or ("wickdiff" in name):
            cands.append(p)
    if not cands:
        return None
    return max(cands, key=lambda x: x.stat().st_mtime)

# -----------------------------------
# Wickdiff-Metadaten aus Ordnernamen
# -----------------------------------
def _parse_wd_meta_from_dirname(dirname: str) -> Tuple[Optional[str], str, Optional[str]]:
    name_up = dirname.upper()
    tokens = [t for t in re.split(r"[ _\\-→]", name_up) if t]

    mode = None
    if "2W" in tokens:
        mode = "2W"
    elif "3D" in tokens:
        mode = "3D"
    elif "M" in tokens or "MONTHLY" in tokens:
        mode = "M"
    elif "W" in tokens or "WEEKLY" in tokens:
        mode = "W"

    variant = "ALL"
    if "OUTSIDE" in tokens:
        variant = "OUTSIDE"
    elif "INNER" in tokens or "INSIDE" in tokens or "VAR3" in tokens:
        variant = "INNER"
    elif "ALL" in tokens:
        variant = "ALL"

    # WD_LTF: explizite Tokens bevorzugen, aber HTF-Mode selbst ignorieren
    wd_ltf = None
    tf_map = {
        "H1": "H1", "1H": "H1",
        "H4": "H4", "4H": "H4",
        "H12": "H12", "12H": "H12",
        "D1": "D1", "1D": "D1",
        "3D": "3D",
    }
    for t in tokens:
        if t in tf_map:
            cand = tf_map[t]
            if mode is not None and cand == mode:
                continue
            wd_ltf = cand
            break

    return mode, variant, wd_ltf

def discover_wickdiff_files(base: Path) -> List[Tuple[str, str, str, Path]]:
    root = base / "outputs" / "wickdiffs"
    results: List[Tuple[str, str, str, Path]] = []
    if not root.exists():
        print(f"❌ Wickdiff-Root fehlt: {root}")
        return results

    for d in root.iterdir():
        if not d.is_dir():
            continue
        mode, variant, wd_ltf = _parse_wd_meta_from_dirname(d.name)
        if mode is None:
            continue
        if wd_ltf is None:
            default_map = {"3D": "H1", "W": "H4", "2W": "D1", "M": "3D"}
            wd_ltf = default_map.get(mode, "H1")

        csv_path = _latest_wickdiff_csv_in_dir(d)
        if csv_path is None:
            continue
        results.append((mode, variant, wd_ltf, csv_path))

    if not results:
        print("⚠️ Keine Wickdiff-Dateien gefunden.")
    return results

def load_wickdiff_csv(path: Path, mode: str, variant: str, wd_ltf: str) -> pd.DataFrame:
    df_raw = pd.read_csv(path)
    if df_raw.empty:
        return pd.DataFrame()
    df_raw.columns = [str(c) for c in df_raw.columns]

    pair_col = _pick_any(df_raw, ["pair", "symbol", "instrument", "pair6"])
    ptype_col = _pick_any(df_raw, ["pivot_type", "direction", "side"])
    if pair_col is None or ptype_col is None:
        print(f"⚠️ Wickdiff-Datei ohne pair/pivot_type: {path}")
        return pd.DataFrame()

    gap_low_col = _pick_any(df_raw, ["pivot_low","gap_low","htf_gap_low","weekly_gap_low","3day_gap_low","2w_gap_low","m_gap_low"])
    gap_high_col = _pick_any(df_raw, ["pivot_high","gap_high","htf_gap_high","weekly_gap_high","3day_gap_high","2w_gap_high","m_gap_high"])

    first_candle_col = _pick_any(df_raw, ["pivot_first_time","first_candle_time","htf_first_candle_time","weekly_first_candle_time","3day_first_candle_time","2w_first_candle_time","m_first_candle_time"])
    second_candle_col = _pick_any(df_raw, ["pivot_second_time","second_candle_time","htf_second_candle_time","weekly_second_candle_time","3day_second_candle_time","2w_second_candle_time","m_second_candle_time"])

    wd_first_col = _pick_any(df_raw, ["wd_first_candle_time", "wd_first_time"])
    wd_second_col = _pick_any(df_raw, ["wd_second_candle_time", "wd_second_time"])
    wd_low_col = _pick_any(df_raw, ["wd_zone_low", "wd_low"])
    wd_high_col = _pick_any(df_raw, ["wd_zone_high", "wd_high"])

    if None in (gap_low_col, gap_high_col, first_candle_col, second_candle_col, wd_first_col, wd_second_col, wd_low_col, wd_high_col):
        print(f"⚠️ Wickdiff-Datei ohne erwartete Spalten (Pivot/WD): {path}")
        return pd.DataFrame()

    touch_col = _pick_any(df_raw, ["pivot_touch_time","first_touch_time","htf_first_touch_time","weekly_first_touch_time","3day_first_touch_time","2w_first_touch_time","m_first_touch_time"])

    htf_wd_low_col = _pick_any(df_raw, ["htf_wick_diff_low", "wick_diff_low"])
    htf_wd_high_col = _pick_any(df_raw, ["htf_wick_diff_high", "wick_diff_high"])

    df = pd.DataFrame(
        {
            "pair_raw": df_raw[pair_col].astype(str),
            "pivot_type": df_raw[ptype_col].astype(str).str.lower().str.strip(),
            "pivot_low": pd.to_numeric(df_raw[gap_low_col], errors="coerce"),
            "pivot_high": pd.to_numeric(df_raw[gap_high_col], errors="coerce"),
            "pivot_first_time": to_naive_datetime(df_raw[first_candle_col]),
            "pivot_second_time": to_naive_datetime(df_raw[second_candle_col]),
            "pivot_touch_time": (to_naive_datetime(df_raw[touch_col]) if touch_col else pd.NaT),
            "wd_first_time": to_naive_datetime(df_raw[wd_first_col]),
            "wd_second_time": to_naive_datetime(df_raw[wd_second_col]),
            "wd_low": pd.to_numeric(df_raw[wd_low_col], errors="coerce"),
            "wd_high": pd.to_numeric(df_raw[wd_high_col], errors="coerce"),
        }
    )

    if htf_wd_low_col and htf_wd_high_col:
        df["htf_wick_diff_low"] = pd.to_numeric(df_raw[htf_wd_low_col], errors="coerce")
        df["htf_wick_diff_high"] = pd.to_numeric(df_raw[htf_wd_high_col], errors="coerce")
    else:
        df["htf_wick_diff_low"] = np.nan
        df["htf_wick_diff_high"] = np.nan

    df["pair6"] = df["pair_raw"].apply(pair_code_from_str).str.upper()
    df["mode"] = mode
    df["wd_variant"] = variant
    df["wd_source_ltf"] = wd_ltf
    df["wickdiff_file"] = path.name  # ✅ für spätere Transparenz

    df = df.dropna(
        subset=[
            "pivot_low","pivot_high","pivot_first_time","pivot_second_time",
            "wd_low","wd_high","wd_first_time","wd_second_time"
        ]
    ).reset_index(drop=True)

    return df

# -----------------------------------
# Vola Daily Index laden
# -----------------------------------
def load_vola_index(base: Path) -> Dict[str, pd.DataFrame]:
    vola_root = base / "time frame data" / "vola index"
    xlsx_path = vola_root / "Vola daily.xlsx"
    if not xlsx_path.exists():
        print(f"⚠️ Keine Vola-Datei gefunden: {xlsx_path} – benutze statische {MIN_SL_PIPS} Pips.")
        return {}

    try:
        sheets = pd.read_excel(xlsx_path, sheet_name=None, engine="openpyxl")
    except Exception as e:
        print(f"⚠️ Fehler beim Laden der Vola-Datei {xlsx_path}: {e} – benutze statische {MIN_SL_PIPS} Pips.")
        return {}

    vola_map: Dict[str, pd.DataFrame] = {}
    for sheet_name, df in sheets.items():
        pair6 = pair_code_from_str(sheet_name).upper()
        if pair6 not in PAIRS_28:
            continue

        date_col = _pick_any(df, ["date", "time", "datum", "day", "datetime"])
        vola_col = _pick_any(
            df,
            [
                "durchschnittliche tageskerzengröße",
                "durchschnittliche_tageskerzengröße",
                "avg_daily_range",
                "vola",
                "tageskerze",
                "daily",
                "pips",
            ],
        )
        if date_col is None or vola_col is None:
            continue

        tmp = df[[date_col, vola_col]].copy()
        tmp["date"] = pd.to_datetime(tmp[date_col], errors="coerce").dt.date
        tmp["vola"] = pd.to_numeric(tmp[vola_col], errors="coerce")
        tmp = tmp.dropna(subset=["date", "vola"]).sort_values("date").reset_index(drop=True)
        if tmp.empty:
            continue

        vola_map[pair6] = tmp[["date", "vola"]]

    if not vola_map:
        print(f"⚠️ Konnte keine gültigen Vola-Sheets in {xlsx_path} finden – benutze statische {MIN_SL_PIPS} Pips.")
    return vola_map

# -----------------------------------
# Volatility Factor Pips laden (Multiplikator)
# -----------------------------------
def _find_weekly_factor_file(base: Path) -> Optional[Path]:
    cands = [
        base / "time frame data" / "Volatility Faktor Pips" / "Volatility_Faktor_Pips.xlsx",
        base / "time frame data" / "Volatility Faktor Pips" / "Volatility_Faktor_Pips.XLSX",
        base / "time frame data" / "volatility faktor pips" / "Volatility_Faktor_Pips.xlsx",
        base / "Volatility_Faktor_Pips.xlsx",
        base / "time frame data" / "Volatility_Faktor_Pips.xlsx",
    ]
    for p in cands:
        if p.exists():
            return p

    tf_root = base / "time frame data"
    if tf_root.exists():
        for p in tf_root.rglob("Volatility_Faktor_Pips.xlsx"):
            return p
        for p in tf_root.rglob("Volatility_Faktor_Pips.XLSX"):
            return p
    return None

def load_weekly_volatility_factors(base: Path) -> Dict[str, float]:
    xlsx = _find_weekly_factor_file(base)
    if xlsx is None:
        print("⚠️ Volatility_Faktor_Pips.xlsx nicht gefunden – Faktor=1.0.")
        return {}

    try:
        df = pd.read_excel(xlsx, sheet_name=0, engine="openpyxl")
    except Exception as e:
        print(f"⚠️ Konnte Volatility_Faktor_Pips.xlsx nicht laden ({xlsx}): {e} – Faktor=1.0.")
        return {}

    if df.empty:
        print(f"⚠️ Volatility_Faktor_Pips.xlsx ist leer – Faktor=1.0.")
        return {}

    df.columns = [str(c) for c in df.columns]

    pair_col = _pick_any(df, ["currency pair", "pair", "symbol", "instrument"])
    fac_col = _pick_any(df, ["volatility factor", "factor", "vola factor", "multiplier"])

    if pair_col is None:
        pair_col = df.columns[0]

    if fac_col is None:
        if len(df.columns) >= 4:
            fac_col = df.columns[3]
        else:
            print(f"⚠️ Volatility_Faktor_Pips.xlsx hat keine 4 Spalten – Faktor=1.0.")
            return {}

    out: Dict[str, float] = {}
    for _, r in df.iterrows():
        pair6 = pair_code_from_str(r[pair_col]).upper()
        if pair6 not in PAIRS_28:
            continue
        fac = pd.to_numeric(r[fac_col], errors="coerce")
        if pd.isna(fac) or not np.isfinite(fac) or fac <= 0:
            continue
        out[pair6] = float(fac)

    if not out:
        print(f"⚠️ Keine gültigen Faktoren gefunden in: {xlsx} – Faktor=1.0.")
    else:
        print(f"✅ Volatility-Faktoren geladen: {xlsx}")
    return out

# -----------------------------------
# Dynamische MIN_SL Pips je HTF-Mode (0.1 Steps)
# -----------------------------------
def _sum_last_days_fractional(hist: pd.DataFrame, days: float) -> Optional[float]:
    if days is None:
        return None
    days = float(days)
    if not np.isfinite(days) or days <= 0:
        return None

    full = int(np.floor(days + 1e-12))
    frac = float(days - full)

    need = full + (1 if frac > 1e-12 else 0)
    if need <= 0:
        return None
    if len(hist) < need:
        return None

    tail = hist.tail(need)["vola"].to_numpy(dtype=float)  # oldest->newest
    tail_newest_first = tail[::-1]  # newest->older

    weights = np.ones(need, dtype=float)
    if frac > 1e-12:
        weights[-1] = frac

    val = float(np.sum(tail_newest_first * weights))
    if not np.isfinite(val) or val <= 0:
        return None
    return val

def get_min_sl_pips_from_vola_by_mode(
    vola_map: Dict[str, pd.DataFrame],
    pair6: str,
    entry_time: pd.Timestamp,
    mode: str,
    factors: Dict[str, float],
    vola_days_by_mode: Dict[str, float],
    apply_factor_by_mode: Dict[str, bool],
) -> float:
    if not vola_map or pd.isna(entry_time):
        return MIN_SL_PIPS

    pair6 = pair6.upper()
    df = vola_map.get(pair6)
    if df is None or df.empty:
        return MIN_SL_PIPS

    entry_date = entry_time.date()
    hist = df[df["date"] < entry_date]
    if hist.empty:
        return MIN_SL_PIPS

    mode = str(mode).upper().strip()
    days = vola_days_by_mode.get(mode) if vola_days_by_mode else None

    if days is None or float(days) <= 0:
        return MIN_SL_PIPS

    val = _sum_last_days_fractional(hist, float(days))
    if val is None:
        return MIN_SL_PIPS

    if apply_factor_by_mode and apply_factor_by_mode.get(mode, False):
        fac = float(factors.get(pair6, 1.0)) if factors else 1.0
        out = float(val * fac)
        return out if np.isfinite(out) and out > 0 else MIN_SL_PIPS

    return float(val)

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
        hit_tp = (h >= tp_level) if direction == "long" else (l <= tp_level)

        if touched_wd:
            return False
        if hit_tp:
            return True

    return False

def is_outside_htf_zone(row: pd.Series, pivot_low: float, pivot_high: float) -> bool:
    if "htf_wick_diff_low" not in row or "htf_wick_diff_high" not in row:
        return True
    wl = row["htf_wick_diff_low"]
    wh = row["htf_wick_diff_high"]
    if pd.isna(wl) or pd.isna(wh):
        return True

    pl, ph = float(min(pivot_low, pivot_high)), float(max(pivot_low, pivot_high))
    wl, wh = float(min(wl, wh)), float(max(wl, wh))
    zl, zh = float(min(row["wd_low"], row["wd_high"])), float(max(row["wd_low"], row["wd_high"]))

    inside_pivot = (zl >= pl) and (zh <= ph)
    outside_wd = (zh <= wl) or (zl >= wh)
    return inside_pivot and outside_wd

# -----------------------------------
# Entry / TP / SL / Simulation
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

        # Cross-Body Kill
        if direction == "long":
            if (o > hi and c < lo) or (c > hi and o < lo):
                return None
        else:
            if (o < lo and c > hi) or (c < lo and o > hi):
                return None

        # Close inside zone => Zone invalid
        if lo <= c <= hi:
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
    min_sl_pips: float,
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
    min_sl_dist = float(min_sl_pips) * pip

    sl_dist = max(sl_dist_baseline, min_sl_dist, tp_dist / RR_MAX)
    planned_rr_pos = tp_dist / sl_dist
    if planned_rr_pos < RR_MIN or planned_rr_pos > RR_MAX:
        return None, None, None

    sl_price = (entry_price - sl_dist) if direction == "long" else (entry_price + sl_dist)
    return tp_price, sl_price, float(planned_rr_pos)

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
# NEW: WD-LTF Auswahl Parsing
# -----------------------------------
def _parse_wd_ltf_selection(raw: str, available: Set[str]) -> Set[str]:
    if not raw:
        return set(available)  # Enter => ALL
    s = raw.strip().upper().replace(" ", "")
    if s in {"ALL", "*"}:
        return set(available)
    parts = [p for p in s.split(",") if p]
    out = set()
    for p in parts:
        out.add(p)
    # nur was wirklich existiert
    out = {x for x in out if x in available}
    if not out:
        return set(available)
    return out

# -----------------------------------
# Run-Logic: kombiniert mehrere WD-Dateien (verschiedene wd_ltf)
# -----------------------------------
def run_trades_for_wd_dfs(
    base: Path,
    mode: str,
    variant: str,
    wd_items: List[Tuple[str, Path]],  # [(wd_ltf, wd_path), ...]
    ltf_dir: Path,
    entry_ltf_label: str,
    vola_map: Dict[str, pd.DataFrame],
    factors: Dict[str, float],
    vola_days_by_mode: Dict[str, float],
    apply_factor_by_mode: Dict[str, bool],
    pair_filter: Optional[Set[str]] = None,
    pivot_start: Optional[pd.Timestamp] = None,
    pivot_end: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    # 1) alle WD-Dateien laden und zusammenwerfen
    wds: List[pd.DataFrame] = []
    for wd_ltf, wd_path in wd_items:
        df_wd = load_wickdiff_csv(wd_path, mode, variant, wd_ltf)
        if not df_wd.empty:
            wds.append(df_wd)

    if not wds:
        print(f"⚠️ Keine Wickdiffs geladen für Mode={mode} Variant={variant}.")
        return pd.DataFrame()

    wd = pd.concat(wds, ignore_index=True)

    if pair_filter is not None:
        wd = wd[wd["pair6"].isin({p.upper() for p in pair_filter})].copy()

    if pivot_start is not None:
        wd = wd[wd["pivot_first_time"] >= pivot_start].copy()
    if pivot_end is not None:
        wd = wd[wd["pivot_first_time"] < pivot_end].copy()

    if wd.empty:
        print(f"⚠️ Keine Wick-Pivots nach Filter für Modus {mode} / {variant}.")
        return pd.DataFrame()

    ltf_map = find_ltf_files_map(ltf_dir)
    if not ltf_map:
        print(f"❌ Keine LTF-Dateien in {ltf_dir} gefunden.")
        return pd.DataFrame()

    max_days_map = {"3D": 6, "W": 14, "2W": 21, "M": 42}
    max_days = max_days_map.get(mode, 14)

    trades: List[dict] = []
    skipped_pairs: Set[str] = set()

    wd["pivot_type"] = wd["pivot_type"].astype(str).str.lower().str.strip()
    wd = wd[wd["pivot_type"].isin({"long", "short"})].copy()

    # ✅ super wichtig: wir sortieren WD-Zonen über ALLE LTFs hinweg chronologisch
    wd = wd.sort_values(
        ["pair6", "pivot_type", "pivot_first_time", "pivot_second_time", "wd_first_time", "wd_source_ltf"]
    ).reset_index(drop=True)

    pivot_groups = wd.groupby(
        ["pair6", "pivot_type", "pivot_first_time", "pivot_second_time"], sort=False
    )

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
            continue
        pivot_touch_time = pd.Timestamp(pivot_touch.iloc[0])
        end_time = pivot_touch_time + pd.Timedelta(days=max_days)

        pivot_low = float(g["pivot_low"].iloc[0])
        pivot_high = float(g["pivot_high"].iloc[0])

        pivot_traded = False

        # iterate ZONEN (auch über mehrere wd_ltf)
        for _, row in g.iterrows():
            if pivot_traded:
                break

            zone_low = float(row["wd_low"])
            zone_high = float(row["wd_high"])
            wd_entry_ltf = str(row.get("wd_source_ltf", "")).upper()
            wickdiff_file = str(row.get("wickdiff_file", ""))

            if variant.upper() == "OUTSIDE":
                if not is_outside_htf_zone(row, pivot_low, pivot_high):
                    continue

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

            entry_time = ltf_df.loc[entry_idx, "time"]
            entry_price = float(ltf_df.loc[entry_idx, "close"])

            dyn_min_sl_pips = get_min_sl_pips_from_vola_by_mode(
                vola_map=vola_map,
                pair6=pair6,
                entry_time=entry_time,
                mode=mode,
                factors=factors,
                vola_days_by_mode=vola_days_by_mode,
                apply_factor_by_mode=apply_factor_by_mode,
            )

            tp_price, sl_price, planned_rr_pos = compute_tp_sl(
                entry_price=entry_price,
                pivot_low=pivot_low,
                pivot_high=pivot_high,
                direction=direction,
                pair6=pair6,
                min_sl_pips=dyn_min_sl_pips,
            )
            if tp_price is None:
                continue

            sim = simulate_trade(
                ltf_df, entry_idx, direction, entry_price, tp_price, sl_price, planned_rr_pos
            )
            if sim is None:
                continue

            vol_setting = float(vola_days_by_mode.get(mode, 0.0)) if vola_days_by_mode else 0.0

            trades.append(
                {
                    "mode": mode,
                    "wd_variant": variant,

                    # ✅ welches WD-LTF wurde tatsächlich für Entry benutzt?
                    "wd_entry_ltf": wd_entry_ltf,

                    # ✅ ursprüngliche Spalten bleiben auch drin
                    "wd_source_ltf": wd_entry_ltf,
                    "entry_ltf": entry_ltf_label,

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
                    "planned_rr": float(planned_rr_pos),

                    # ✅ Transparenz: welche Wickdiff-CSV hat die Zone geliefert
                    "wickdiff_file": wickdiff_file,

                    "dyn_min_sl_pips": float(dyn_min_sl_pips),
                    "VOL": vol_setting,
                }
            )

            pivot_traded = True

    if skipped_pairs:
        print("ℹ️ Übersprungen (fehlende/ungültige LTF-Datei):", ", ".join(sorted(skipped_pairs)))

    df_out = pd.DataFrame(trades)

    # FINALER FILTER: nur planned_rr
    if not df_out.empty and "planned_rr" in df_out.columns:
        df_out = df_out[
            df_out["planned_rr"].between(RR_FILTER_MIN, RR_FILTER_MAX, inclusive="both")
        ].reset_index(drop=True)

    return df_out

# -----------------------------------
# Summary
# -----------------------------------
def summarize_trades(df_trades: pd.DataFrame, out_path: Path) -> None:
    if df_trades.empty:
        print(f"⚠️ Keine Trades für {out_path.name}.")
        return

    df = df_trades.copy()
    df["is_win"] = df["result"] == "win"

    rows = []
    for (pair, mode, variant, wd_entry_ltf, entry_ltf), g in df.groupby(
        ["pair", "mode", "wd_variant", "wd_entry_ltf", "entry_ltf"]
    ):
        n = len(g)
        wins = int(g["is_win"].sum())
        losses = int(n - wins)
        avg_rr_all = float(g["rr_signed"].mean())
        win_rate = float(wins / n * 100.0) if n else 0.0
        avg_rr_win = float(g.loc[g["is_win"], "rr_signed"].mean()) if wins else float("nan")
        avg_rr_loss = float(g.loc[~g["is_win"], "rr_signed"].mean()) if losses else float("nan")
        avg_planned_rr = float(g["planned_rr"].mean()) if "planned_rr" in g.columns else float("nan")

        rows.append(
            {
                "pair": pair,
                "mode": mode,
                "wd_variant": variant,
                "wd_entry_ltf": wd_entry_ltf,
                "entry_ltf": entry_ltf,
                "n_trades": n,
                "wins": wins,
                "losses": losses,
                "avg_rr": avg_rr_all,
                "win_rate_%": win_rate,
                "avg_rr_win": avg_rr_win,
                "avg_rr_loss": avg_rr_loss,
                "avg_planned_rr": avg_planned_rr,
            }
        )

    summary = (
        pd.DataFrame(rows)
        .sort_values(["pair", "mode", "wd_variant", "wd_entry_ltf", "entry_ltf"])
        .reset_index(drop=True)
    )

    n_tot = len(df)
    wins_tot = int(df["is_win"].sum())
    losses_tot = int(n_tot - wins_tot)
    win_rate_tot = float(wins_tot / n_tot * 100.0) if n_tot else 0.0
    avg_rr_tot = float(df["rr_signed"].mean()) if n_tot else float("nan")
    avg_rr_win_tot = float(df.loc[df["is_win"], "rr_signed"].mean()) if wins_tot else float("nan")
    avg_rr_loss_tot = float(df.loc[~df["is_win"], "rr_signed"].mean()) if losses_tot else float("nan")
    avg_planned_rr_tot = float(df["planned_rr"].mean()) if "planned_rr" in df.columns and n_tot else float("nan")

    summary = pd.concat(
        [
            summary,
            pd.DataFrame(
                [
                    {
                        "pair": "ALL",
                        "mode": "ALL",
                        "wd_variant": "ALL",
                        "wd_entry_ltf": "ALL",
                        "entry_ltf": "ALL",
                        "n_trades": n_tot,
                        "wins": wins_tot,
                        "losses": losses_tot,
                        "avg_rr": avg_rr_tot,
                        "win_rate_%": win_rate_tot,
                        "avg_rr_win": avg_rr_win_tot,
                        "avg_rr_loss": avg_rr_loss_tot,
                        "avg_planned_rr": avg_planned_rr_tot,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    summary.to_csv(out_path, index=False)

    print(f"\n===== Zusammenfassung für {out_path.name} =====")
    print(f"Gesamt: {n_tot} Trades | Wins: {wins_tot} | Losses: {losses_tot} | Win-Rate: {win_rate_tot:.1f}%")
    print(f"Ø RRR Gewinn: {avg_rr_win_tot:.3f} | Ø RRR Verlust: {avg_rr_loss_tot:.3f} | Erwartungswert: {avg_rr_tot:.3f}")
    print(f"Ø planned_rr (positiv): {avg_planned_rr_tot:.3f} | Filter: [{RR_FILTER_MIN:.2f} .. {RR_FILTER_MAX:.2f}]")
    print(f"💾 Summary gespeichert in: {out_path.resolve()}")

# -----------------------------------
# Helper: User-LTF normalisieren
# -----------------------------------
def normalize_ltf_input(raw: str) -> str:
    if raw is None:
        return ""
    up = raw.strip().upper()
    compact = up.replace(" ", "")
    if compact in {"", "AUTO"}:
        return "AUTO"

    syn_map = {
        "H1": "H1", "1H": "H1", "1": "H1", "60": "H1",
        "H4": "H4", "4H": "H4", "4": "H4", "240": "H4",
        "H12": "H12", "12H": "H12", "12": "H12",
        "D1": "D1", "1D": "D1", "D": "D1", "DAY": "D1", "1440": "D1",
        "3D": "3D", "D3": "3D",
    }
    return syn_map.get(compact, compact)

# -----------------------------------
# Main
# -----------------------------------
def main() -> None:
    base = Path(__file__).resolve().parent
    out_base = base / "outputs" / "trades"
    out_base.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    vola_map = load_vola_index(base)
    factors = load_weekly_volatility_factors(base)

    print("Unteres Timeframe (LTF) für Trade-Entry wählen.")
    print("  Optionen: H1, H4, H12, D1, 3D, AUTO")
    print("  Leer lassen = AUTO (3D→H1, W→H4, 2W→D1, M→3D)")
    user_raw = input("LTF-Eingabe: ")
    user_ltf_norm = normalize_ltf_input(user_raw)

    if user_ltf_norm == "AUTO":
        entry_specs = {
            "3D": (base / "time frame data" / "1h data", "H1"),
            "W":  (base / "time frame data" / "4h data", "H4"),
            "2W": (base / "time frame data" / "daily data", "D1"),
            "M":  (base / "time frame data" / "3D", "3D"),
        }
        print("➡️ AUTO-Modus aktiv – klassisches Mapping 3D→H1, W→H4, 2W→D1, M→3D.")
    else:
        tf_map = {
            "H1": (base / "time frame data" / "1h data", "H1"),
            "H4": (base / "time frame data" / "4h data", "H4"),
            "H12": (base / "time frame data" / "12h", "H12"),
            "D1": (base / "time frame data" / "daily data", "D1"),
            "3D": (base / "time frame data" / "3D", "3D"),
        }
        if user_ltf_norm not in tf_map:
            raise ValueError(f"Unbekanntes LTF '{user_ltf_norm}'. Erlaubt: H1, H4, H12, D1, 3D, AUTO.")
        print(f"➡️ Interpretiere Eingabe '{user_raw}' als Entry-LTF: {user_ltf_norm}")
        entry_specs = {mode: tf_map[user_ltf_norm] for mode in ["3D", "W", "2W", "M"]}

    # VOL pro Mode
    defaults = {"3D": 45.0, "W": 2.0, "2W": 2.0, "M": 3.5}
    print("\nVOL-Konfiguration (Daily-Vola-Tage für MIN-SL).")
    print("  Eingabe in 0,1 Schritten (z.B. 3,4).")
    print("  0 / non / none = deaktiviert -> Fallback MIN_SL_PIPS.")
    vola_days_by_mode: Dict[str, float] = {}
    for m in ["3D", "W", "2W", "M"]:
        dflt = defaults[m]
        raw = input(f"VOL Tage für {m} [Default {str(dflt).replace('.', ',')}]: ")
        vola_days_by_mode[m] = _parse_float_01(raw, dflt)

    # Faktor Toggle pro Mode
    print("\nFaktor-Multiplikator anwenden? (pro HTF-Mode)")
    print("  Enter = JA, 'n' = NEIN")
    apply_factor_by_mode: Dict[str, bool] = {}
    for m in ["3D", "W", "2W", "M"]:
        raw = input(f"Faktor auf {m} anwenden? (Enter=ja, 'n'=nein): ")
        apply_factor_by_mode[m] = _parse_yes_no_default_yes(raw)

    print("\n✅ VOL-Settings aktiv:")
    for m in ["3D", "W", "2W", "M"]:
        flag = " (× Faktor)" if apply_factor_by_mode.get(m, False) else ""
        print(f"  {m}: {vola_days_by_mode[m]:.1f} Tage{flag}")

    wd_files = discover_wickdiff_files(base)
    if not wd_files:
        return

    # ✅ NEU: WD-LTF Auswahl pro Mode
    # Sammle verfügbare WD-LTFs je Mode aus dem Wickdiff-Folder
    available_by_mode: Dict[str, Set[str]] = {m: set() for m in ["3D", "W", "2W", "M"]}
    for (m, _variant, wd_ltf, _path) in wd_files:
        if m in available_by_mode:
            available_by_mode[m].add(str(wd_ltf).upper())

    wd_use_by_mode: Dict[str, Set[str]] = {}
    print("\nWD-LTF Auswahl pro Mode (Enter = ALL). Beispiel: H1,H4")
    for m in ["3D", "W", "2W", "M"]:
        av = available_by_mode.get(m, set())
        if not av:
            wd_use_by_mode[m] = set()
            continue
        raw = input(f"WD-LTF für {m} (verfügbar: {', '.join(sorted(av))}) [Default ALL]: ")
        wd_use_by_mode[m] = _parse_wd_ltf_selection(raw, av)
        print(f"  -> {m}: nutze {', '.join(sorted(wd_use_by_mode[m]))}")

    # Indexiere Wickdiff-Files nach (mode, variant) => List[(wd_ltf, path)]
    wd_index: Dict[Tuple[str, str], List[Tuple[str, Path]]] = {}
    for (m, variant, wd_ltf, path) in wd_files:
        key = (m, variant)
        wd_index.setdefault(key, []).append((str(wd_ltf).upper(), path))

    for mode in ["3D", "W", "2W", "M"]:
        if mode not in wd_use_by_mode or not wd_use_by_mode[mode]:
            print(f"\nℹ️ Keine WD-LTF Auswahl / keine Wickdiffs für Mode {mode} – übersprungen.")
            continue

        ltf_dir, entry_ltf_label = entry_specs[mode]

        # pro Variant separat laufen, aber jeweils mit mehreren WD-LTFs gleichzeitig
        variants_here = sorted({v for (m, v) in wd_index.keys() if m == mode})
        for variant in variants_here:
            items_all = wd_index.get((mode, variant), [])
            items = [(wd_ltf, p) for (wd_ltf, p) in items_all if wd_ltf in wd_use_by_mode[mode]]
            if not items:
                continue

            print(f"\n🚀 Starte Trades für Mode={mode}, Variant={variant}, WD_LTFs={sorted(wd_use_by_mode[mode])}, Entry_LTF={entry_ltf_label}")

            trades = run_trades_for_wd_dfs(
                base=base,
                mode=mode,
                variant=variant,
                wd_items=items,
                ltf_dir=ltf_dir,
                entry_ltf_label=entry_ltf_label,
                vola_map=vola_map,
                factors=factors,
                vola_days_by_mode=vola_days_by_mode,
                apply_factor_by_mode=apply_factor_by_mode,
            )
            if trades.empty:
                continue

            wd_tag = "-".join(sorted(wd_use_by_mode[mode]))
            out_name = f"trades_{mode}_{variant}_wd{wd_tag}_entry{entry_ltf_label}_{stamp}.csv"
            out_path = out_base / out_name
            trades.to_csv(out_path, index=False)
            print(f"💾 Trades gespeichert in: {out_path.resolve()}")

            summary_path = out_base / f"summary_{mode}_{variant}_wd{wd_tag}_entry{entry_ltf_label}_{stamp}.csv"
            summarize_trades(trades, summary_path)

if __name__ == "__main__":
    main()