from __future__ import annotations

from pathlib import Path
import re
import pandas as pd
import numpy as np
from pandas.api.types import is_datetime64tz_dtype

# -----------------------------------
# Einstellungen
# -----------------------------------
MAX_REL_WIDTH = 0.19      # 19%-Grenze für Wick-Zone vs. Pivot-Range
PREVIEW_ROWS  = 20        # Zeilen in der Terminal-Vorschau

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

# -----------------------------------
# Zeit normalisieren
# -----------------------------------
def to_naive_datetime(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce", utc=True)
    if is_datetime64tz_dtype(dt.dtype):
        dt = dt.dt.tz_convert(None)
    return dt

# -----------------------------------
# I/O OHLC
# -----------------------------------
CAND_TIME  = ["time", "timestamp", "date", "datetime", "unnamed: 0"]
CAND_OPEN  = ["open", "o"]
CAND_HIGH  = ["high", "h"]
CAND_LOW   = ["low", "l"]
CAND_CLOSE = ["close", "c"]

def _pick_col(df: pd.DataFrame, cands: list[str]) -> str:
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

def read_ohlc_file(path: Path) -> pd.DataFrame | None:
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

# -----------------------------------
# Pair Extract
# -----------------------------------
def extract_pair6_from_any_text(txt: str) -> str | None:
    s = str(txt)
    s_clean = s.upper().replace(" ", "")

    m = re.search(r"OANDA_([A-Z]{6})", s_clean)
    if m:
        code = SPECIAL_PAIR_FIX.get(m.group(1), m.group(1))
        return code

    up = re.sub(r"[^A-Z]", "", s.upper())
    for p in PAIRS_28:
        if p in up:
            return p

    m2 = re.search(r"([A-Z]{6})", up)
    if m2:
        code = SPECIAL_PAIR_FIX.get(m2.group(1), m2.group(1))
        return code
    return None

def find_ltf_files_map(ltf_dir: Path) -> dict[str, Path]:
    mp: dict[str, Path] = {}
    if not ltf_dir.exists():
        return mp
    for p in ltf_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".csv", ".xlsx"}:
            code = extract_pair6_from_any_text(p.name)
            if code and len(code) == 6 and code.upper() in PAIRS_28 and code.upper() not in mp:
                mp[code.upper()] = p
    return mp

# -----------------------------------
# Pivot-CSV Loader
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

def load_pivots(path: Path, tf_tags: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)

    first_touch_col = None
    for c in df.columns:
        if str(c).lower() == "first_touch_time":
            first_touch_col = c
            break

    pair_col          = _pick_col_generic(df, "pair")
    timeframe_col     = _pick_col_generic(df, "timeframe")
    pivot_type_col    = _pick_col_generic(df, "pivot_type")
    first_candle_col  = _pick_col_generic(df, "first_candle_time")
    second_candle_col = _pick_col_generic(df, "second_candle_time")
    gap_low_col       = _pick_col_generic(df, "gap_low")
    gap_high_col      = _pick_col_generic(df, "gap_high")

    wick_low = None
    wick_high = None
    for a, b in [("wick_diff_low","wick_diff_high"), ("wd_low","wd_high"), ("wdiff_low","wdiff_high")]:
        try:
            wl = _pick_col_generic(df, a)
            wh = _pick_col_generic(df, b)
            wick_low = pd.to_numeric(df[wl], errors="coerce")
            wick_high = pd.to_numeric(df[wh], errors="coerce")
            break
        except KeyError:
            continue

    if wick_low is None or wick_high is None:
        wick_low = pd.to_numeric(df[gap_low_col], errors="coerce")
        wick_high = pd.to_numeric(df[gap_high_col], errors="coerce")

    out = pd.DataFrame({
        "pair": df[pair_col].astype(str),
        "timeframe": df[timeframe_col].astype(str),
        "pivot_type": df[pivot_type_col].astype(str).str.lower().str.strip(),
        "first_candle_time": to_naive_datetime(df[first_candle_col]),
        "second_candle_time": to_naive_datetime(df[second_candle_col]),
        "gap_low": pd.to_numeric(df[gap_low_col], errors="coerce"),
        "gap_high": pd.to_numeric(df[gap_high_col], errors="coerce"),
        "wick_diff_low": wick_low,
        "wick_diff_high": wick_high,
        "first_touch_time": (
            to_naive_datetime(df[first_touch_col]) if first_touch_col is not None else pd.NaT
        ),
    })

    out["gap_low"], out["gap_high"] = out[["gap_low","gap_high"]].min(axis=1), out[["gap_low","gap_high"]].max(axis=1)
    out["wick_diff_low"], out["wick_diff_high"] = out[["wick_diff_low","wick_diff_high"]].min(axis=1), out[["wick_diff_low","wick_diff_high"]].max(axis=1)

    out["pair6"] = out["pair"].apply(lambda x: (extract_pair6_from_any_text(x) or str(x))[:6].upper())
    out["pair6"] = out["pair6"].apply(lambda x: SPECIAL_PAIR_FIX.get(x, x))

    tf_upper = out["timeframe"].astype(str).str.upper()
    mask = False
    for tag in tf_tags:
        mask = mask | tf_upper.str.contains(tag.upper())
    out = out[mask].dropna(
        subset=["first_candle_time","second_candle_time","gap_low","gap_high","wick_diff_low","wick_diff_high"]
    ).reset_index(drop=True)

    return out

# -----------------------------------
# Utils
# -----------------------------------
def any_touch_between(df: pd.DataFrame, low: float, high: float, start_time: pd.Timestamp, end_time: pd.Timestamp) -> bool:
    seg = df[(df["time"] > start_time) & (df["time"] <= end_time)]
    if seg.empty:
        return False
    lo = min(low, high)
    hi = max(low, high)
    return bool(((seg["high"] >= lo) & (seg["low"] <= hi)).any())

def first_last_time(df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    return df["time"].iloc[0], df["time"].iloc[-1]

def compute_pivot_window_end(t1: pd.Timestamp, t2: pd.Timestamp) -> pd.Timestamp:
    """
    FIX (wie bei V1):
      Zeitfenster = komplette 1. Candle + komplette 2. Candle
      end = t2 + (t2 - t1)
    """
    if pd.isna(t1) or pd.isna(t2):
        return t2
    d = t2 - t1
    if d <= pd.Timedelta(0):
        return t2
    return t2 + d

# -----------------------------------
# Kernlogik: OUTSIDE Finder (VARIANTE 2)
#   - Preislich: Zone komplett im Pivot
#   - V2-Regel: Zone komplett OUTSIDE HTF-WickDiff
#   - Zeitlich: NUR innerhalb Candle1+Candle2
# -----------------------------------
def scan_pair_for_pivot_outside(
    ltf: pd.DataFrame,
    pair: str,
    pivot_row: pd.Series,
    ltf_label: str,
    buffer_hours_before_touch: int,
) -> list[dict]:
    ptype = str(pivot_row["pivot_type"]).lower().strip()  # "long" / "short"
    t1 = pd.Timestamp(pivot_row["first_candle_time"])
    t2 = pd.Timestamp(pivot_row["second_candle_time"])
    w_touch = pivot_row.get("first_touch_time", pd.NaT)

    lo_w = float(pivot_row["gap_low"])
    hi_w = float(pivot_row["gap_high"])
    lo_w, hi_w = min(lo_w, hi_w), max(lo_w, hi_w)
    width_w = hi_w - lo_w
    if width_w <= 0:
        return []

    wd_lo = float(pivot_row["wick_diff_low"])
    wd_hi = float(pivot_row["wick_diff_high"])
    wd_lo, wd_hi = min(wd_lo, wd_hi), max(wd_lo, wd_hi)

    # FIX: Candle1 + Candle2 (statt days_window)
    win_start = t1
    win_end = compute_pivot_window_end(t1, t2)

    dfw = ltf[(ltf["time"] >= win_start) & (ltf["time"] <= win_end)].reset_index(drop=True)
    if dfw.shape[0] < 2:
        return []

    _, data_end = first_last_time(ltf)
    results: list[dict] = []

    o = dfw["open"].to_numpy()
    h = dfw["high"].to_numpy()
    l = dfw["low"].to_numpy()
    c = dfw["close"].to_numpy()
    t = dfw["time"].to_numpy()

    col = np.where(c > o, 1, np.where(c < o, -1, 0))  # 1 bull, -1 bear, 0 doji

    for i in range(len(dfw) - 1):
        col1, col2 = col[i], col[i + 1]
        if col1 == 0 or col2 == 0:
            continue

        if ptype == "short":
            if not (col1 == 1 and col2 == -1):
                continue
            z_lo = float(min(h[i], h[i + 1]))
            z_hi = float(max(h[i], h[i + 1]))
        elif ptype == "long":
            if not (col1 == -1 and col2 == 1):
                continue
            z_lo = float(min(l[i], l[i + 1]))
            z_hi = float(max(l[i], l[i + 1]))
        else:
            continue

        # 1) Zone muss komplett im Pivot liegen
        if not (z_lo >= lo_w and z_hi <= hi_w):
            continue

        # 2) OUTSIDE-Regel: Zone komplett außerhalb HTF Wickdiff
        outside_wd = (z_hi <= wd_lo) or (z_lo >= wd_hi)
        if not outside_wd:
            continue

        # 3) Breite ≤ 19% der Pivot-Range
        z_width = z_hi - z_lo
        if z_width <= 0:
            continue
        if (z_width / width_w) > MAX_REL_WIDTH:
            continue

        t_a = pd.Timestamp(t[i])
        t_b = pd.Timestamp(t[i + 1])

        # 4) Unberührt-Regel (identisch)
        if pd.notna(w_touch):
            w_touch_ts = pd.Timestamp(w_touch)
            buffer_end = w_touch_ts - pd.Timedelta(hours=buffer_hours_before_touch) if buffer_hours_before_touch > 0 else w_touch_ts
            if buffer_end > t_b:
                if any_touch_between(ltf, z_lo, z_hi, t_b, buffer_end):
                    continue
        else:
            if any_touch_between(ltf, z_lo, z_hi, t_b, data_end):
                continue

        results.append({
            "pair": pair,
            "pivot_type": ptype,
            "htf_first_candle_time": t1,
            "htf_second_candle_time": t2,
            "htf_gap_low": lo_w,
            "htf_gap_high": hi_w,
            "htf_gap_width": width_w,
            "htf_wick_diff_low": wd_lo,
            "htf_wick_diff_high": wd_hi,
            "ltf": ltf_label,
            "wd_first_candle_time": t_a,
            "wd_second_candle_time": t_b,
            "wd_zone_low": z_lo,
            "wd_zone_high": z_hi,
            "wd_zone_width": z_width,
            "wd_zone_pct_of_htf_gap": z_width / width_w,
            "htf_first_touch_time": (pd.Timestamp(w_touch) if pd.notna(w_touch) else pd.NaT),
            "pending_until_htf_touch": pd.isna(w_touch),
            "inside_or_outside": "OUTSIDE",
        })

    return results

# -----------------------------------
# Runner (identisch – days_win bleibt im Signature, wird aber NICHT mehr genutzt)
# -----------------------------------
def run_mode(
    base: Path,
    mode_name: str,
    piv_dir: Path,
    piv_pattern: str,
    tf_tags: list[str],
    ltf_dir: Path,
    ltf_label: str,
    days_win: int,  # bleibt nur für API-Kompatibilität, wird NICHT genutzt
    out_sub: str,
    buffer_hours_before_touch: int,
) -> None:
    if not piv_dir.exists():
        print(f"❌ Pivot-Ordner nicht gefunden: {piv_dir}")
        return

    pivot_files = sorted(piv_dir.glob(piv_pattern))
    if not pivot_files:
        print(f"❌ Keine Pivot-CSV in {piv_dir} gefunden (Pattern: {piv_pattern}).")
        return

    piv_path = pivot_files[-1]
    print(f"\n🔄 Verwende Pivot-CSV ({mode_name}): {piv_path}")

    pivots = load_pivots(piv_path, tf_tags=tf_tags)
    if pivots.empty:
        print(f"⚠️ Keine gültigen Pivots gefunden für {mode_name}.")
        return

    ltf_map = find_ltf_files_map(ltf_dir)
    if not ltf_map:
        print(f"❌ Keine LTF-Dateien in {ltf_dir} gefunden.")
        return

    all_rows: list[dict] = []
    skipped: list[str] = []

    for _, row in pivots.iterrows():
        pair6 = str(row["pair6"]).upper()
        pair6 = SPECIAL_PAIR_FIX.get(pair6, pair6)

        ltf_path = ltf_map.get(pair6)
        if not ltf_path:
            skipped.append(pair6)
            continue

        ltf_df = read_ohlc_file(ltf_path)
        if ltf_df is None or ltf_df.empty:
            skipped.append(pair6)
            continue

        all_rows += scan_pair_for_pivot_outside(
            ltf=ltf_df,
            pair=pair6,
            pivot_row=row,
            ltf_label=ltf_label,
            buffer_hours_before_touch=buffer_hours_before_touch,
        )

    if not all_rows:
        print(f"⚠️ Keine gültigen OUTSIDE Wick-Differences gefunden für {mode_name} (ZEITFENSTER Candle1+Candle2).")
        if skipped:
            print("ℹ️ Übersprungen (fehlende/ungültige LTF-Datei):", ", ".join(sorted(set(skipped))))
        return

    out = pd.DataFrame(all_rows)

    dedup_cols = [
        "pair", "pivot_type",
        "htf_first_candle_time", "htf_second_candle_time",
        "wd_first_candle_time", "wd_second_candle_time",
        "wd_zone_low", "wd_zone_high",
    ]
    out = out.drop_duplicates(subset=dedup_cols, keep="first").reset_index(drop=True)

    time_cols = [
        "htf_first_candle_time", "htf_second_candle_time",
        "wd_first_candle_time", "wd_second_candle_time",
        "htf_first_touch_time",
    ]
    for col in time_cols:
        out[col] = pd.to_datetime(out[col]).dt.strftime("%Y-%m-%d %H:%M")

    out_dir = base / "outputs" / "wickdiffs" / out_sub
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"wick_diffs_{mode_name}_{ltf_label}_OUTSIDE_{stamp}.csv"
    out.to_csv(out_path, index=False)

    print(f"✅ OUTSIDE Wick-Differences ({mode_name} → {ltf_label}): {len(out)}")
    with pd.option_context("display.width", 240, "display.max_columns", None):
        print(out.head(PREVIEW_ROWS).to_string(index=False))
    print(f"💾 Gespeichert in: {out_path.resolve()}")

    if skipped:
        print("ℹ️ Übersprungen (keine/ungültige oder fehlende LTF-Datei):", ", ".join(sorted(set(skipped))))

def main() -> None:
    base = Path(__file__).resolve().parent

    run_mode(
        base=base,
        mode_name="W",
        piv_dir=base / "outputs" / "pivots" / "W",
        piv_pattern="pivots_gap_ALL_W_*.csv",
        tf_tags=["W", "1W", "WEEK"],
        ltf_dir=base / "time frame data" / "4h data",
        ltf_label="H4",
        days_win=13,
        out_sub="W→H4_OUTSIDE",
        buffer_hours_before_touch=24,
    )

    run_mode(
        base=base,
        mode_name="3D",
        piv_dir=base / "outputs" / "pivots" / "3D",
        piv_pattern="pivots_gap_ALL_3D_*.csv",
        tf_tags=["3D", "3DAY", "3-D"],
        ltf_dir=base / "time frame data" / "1h data",
        ltf_label="H1",
        days_win=5,
        out_sub="3D→H1_OUTSIDE",
        buffer_hours_before_touch=0,
    )

    run_mode(
        base=base,
        mode_name="2W",
        piv_dir=base / "outputs" / "pivots" / "2Weekly",
        piv_pattern="pivots_gap_ALL_2W_*.csv",
        tf_tags=["2W", "2WEEK", "2WEEKLY"],
        ltf_dir=base / "time frame data" / "daily data",
        ltf_label="D1",
        days_win=27,
        out_sub="2W→1D_OUTSIDE",
        buffer_hours_before_touch=48,
    )

    run_mode(
        base=base,
        mode_name="M",
        piv_dir=base / "outputs" / "pivots" / "Monthly",
        piv_pattern="pivots_gap_ALL_M_*.csv",
        tf_tags=["M", "MON", "MONTH"],
        ltf_dir=base / "time frame data" / "3D",
        ltf_label="3D",
        days_win=55,
        out_sub="M→3D_OUTSIDE",
        buffer_hours_before_touch=96,
    )

if __name__ == "__main__":
    main()
