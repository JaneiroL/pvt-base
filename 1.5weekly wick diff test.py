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

# Sonder-Fixes für alte kaputte Codes (falls sie nochmal auftauchen)
SPECIAL_PAIR_FIX = {
    "OANDAG": "GBPNZD",
}

# -----------------------------------
# Hilfsfunktionen: Zeit normalisieren
# -----------------------------------
def to_naive_datetime(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce", utc=True)
    if is_datetime64tz_dtype(dt.dtype):
        dt = dt.dt.tz_convert(None)
    return dt

# -----------------------------------
# I/O: robuste CSV/Excel Leser (OHLC)
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
# Pivot-CSV laden (Weekly ODER 3D)
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

def load_pivots(path: Path, tf_tag: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # optional: first_touch_time
    first_touch_col = None
    for c in df.columns:
        if str(c).lower() == "first_touch_time":
            first_touch_col = c
            break

    # Basis-Spalten
    pair_col          = _pick_col_generic(df, "pair")
    timeframe_col     = _pick_col_generic(df, "timeframe")
    pivot_type_col    = _pick_col_generic(df, "pivot_type")
    first_candle_col  = _pick_col_generic(df, "first_candle_time")
    second_candle_col = _pick_col_generic(df, "second_candle_time")
    gap_low_col       = _pick_col_generic(df, "gap_low")
    gap_high_col      = _pick_col_generic(df, "gap_high")

    # Wick-Diff-Spalten (falls vorhanden)
    try:
        wd_low_col = _pick_col_generic(df, "wick_diff_low")
        wd_high_col = _pick_col_generic(df, "wick_diff_high")
        wick_diff_low = pd.to_numeric(df[wd_low_col], errors="coerce")
        wick_diff_high = pd.to_numeric(df[wd_high_col], errors="coerce")
    except KeyError:
        # Fallback: falls alte Pivot-CSV ohne Wick-Diff -> gesamte Gap
        wick_diff_low = pd.to_numeric(df[gap_low_col], errors="coerce")
        wick_diff_high = pd.to_numeric(df[gap_high_col], errors="coerce")

    out = pd.DataFrame({
        "pair": df[pair_col].astype(str),
        "timeframe": df[timeframe_col].astype(str),
        "pivot_type": df[pivot_type_col].astype(str).str.lower(),
        "first_candle_time": to_naive_datetime(df[first_candle_col]),
        "second_candle_time": to_naive_datetime(df[second_candle_col]),
        "gap_low": pd.to_numeric(df[gap_low_col], errors="coerce"),
        "gap_high": pd.to_numeric(df[gap_high_col], errors="coerce"),
        "wick_diff_low": wick_diff_low,
        "wick_diff_high": wick_diff_high,
        "first_touch_time": (
            to_naive_datetime(df[first_touch_col])
            if first_touch_col is not None
            else pd.NaT
        ),
    })

    # Gap-Order normalisieren
    lo = out[["gap_low", "gap_high"]].min(axis=1)
    hi = out[["gap_low", "gap_high"]].max(axis=1)
    out["gap_low"], out["gap_high"] = lo, hi

    # Wick-Diff-Order normalisieren
    wd_lo = out[["wick_diff_low", "wick_diff_high"]].min(axis=1)
    wd_hi = out[["wick_diff_low", "wick_diff_high"]].max(axis=1)
    out["wick_diff_low"], out["wick_diff_high"] = wd_lo, wd_hi

    # Pair-Code robust aus der 'pair'-Spalte ziehen
    def extract_pair6(x: str) -> str:
        up = re.sub(r"[^A-Z]", "", str(x).upper())
        # Sonderfix zuerst
        for bad, real in SPECIAL_PAIR_FIX.items():
            if bad in up:
                return real
        # Direkt nach 28er-Pair suchen
        for p in PAIRS_28:
            if p in up:
                return p
        # generischer 6-Letter-Fallback
        m = re.search(r"([A-Z]{6})", up)
        return m.group(1) if m else up[:6] or str(x)

    out["pair6"] = out["pair"].apply(extract_pair6)

    # Filter auf gewünschtes TF
    mask = out["timeframe"].str.upper().str.contains(tf_tag.upper())
    out = out[mask].dropna(
        subset=[
            "first_candle_time",
            "second_candle_time",
            "gap_low",
            "gap_high",
            "wick_diff_low",
            "wick_diff_high",
        ]
    ).reset_index(drop=True)

    return out

# -----------------------------------
# Utils
# -----------------------------------
def any_touch_between(
    df: pd.DataFrame,
    low: float,
    high: float,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
) -> bool:
    if pd.isna(end_time):
        seg = df[df["time"] > start_time]
    else:
        seg = df[(df["time"] > start_time) & (df["time"] <= end_time)]

    if seg.empty:
        return False
    lo = min(low, high)
    hi = max(low, high)
    return bool(((seg["high"] >= lo) & (seg["low"] <= hi)).any())

def first_last_time(df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    return df["time"].iloc[0], df["time"].iloc[-1]

def pair_code_from_str(s: str) -> str:
    """
    Holt den 6-Letter-Paircode aus einem Dateinamen oder Text.
    Speziell abgestimmt auf:
      - OANDA_GBPNZD_merged.csv
      - OANDA_GBPAUD, 1W_1234.csv
    """
    txt = str(s)
    # 1) Explizit OANDA_{PAIR}
    m = re.search(r"OANDA_([A-Z]{6})", txt.upper().replace(" ", ""))
    if m:
        code = m.group(1)
    else:
        up = re.sub(r"[^A-Z]", "", txt.upper())
        # Sonderfix zuerst
        for bad, real in SPECIAL_PAIR_FIX.items():
            if bad in up:
                return real
        # nach 28er-Paar suchen
        for p in PAIRS_28:
            if p in up:
                return p
        m2 = re.search(r"([A-Z]{6})", up)
        code = m2.group(1) if m2 else up[:6] or txt

    # wenn es in SPECIAL_PAIR_FIX gemappt ist, anwenden
    return SPECIAL_PAIR_FIX.get(code, code)

def find_ltf_files_map(ltf_dir: Path) -> dict[str, Path]:
    """
    Mappt 6-Letter-Paircode -> Pfad.
    Nutzt direkt das OANDA_{PAIR}_ Muster, damit alle 28 Paare sicher erkannt werden.
    """
    mp: dict[str, Path] = {}
    if not ltf_dir.exists():
        return mp
    for p in ltf_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".csv", ".xlsx"}:
            code = pair_code_from_str(p.name)
            if len(code) == 6 and code.upper() in PAIRS_28 and code not in mp:
                mp[code] = p
    return mp

# -----------------------------------
# Kernlogik: pro Pivot scannen
# -----------------------------------
def scan_pair_for_pivot(
    ltf: pd.DataFrame,
    pair: str,
    pivot_row: pd.Series,
    ltf_label: str,
    days_window: int,
    buffer_hours_before_touch: int,
) -> list[dict]:
    ptype = pivot_row["pivot_type"]  # "long" / "short"
    t1 = pd.Timestamp(pivot_row["first_candle_time"])
    t2 = pd.Timestamp(pivot_row["second_candle_time"])
    w_touch = pivot_row.get("first_touch_time", pd.NaT)

    # große Pivot-Gap
    lo_w = float(pivot_row["gap_low"])
    hi_w = float(pivot_row["gap_high"])
    lo_w, hi_w = (min(lo_w, hi_w), max(lo_w, hi_w))
    width_w = hi_w - lo_w
    if width_w <= 0:
        return []

    # HTF-Wick-Diff-Bereich
    wd_lo = float(pivot_row["wick_diff_low"])
    wd_hi = float(pivot_row["wick_diff_high"])
    wd_lo, wd_hi = (min(wd_lo, wd_hi), max(wd_lo, wd_hi))

    # Zeitfenster im LTF
    win_start = t1
    win_end = t1 + pd.Timedelta(days=days_window, hours=23, minutes=59, seconds=59)

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
    TOL = 1e-6

    for i in range(len(dfw) - 1):
        col1, col2 = col[i], col[i + 1]

        if ptype == "short":
            # LTF-Short-Pivot-Kombi (bull -> bear) im oberen Bereich
            if not (col1 == 1 and col2 == -1):
                continue
            z_lo = float(min(h[i], h[i + 1]))
            z_hi = float(max(h[i], h[i + 1]))
        elif ptype == "long":
            # LTF-Long-Pivot-Kombi (bear -> bull) im unteren Bereich
            if not (col1 == -1 and col2 == 1):
                continue
            z_lo = float(min(l[i], l[i + 1]))
            z_hi = float(max(l[i], l[i + 1]))
        else:
            continue

        # 1) Zone muss komplett im großen Pivot-Gap liegen
        if not (z_lo >= lo_w and z_hi <= hi_w):
            continue

        # 2) Alignment mit HTF-Wick-Diff:
        #    - Standard: komplett im Wick-Diff-Bereich
        #    - Sonderfall Long: eine Grenze == wd_hi
        #    - Sonderfall Short: eine Grenze == wd_lo
        inside_wd = (z_lo >= wd_lo and z_hi <= wd_hi)

        if ptype == "long":
            match_edge = (
                abs(z_lo - wd_hi) <= TOL or
                abs(z_hi - wd_hi) <= TOL
            )
        elif ptype == "short":
            match_edge = (
                abs(z_lo - wd_lo) <= TOL or
                abs(z_hi - wd_lo) <= TOL
            )
        else:
            match_edge = False

        if not (inside_wd or match_edge):
            continue

        # 3) Breite ≤ 19% der großen Pivot-Range
        z_width = z_hi - z_lo
        if z_width <= 0:
            continue
        if (z_width / width_w) > MAX_REL_WIDTH:
            continue

        t_a = pd.Timestamp(t[i])
        t_b = pd.Timestamp(t[i + 1])

        # 4) Unberührt-Regel
        if pd.notna(w_touch):
            w_touch_ts = pd.Timestamp(w_touch)
            if buffer_hours_before_touch > 0:
                buffer_end = w_touch_ts - pd.Timedelta(hours=buffer_hours_before_touch)
            else:
                buffer_end = w_touch_ts

            if buffer_end > t_b:
                end_check = buffer_end
                if any_touch_between(ltf, z_lo, z_hi, t_b, end_check):
                    continue
        else:
            end_check = data_end
            if any_touch_between(ltf, z_lo, z_hi, t_b, end_check):
                continue

        results.append({
            "pair": pair,
            "pivot_type": ptype,
            "weekly_first_candle_time": t1,
            "weekly_second_candle_time": t2,
            "weekly_gap_low": lo_w,
            "weekly_gap_high": hi_w,
            "weekly_gap_width": width_w,
            "ltf": ltf_label,
            "wd_first_candle_time": t_a,
            "wd_second_candle_time": t_b,
            "wd_zone_low": z_lo,
            "wd_zone_high": z_hi,
            "wd_zone_width": z_width,
            "wd_zone_pct_of_weekly": z_width / width_w,
            "weekly_first_touch_time": (pd.Timestamp(w_touch) if pd.notna(w_touch) else pd.NaT),
            "pending_until_weekly_touch": pd.isna(w_touch),
        })

    return results

# -----------------------------------
# Modus ausführen (W oder 3D)
# -----------------------------------
def run_mode(
    base: Path,
    mode: str,
    piv_dir: Path,
    pattern: str,
    ltf_dir: Path,
    ltf_label: str,
    days_win: int,
    tf_tag: str,
    out_sub: str,
    buffer_hours_before_touch: int,
) -> None:
    if not piv_dir.exists():
        print(f"❌ Pivot-Ordner nicht gefunden: {piv_dir}")
        return

    pivot_files = sorted(piv_dir.glob(pattern))
    if not pivot_files:
        print(f"❌ Keine Pivot-CSV in {piv_dir} gefunden.")
        return

    piv_path = pivot_files[-1]
    print(f"\n🔄 Verwende Pivot-CSV ({mode}): {piv_path}")

    pivots = load_pivots(piv_path, tf_tag)
    if pivots.empty:
        print(f"⚠️ Keine gültigen Pivots gefunden für {mode}.")
        return

    ltf_map = find_ltf_files_map(ltf_dir)
    if not ltf_map:
        print(f"❌ Keine LTF-Dateien in {ltf_dir} gefunden.")
        return

    all_rows: list[dict] = []
    skipped: list[str] = []

    for _, row in pivots.iterrows():
        pair6_raw = str(row["pair6"])
        pair6 = SPECIAL_PAIR_FIX.get(pair6_raw, pair6_raw)

        ltf_path = ltf_map.get(pair6)
        if not ltf_path:
            skipped.append(pair6)
            continue

        ltf_df = read_ohlc_file(ltf_path)
        if ltf_df is None or ltf_df.empty:
            skipped.append(pair6)
            continue

        all_rows += scan_pair_for_pivot(
            ltf_df, pair6, row, ltf_label, days_win, buffer_hours_before_touch
        )

    if not all_rows:
        print(f"⚠️ Keine gültigen Wick-Differences gefunden für {mode} "
              "(Richtung, Pivot-Gap, Wick-Diff-Alignment, ≤19%, Unberührtheit).")
        if skipped:
            print("ℹ️ Übersprungen (fehlende/ungültige LTF-Datei):",
                  ", ".join(sorted(set(skipped))))
        return

    out = pd.DataFrame(all_rows)

    # Duplikate entfernen
    dedup_cols = [
        "pair", "pivot_type",
        "weekly_first_candle_time", "weekly_second_candle_time",
        "wd_first_candle_time", "wd_second_candle_time",
        "wd_zone_low", "wd_zone_high",
    ]
    out = out.drop_duplicates(subset=dedup_cols, keep="first").reset_index(drop=True)

    # Zeiten schön formatieren
    time_cols = [
        "weekly_first_candle_time", "weekly_second_candle_time",
        "wd_first_candle_time", "wd_second_candle_time",
        "weekly_first_touch_time",
    ]
    for col in time_cols:
        out[col] = pd.to_datetime(out[col]).dt.strftime("%Y-%m-%d %H:%M")

    # Spaltennamen für 3D anpassen
    if mode == "3D":
        rename_map = {
            "weekly_first_candle_time":   "3day_first_candle_time",
            "weekly_second_candle_time":  "3day_second_candle_time",
            "weekly_gap_low":             "3day_gap_low",
            "weekly_gap_high":            "3day_gap_high",
            "weekly_gap_width":           "3day_gap_width",
            "weekly_first_touch_time":    "3day_first_touch_time",
            "pending_until_weekly_touch": "pending_until_3day_touch",
        }
        existing_renames = {k: v for k, v in rename_map.items() if k in out.columns}
        out = out.rename(columns=existing_renames)

    # Output-Pfad
    out_dir = base / "outputs" / "wickdiffs" / out_sub
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"wick_diffs_{ltf_label}_{stamp}.csv"
    out.to_csv(out_path, index=False)

    print(f"✅ Gefundene (UNBERÜHRTE) Wick-Differences ({mode}): {len(out)}")
    with pd.option_context("display.width", 220, "display.max_columns", None):
        print(out.head(PREVIEW_ROWS).to_string(index=False))
    print(f"💾 Ergebnisse gespeichert in: {out_path.resolve()}")

    if skipped:
        print("ℹ️ Übersprungen (keine/ungültige oder fehlende LTF-Datei):",
              ", ".join(sorted(set(skipped))))

# -----------------------------------
# Main – läuft automatisch W UND 3D
# -----------------------------------
def main() -> None:
    base = Path(__file__).resolve().parent

    # Weekly: W → H4 mit 24h-Puffer
    run_mode(
        base=base,
        mode="W",
        piv_dir=base / "outputs" / "pivots" / "W",
        pattern="pivots_gap_ALL_W_*.csv",
        ltf_dir=base / "time frame data" / "4h data",
        ltf_label="H4",
        days_win=13,
        tf_tag="W",
        out_sub="W→H4",
        buffer_hours_before_touch=24,
    )

    # 3D: 3D → H1 ohne Puffer
    run_mode(
        base=base,
        mode="3D",
        piv_dir=base / "outputs" / "pivots" / "3D",
        pattern="pivots_gap_ALL_3D_*.csv",
        ltf_dir=base / "time frame data" / "1h data",
        ltf_label="H1",
        days_win=5,
        tf_tag="3D",
        out_sub="3D→H1",
        buffer_hours_before_touch=0,
    )

if __name__ == "__main__":
    main()
