# wick_diffs_run.py
from __future__ import annotations

from pathlib import Path
import re
import pandas as pd
import numpy as np
from pandas.api.types import is_datetime64tz_dtype

# -----------------------------------
# Einstellungen
# -----------------------------------
MAX_REL_WIDTH = 0.19      # 19%-Grenze für Wick-Zone vs. Weekly-Range
PREVIEW_ROWS  = 20        # Zeilen in der Terminal-Vorschau


# -----------------------------------
# Hilfsfunktionen: Zeit normalisieren
# -----------------------------------
def to_naive_datetime(s: pd.Series) -> pd.Series:
    """
    Parsed eine Spalte robust zu datetime und entfernt ggf. Zeitzonen-Info,
    so dass alle Werte tz-naiv sind (kein UTC-Versatz, gut vergleichbar).
    """
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
    # 1) exakte Matches
    for cand in cands:
        if cand in low:
            return low[cand]
    # 2) Teil-String-Matches
    for c in df.columns:
        lc = str(c).strip().lower()
        for cand in cands:
            if cand in lc:
                return c
    # 3) Fallback: "unnamed"
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

    # Zeit robust parsen + TZ entfernen
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
# Weekly-Pivots CSV laden
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


def load_weeklies(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    first_touch_col = None
    for c in df.columns:
        if str(c).lower() == "first_touch_time":
            first_touch_col = c
            break

    out = pd.DataFrame({
        "pair": df[_pick_col_generic(df, "pair")].astype(str),
        "timeframe": df[_pick_col_generic(df, "timeframe")].astype(str),
        "pivot_type": df[_pick_col_generic(df, "pivot_type")].astype(str).str.lower(),
        "first_candle_time": to_naive_datetime(
            df[_pick_col_generic(df, "first_candle_time")]
        ),
        "second_candle_time": to_naive_datetime(
            df[_pick_col_generic(df, "second_candle_time")]
        ),
        "gap_low": pd.to_numeric(df[_pick_col_generic(df, "gap_low")], errors="coerce"),
        "gap_high": pd.to_numeric(df[_pick_col_generic(df, "gap_high")], errors="coerce"),
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

    # 6-Letter Pair-Code extrahieren
    out["pair6"] = (
        out["pair"]
        .str.upper()
        .str.replace(r"[^A-Z]", "", regex=True)
        .str.extract(r"([A-Z]{6})", expand=False)
        .fillna(out["pair"].str.upper())
    )

    out = out[
        out["timeframe"].str.upper().str.startswith("W")
    ].dropna(
        subset=["first_candle_time", "second_candle_time", "gap_low", "gap_high"]
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
    """
    Gibt es IRGENDEINE 4h-Kerze, deren Range [low,high] schneidet,
    im Intervall (start_time, end_time] ?
    """
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


# -----------------------------------
# Kernlogik: pro Weekly-Pivot scannen
# -----------------------------------
def scan_pair_for_pivot(h4: pd.DataFrame, pair: str, pivot_row: pd.Series) -> list[dict]:
    ptype = pivot_row["pivot_type"]  # "long" / "short"
    t1 = pd.Timestamp(pivot_row["first_candle_time"])
    t2 = pd.Timestamp(pivot_row["second_candle_time"])
    w_touch = pivot_row.get("first_touch_time", pd.NaT)

    lo_w = float(pivot_row["gap_low"])
    hi_w = float(pivot_row["gap_high"])
    lo_w, hi_w = (min(lo_w, hi_w), max(lo_w, hi_w))
    width_w = hi_w - lo_w
    if width_w <= 0:
        return []

    # Zeitfenster: von Start der ersten Weekly-Kerze bis Ende der Woche der zweiten Weekly-Kerze
    win_start = t1.normalize()
    win_end = t2 + pd.Timedelta(days=4, hours=23, minutes=59, seconds=59)

    dfw = h4[(h4["time"] >= win_start) & (h4["time"] <= win_end)].reset_index(drop=True)
    if dfw.shape[0] < 2:
        return []

    _, data_end = first_last_time(h4)
    results: list[dict] = []

    o = dfw["open"].to_numpy()
    h = dfw["high"].to_numpy()
    l = dfw["low"].to_numpy()
    c = dfw["close"].to_numpy()
    t = dfw["time"].to_numpy()

    # 1 = bull, -1 = bear, 0 = doji
    col = np.where(c > o, 1, np.where(c < o, -1, 0))

    for i in range(len(dfw) - 1):
        col1, col2 = col[i], col[i + 1]

        if ptype == "short":
            # Weekly-Short → 4h bull → bear
            if not (col1 == 1 and col2 == -1):
                continue
            z_lo = float(min(h[i], h[i + 1]))
            z_hi = float(max(h[i], h[i + 1]))
        elif ptype == "long":
            # Weekly-Long → 4h bear → bull
            if not (col1 == -1 and col2 == 1):
                continue
            z_lo = float(min(l[i], l[i + 1]))
            z_hi = float(max(l[i], l[i + 1]))
        else:
            continue

        # Zone muss komplett im Weekly-Gap liegen
        if not (z_lo >= lo_w and z_hi <= hi_w):
            continue

        # Breite ≤ 19% der Weekly-Range
        z_width = z_hi - z_lo
        if (z_width / width_w) > MAX_REL_WIDTH:
            continue

        t_a = pd.Timestamp(t[i])
        t_b = pd.Timestamp(t[i + 1])

        # UNBERÜHRT-REGEL
        end_check = pd.Timestamp(w_touch) if pd.notna(w_touch) else data_end
        if any_touch_between(h4, z_lo, z_hi, t_b, end_check):
            continue

        results.append({
            "pair": pair,
            "pivot_type": ptype,
            "weekly_first_candle_time": t1,
            "weekly_second_candle_time": t2,
            "weekly_gap_low": lo_w,
            "weekly_gap_high": hi_w,
            "weekly_gap_width": width_w,
            "ltf": "H4",
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
# Pair-Dateien im H4-Ordner finden
# -----------------------------------
def pair_code_from_str(s: str) -> str:
    up = re.sub(r"[^A-Z]", "", str(s).upper())
    m = re.search(r"[A-Z]{6}", up)
    return m.group(0) if m else up[:6] or str(s)


def find_h4_files_map(h4_dir: Path) -> dict[str, Path]:
    mp: dict[str, Path] = {}
    if not h4_dir.exists():
        return mp
    for p in h4_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".csv", ".xlsx"}:
            code = pair_code_from_str(p.name)
            if len(code) == 6 and code not in mp:
                mp[code] = p
    return mp


# -----------------------------------
# Main – komplett ohne argparse,
# nutzt feste Repo-Struktur
# -----------------------------------
def main() -> None:
    base = Path(__file__).resolve().parent

    # 1) Weekly-Pivot CSV finden (neueste Datei)
    weeklies_dir = base / "outputs" / "pivots" / "W"
    if not weeklies_dir.exists():
        print(f"❌ Weekly-Pivots-Ordner nicht gefunden: {weeklies_dir}")
        return

    weekly_files = sorted(weeklies_dir.glob("pivots_gap_ALL_W_*.csv"))
    if not weekly_files:
        print(f"❌ Keine Weekly-Pivots CSV in {weeklies_dir} gefunden.")
        return

    weeklies_path = weekly_files[-1]
    print(f"🔄 Verwende Weekly-Pivots CSV: {weeklies_path}")

    # 2) H4-Daten-Ordner
    h4_dir = base / "time frame data" / "4h data"
    if not h4_dir.exists():
        print(f"❌ 4h-Ordner nicht gefunden: {h4_dir}")
        return

    weeklies = load_weeklies(weeklies_path)
    if weeklies.empty:
        print("⚠️ Keine gültigen Weekly-Pivots gefunden.")
        return

    h4_map = find_h4_files_map(h4_dir)
    if not h4_map:
        print(f"❌ Keine H4-Dateien in {h4_dir} gefunden.")
        return

    all_rows: list[dict] = []
    skipped: list[str] = []

    for _, row in weeklies.iterrows():
        pair6 = pair_code_from_str(row["pair"])
        h4_path = h4_map.get(pair6)
        if not h4_path:
            skipped.append(pair6)
            continue

        h4_df = read_ohlc_file(h4_path)
        if h4_df is None or h4_df.empty:
            skipped.append(pair6)
            continue

        all_rows += scan_pair_for_pivot(h4_df, pair6, row)

    if not all_rows:
        print("⚠️ Keine gültigen Wick-Differences gefunden "
              "(Regeln: Richtung, komplett in Weekly-Gap, ≤19%, UNBERÜHRT bis Weekly-Touch/Datenende).")
        if skipped:
            print("ℹ️ Übersprungen (keine/ungültige 4h-Datei):",
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

    # 3) Output-Pfad
    out_dir = base / "outputs" / "wickdiffs" / "W→H4"
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"wick_diffs_H4_{stamp}.csv"
    out.to_csv(out_path, index=False)

    print(f"✅ Gefundene (UNBERÜHRTE) Wick-Differences: {len(out)}")
    with pd.option_context("display.width", 220, "display.max_columns", None):
        print(out.head(PREVIEW_ROWS).to_string(index=False))
    print(f"\n💾 Ergebnisse gespeichert in: {out_path.resolve()}")

    if skipped:
        print("ℹ️ Übersprungen (keine/ungültige oder fehlende 4h-Datei):",
              ", ".join(sorted(set(skipped))))


if __name__ == "__main__":
    main()
