# wick_diffs_run.py
from __future__ import annotations
import argparse, re
from pathlib import Path
import pandas as pd
import numpy as np

# -------------------------
# Einstellungen
# -------------------------
MAX_REL_WIDTH = 0.19   # 19% Grenze

# -------------------------
# I/O: robuste CSV/Excel Leser (OHLC)
# -------------------------
CAND_TIME  = ["time","timestamp","date","datetime","unnamed: 0"]
CAND_OPEN  = ["open","o"]
CAND_HIGH  = ["high","h"]
CAND_LOW   = ["low","l"]
CAND_CLOSE = ["close","c"]

def _pick_col(df: pd.DataFrame, cands: list[str]) -> str:
    low = {str(c).strip().lower(): c for c in df.columns}
    # 1) exakte matches
    for cand in cands:
        if cand in low: return low[cand]
    # 2) teil-string matches
    for c in df.columns:
        lc = str(c).strip().lower()
        for cand in cands:
            if cand in lc:
                return c
    # 3) fallback: unnamed
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

    # Zeit robust parsen – KEINE TZ-Verschiebung (wir lassen alles naiv, wie geliefert; z.B. UTC+2)
    if pd.api.types.is_numeric_dtype(out["time"]):
        vmax = pd.Series(out["time"]).astype(float).abs().max()
        unit = "ms" if vmax > 1e12 else "s"
        out["time"] = pd.to_datetime(out["time"], unit=unit, utc=False)
    else:
        out["time"] = pd.to_datetime(out["time"], errors="coerce", utc=False)

    for col in ["open","high","low","close"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["time","open","high","low","close"]).sort_values("time").reset_index(drop=True)
    return out

def read_ohlc_file(path: Path) -> pd.DataFrame | None:
    if not path.exists(): return None
    if path.suffix.lower()==".csv":
        return _normalize_ohlc(pd.read_csv(path))
    try:
        sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")
        for _, df in sheets.items():
            try: return _normalize_ohlc(df)
            except: pass
    except Exception:
        return None
    return None

# -------------------------
# Weeklies CSV laden (beliebige Spaltennamen tolerant)
# -------------------------
def _pick_col_generic(df: pd.DataFrame, name: str) -> str:
    low = {str(c).lower(): c for c in df.columns}
    n2 = name.lower()
    if n2 in low: return low[n2]
    spaced = n2.replace("_"," ")
    if spaced in low: return low[spaced]
    for c in df.columns:
        if n2 in str(c).lower():
            return c
    raise KeyError(name)

def load_weeklies(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = pd.DataFrame({
        "pair"               : df[_pick_col_generic(df,"pair")].astype(str),
        "timeframe"          : df[_pick_col_generic(df,"timeframe")].astype(str),
        "pivot_type"         : df[_pick_col_generic(df,"pivot_type")].astype(str).str.lower(),
        "first_candle_time"  : pd.to_datetime(df[_pick_col_generic(df,"first_candle_time")],  errors="coerce", utc=False),
        "second_candle_time" : pd.to_datetime(df[_pick_col_generic(df,"second_candle_time")], errors="coerce", utc=False),
        "gap_low"            : pd.to_numeric(df[_pick_col_generic(df,"gap_low")],  errors="coerce"),
        "gap_high"           : pd.to_numeric(df[_pick_col_generic(df,"gap_high")], errors="coerce"),
        # optional:
        "first_touch_time"   : pd.to_datetime(df[_pick_col_generic(df,"first_touch_time")] if "first_touch_time" in [c.lower() for c in df.columns] else pd.NaT, errors="coerce", utc=False)
    })
    lo = out[["gap_low","gap_high"]].min(axis=1)
    hi = out[["gap_low","gap_high"]].max(axis=1)
    out["gap_low"], out["gap_high"] = lo, hi
    out["pair6"] = out["pair"].str.upper().str.replace(r"[^A-Z]","", regex=True).str.extract(r"([A-Z]{6})", expand=False).fillna(out["pair"].str.upper())
    out = out[out["timeframe"].str.upper().str.startswith("W")].dropna(subset=["first_candle_time","second_candle_time","gap_low","gap_high"]).reset_index(drop=True)
    return out

# -------------------------
# Utils
# -------------------------
def candle_color(o: float, c: float) -> str:
    return "bull" if c>o else ("bear" if c<o else "doji")

def any_touch_between(df: pd.DataFrame, low: float, high: float, start_time: pd.Timestamp, end_time: pd.Timestamp) -> bool:
    """Gibt es IRGENDEINE 4h-Kerze, deren Range [low,high] schneidet, im Intervall (start_time, end_time]?"""
    if pd.isna(end_time):
        seg = df[df["time"] > start_time]
    else:
        seg = df[(df["time"] > start_time) & (df["time"] <= end_time)]
    # schneller: vektoriell prüfen
    if seg.empty: return False
    lo = min(low, high); hi = max(low, high)
    return bool(((seg["high"] >= lo) & (seg["low"] <= hi)).any())

def first_last_time(df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    return df["time"].iloc[0], df["time"].iloc[-1]

# -------------------------
# Kernlogik: pro Weekly-Pivot scannen
# -------------------------
def scan_pair_for_pivot(h4: pd.DataFrame, pair: str, pivot_row: pd.Series) -> list[dict]:
    ptype = pivot_row["pivot_type"]         # "long" / "short"
    t1    = pd.Timestamp(pivot_row["first_candle_time"])
    t2    = pd.Timestamp(pivot_row["second_candle_time"])
    w_touch = pivot_row.get("first_touch_time", pd.NaT)
    lo_w  = float(pivot_row["gap_low"])
    hi_w  = float(pivot_row["gap_high"])
    lo_w, hi_w = (min(lo_w, hi_w), max(lo_w, hi_w))
    width_w = hi_w - lo_w
    if width_w <= 0:
        return []

    # Zeitfenster: von Start der ersten Weekly-Kerze bis Ende der Woche der zweiten Weekly-Kerze
    win_start = t1.normalize()  # 00:00 dieses Tages
    win_end   = (t2 + pd.Timedelta(days=4, hours=23, minutes=59, seconds=59))

    dfw = h4[(h4["time"] >= win_start) & (h4["time"] <= win_end)].reset_index(drop=True)
    if dfw.shape[0] < 2:
        return []

    _, data_end = first_last_time(h4)
    results = []

    # Kandidaten: benachbarte Kerzen mit passendem Farb-Muster
    o = dfw["open"].to_numpy()
    h = dfw["high"].to_numpy()
    l = dfw["low"].to_numpy()
    c = dfw["close"].to_numpy()
    t = dfw["time"].to_numpy()

    col = np.where(c>o, 1, np.where(c<o, -1, 0))  # 1=bull, -1=bear, 0=doji

    for i in range(len(dfw)-1):
        col1, col2 = col[i], col[i+1]

        if ptype == "short":
            # bull → bear
            if not (col1==1 and col2==-1): 
                continue
            z_lo = float(min(h[i], h[i+1]))
            z_hi = float(max(h[i], h[i+1]))
        elif ptype == "long":
            # bear → bull
            if not (col1==-1 and col2==1):
                continue
            z_lo = float(min(l[i], l[i+1]))
            z_hi = float(max(l[i], l[i+1]))
        else:
            continue

        # komplett innerhalb Weekly-Gap
        if not (z_lo >= lo_w and z_hi <= hi_w):
            continue

        # Breite ≤ 19% der Weekly-Range
        z_width = z_hi - z_lo
        if (z_width / width_w) > MAX_REL_WIDTH:
            continue

        t_a = pd.Timestamp(t[i])
        t_b = pd.Timestamp(t[i+1])

        # --- UNBERÜHRT-REGEL ALS HARTE VORAUSSETZUNG ---
        # Prüffenster: (t_b, weekly_touch] wenn vorhanden, sonst (t_b, Datenende]
        end_check = pd.Timestamp(w_touch) if pd.notna(w_touch) else data_end

        touched = any_touch_between(h4, z_lo, z_hi, t_b, end_check)
        if touched:
            # wird bis zum weekly touch (oder bis Datenende) berührt -> ungültig
            continue

        results.append({
            "pair": pair,
            "pivot_type": ptype,
            "weekly_first_candle_time": t1,
            "weekly_second_candle_time": t2,
            "weekly_gap_low": lo_w, "weekly_gap_high": hi_w,
            "weekly_gap_width": width_w,
            "ltf": "H4",
            "wd_first_candle_time": t_a, 
            "wd_second_candle_time": t_b,
            "wd_zone_low": z_lo, "wd_zone_high": z_hi, 
            "wd_zone_width": z_width,
            "wd_zone_pct_of_weekly": z_width/width_w,
            "weekly_first_touch_time": (pd.Timestamp(w_touch) if pd.notna(w_touch) else pd.NaT),
            "pending_until_weekly_touch": pd.isna(w_touch),  # True, wenn noch kein Weekly-Touch stattgefunden hat
        })

    return results

# -------------------------
# Pair-Dateien im H4-Ordner finden
# -------------------------
def pair_code_from_str(s: str) -> str:
    up = re.sub(r"[^A-Z]", "", str(s).upper())
    m = re.search(r"[A-Z]{6}", up)
    return m.group(0) if m else up[:6] or str(s)

def find_h4_files_map(h4_dir: Path) -> dict[str, Path]:
    mp: dict[str, Path] = {}
    if not h4_dir.exists(): return mp
    for p in h4_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".csv",".xlsx"}:
            code = pair_code_from_str(p.name)
            if len(code)==6 and code not in mp:
                mp[code] = p
    return mp

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="Finde 4h Wick-Differences innerhalb Weekly-Pivot-Gaps (≤19% + UNBERÜHRT bis Weekly-Touch / Datenende).")
    ap.add_argument("--weeklies", required=True, help="Pfad zur Weekly-Pivots CSV (pivots_gap_ALL_W_*.csv)")
    ap.add_argument("--h4dir",   required=True, help="Ordner mit 4h-CSV/Excel pro Pair")
    ap.add_argument("--only_pair", default=None, help="Optional: nur dieses 6-Letter Pair scannen, z.B. AUDCHF")
    ap.add_argument("--out",     default=None,  help="Ausgabe-CSV (optional). Standard: wick_diffs_H4_<timestamp>.csv")
    ap.add_argument("--preview", type=int, default=20, help="Zeilen in der Terminal-Vorschau (Default 20)")
    args = ap.parse_args()

    weeklies_path = Path(args.weeklies)
    if not weeklies_path.exists():
        print(f"❌ Weekly-CSV nicht gefunden: {weeklies_path}")
        return

    weeklies = load_weeklies(weeklies_path)
    if args.only_pair:
        p6 = pair_code_from_str(args.only_pair)
        weeklies = weeklies[weeklies["pair6"] == p6].reset_index(drop=True)

    h4_map = find_h4_files_map(Path(args.h4dir))

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
        print("⚠️ Keine gültigen Wick-Differences gefunden (Regeln: Richtung, komplett in Weekly-Gap, ≤19%, UNBERÜHRT bis Weekly-Touch/Datenende).")
        if skipped:
            print("ℹ️ Übersprungen (keine/ungültige 4h-Datei):", ", ".join(sorted(set(skipped))))
        return

    out = pd.DataFrame(all_rows)

    # Duplikate entfernen (manche Feeds erzeugen doppelte Kandidaten)
    dedup_cols = ["pair","pivot_type","weekly_first_candle_time","weekly_second_candle_time",
                  "wd_first_candle_time","wd_second_candle_time","wd_zone_low","wd_zone_high"]
    out = out.drop_duplicates(subset=dedup_cols, keep="first").reset_index(drop=True)

    # Times hübsch formatieren (keine TZ-Umrechnung)
    time_cols = ["weekly_first_candle_time","weekly_second_candle_time",
                 "wd_first_candle_time","wd_second_candle_time","weekly_first_touch_time"]
    for col in time_cols:
        out[col] = pd.to_datetime(out[col]).dt.strftime("%Y-%m-%d %H:%M")

    # Ausgabe
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.out) if args.out else Path(f"wick_diffs_H4_{stamp}.csv")
    out.to_csv(out_path, index=False)

    print(f"✅ Gefundene (UNBERÜHRTE) Wick-Differences: {len(out)}")
    with pd.option_context("display.width", 220, "display.max_columns", None):
        print(out.head(args.preview).to_string(index=False))
    print(f"\n💾 Ergebnisse gespeichert in: {out_path.resolve()}")
    if skipped:
        print("ℹ️ Übersprungen (keine/ungültige 4h-Datei):", ", ".join(sorted(set(skipped))))

if __name__ == "__main__":
    main()
