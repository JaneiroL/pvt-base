#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, sys, re
from pathlib import Path
from datetime import datetime
from dateutil import parser as dtparse
import pandas as pd

# -------- Flexible Spaltenkandidaten --------
CANDIDATE_TIME_COLS = ["time", "timestamp", "date", "datetime"]
CANDIDATE_OPEN_COLS = ["open", "o"]
CANDIDATE_HIGH_COLS = ["high", "h"]
CANDIDATE_LOW_COLS  = ["low", "l"]
CANDIDATE_CLOSE_COLS= ["close", "c"]

PAIRS_28 = {
    "AUDCAD","AUDCHF","AUDJPY","AUDNZD","AUDUSD",
    "CADCHF","CADJPY",
    "CHFJPY",
    "EURAUD","EURCAD","EURCHF","EURGBP","EURJPY","EURNZD","EURUSD",
    "GBPAUD","GBPCAD","GBPCHF","GBPJPY","GBPUSD",
    "NZDCAD","NZDCHF","NZDJPY","NZDUSD",
    "USDCAD","USDCHF","USDJPY",
    "XAUUSD","XAGUSD"
}

# -------- Hilfsfunktionen --------
def find_col(df: pd.DataFrame, candidates):
    cols_map = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand in cols_map:
            return cols_map[cand]
    for c in df.columns:
        lc = str(c).strip().lower()
        for cand in candidates:
            if re.search(rf"\b{re.escape(cand)}\b", lc):
                return c
    raise ValueError(f"Keine passende Spalte gefunden. Erwartet eine von {candidates}, gefunden: {list(df.columns)}")

def infer_pair_from_text(txt: str):
    up = re.sub(r"[^A-Z]", "", txt.upper())
    for p in PAIRS_28:
        if p in up:
            return p
    m = re.search(r"([A-Z]{6})", up)
    return m.group(1) if m else None

def infer_pair_from_filename(path: Path) -> str:
    p = infer_pair_from_text(path.name)
    return p if p else path.stem

def infer_tf_from_name(name: str):
    n = name.upper()
    if "3D" in n:
        return "3D"
    if "1W" in n or "W1" in n or re.search(r"\bW\b", n):
        return "W"
    return None

def is_output_file(path: Path) -> bool:
    n = path.name.lower()
    return n.startswith("pivots") or "pivots_gap_" in n or "pivots_" in n

def is_probably_raw_oanda_csv(path: Path) -> bool:
    # nur CSVs wie "... , 1W_*" oder "... , 3D_*"
    return path.suffix.lower()==".csv" and not is_output_file(path) and re.search(r",\s*(1W|3D)_", path.name, flags=re.IGNORECASE)

def _standardize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def read_ohlc_from_df(df: pd.DataFrame) -> pd.DataFrame:
    df = _standardize_cols(df)
    tcol = find_col(df, CANDIDATE_TIME_COLS)
    ocol = find_col(df, CANDIDATE_OPEN_COLS)
    hcol = find_col(df, CANDIDATE_HIGH_COLS)
    lcol = find_col(df, CANDIDATE_LOW_COLS)
    ccol = find_col(df, CANDIDATE_CLOSE_COLS)
    df = df.rename(columns={tcol: "time", ocol: "open", hcol: "high", lcol: "low", ccol: "close"})
    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=False)
    df = df.sort_values("time").reset_index(drop=True)
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close", "time"]).reset_index(drop=True)
    return df[["time", "open", "high", "low", "close"]]

def try_read_ohlc_with_header_scan(df_raw: pd.DataFrame) -> pd.DataFrame:
    # 1) normal versuchen
    try:
        return read_ohlc_from_df(df_raw)
    except Exception:
        pass
    # 2) erste 25 Zeilen nach Header-Zeile scannen
    df = df_raw.copy()
    df.columns = [str(c) for c in df.columns]
    best_row = None
    for i in range(min(25, len(df))):
        row_vals = [str(x).strip().lower() for x in list(df.iloc[i].values)]
        ohlc_hits = sum(any(re.search(rf"\b{k}\b", v) for v in row_vals) for k in ["open","high","low","close"])
        time_hits = any(re.search(r"\b(time|timestamp|date|datetime)\b", v) for v in row_vals)
        if ohlc_hits >= 3 and time_hits:
            best_row = i
            break
    if best_row is None:
        return read_ohlc_from_df(df)
    new_cols = [str(x).strip() for x in list(df.iloc[best_row].values)]
    df2 = df.iloc[best_row+1:].reset_index(drop=True)
    df2.columns = new_cols
    return read_ohlc_from_df(df2)

def read_ohlc_from_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return try_read_ohlc_with_header_scan(df)

def read_ohlc_sheets_from_xlsx(xlsx_path: Path) -> dict:
    # {sheet_name: ohlc_df}
    sheets = pd.read_excel(xlsx_path, sheet_name=None, engine="openpyxl", header=0)
    out = {}
    for sh, df in sheets.items():
        try:
            out[sh] = try_read_ohlc_with_header_scan(df)
        except Exception:
            continue
    return out

def color(o, c):
    if c > o: return "bull"
    if c < o: return "bear"
    return "doji"

def first_touch_after(df: pd.DataFrame, zone_low: float, zone_high: float, start_idx: int):
    zl, zh = min(zone_low, zone_high), max(zone_low, zone_high)
    for j in range(start_idx, len(df)):
        lo, hi = df.loc[j, "low"], df.loc[j, "high"]
        if (hi >= zl) and (lo <= zh):
            touch_price = (max(lo, zl) + min(hi, zh)) / 2.0
            return j, df.loc[j, "time"], float(touch_price)
    return None, None, None

def detect_pivots_for_pair(pair: str, timeframe: str, df: pd.DataFrame, start_dt, end_dt) -> pd.DataFrame:
    if start_dt: df = df[df["time"] >= pd.Timestamp(start_dt)]
    if end_dt:   df = df[df["time"] <= pd.Timestamp(end_dt)]
    df = df.reset_index(drop=True)
    rows = []
    for i in range(len(df) - 1):
        o1, h1, l1, c1, t1 = df.loc[i,   ["open","high","low","close","time"]]
        o2, h2, l2, c2, t2 = df.loc[i+1, ["open","high","low","close","time"]]
        col1, col2 = color(o1, c1), color(o2, c2)
        if col1 == "doji" or col2 == "doji":
            continue
        if   col1 == "bear" and col2 == "bull": ptype = "long"
        elif col1 == "bull" and col2 == "bear": ptype = "short"
        else:
            continue

        # Deine Spezifikation:
        # chosen_price = IMMER Open der 2. Kerze
        chosen_price = float(o2)
        if ptype == "long":
            # LONG-Gap: tieferer Wick bis Open2
            gap_low, gap_high = float(min(l1, l2)), chosen_price
        else:
            # SHORT-Gap: Open2 bis höherer Wick
            gap_low, gap_high = chosen_price, float(max(h1, h2))

        j_idx, j_time, j_price = first_touch_after(df, gap_low, gap_high, i + 2)
        weeks_to_touch = (j_idx - (i + 1)) if j_idx is not None else None
        days_to_touch  = (pd.Timestamp(j_time) - pd.Timestamp(t2)).days if j_time is not None else None

        rows.append({
            "pair": pair, "timeframe": timeframe, "pivot_type": ptype,
            "first_candle_time": t1, "second_candle_time": t2,
            "chosen_price": chosen_price, "gap_low": float(min(gap_low, gap_high)),
            "gap_high": float(max(gap_low, gap_high)), "first_touch_time": j_time,
            "first_touch_price_repr": j_price, "weeks_to_first_touch": weeks_to_touch,
            "days_to_first_touch": days_to_touch,
        })
    cols = ["pair","timeframe","pivot_type","first_candle_time","second_candle_time",
            "chosen_price","gap_low","gap_high","first_touch_time","first_touch_price_repr",
            "weeks_to_first_touch","days_to_first_touch"]
    return pd.DataFrame(rows, columns=cols)

def sanitize_date_input(s: str | None) -> str | None:
    if s is None: return None
    s = s.strip()
    if not s: return None
    return s.lstrip("> ").strip()

def parse_tf_input(s: str):
    s = s.strip().lower()
    if s in {"w", "week", "weekly"}: return {"W"}
    if s in {"3d", "3", "3days", "3-day", "3tage", "3-tage"}: return {"3D"}
    if s in {"both", "all", "w+3d", "3d+w"}: return {"W","3D"}
    raise ValueError("Ungültige Eingabe. Erlaubt: W, 3D oder Both")

def prompt_if_missing(args):
    if not args.inpath:
        print("Pfad zu Datei/Ordner (leer = aktuelles Verzeichnis):")
        ip = input("> ").strip('"').strip()
        args.inpath = ip if ip else "."
    if not args.start:
        print("Start-Datum (YYYY-MM-DD, leer = Anfang):"); args.start = sanitize_date_input(input("> "))
    if not args.end:
        print("End-Datum (YYYY-MM-DD, leer = Ende):");   args.end   = sanitize_date_input(input("> "))
    if not args.tf:
        print("Timeframe wählen: 'W' (Weekly), '3D' (3-Tage) oder 'Both'"); args.tf = input("> ").strip()
    return args

# -------- Hauptlogik --------
def main():
    parser = argparse.ArgumentParser(description="Multi-Pair Pivot-Gap Finder (W / 3D / Both).")
    parser.add_argument("--inpath", type=str, help="Datei oder Ordner (CSV/XLSX). Default: aktuelles Verzeichnis")
    parser.add_argument("--start", type=str, help="Startdatum YYYY-MM-DD (optional)")
    parser.add_argument("--end", type=str, help="Enddatum YYYY-MM-DD (optional)")
    parser.add_argument("--tf", type=str, help="W | 3D | Both (optional)")
    args = parser.parse_args()
    args = prompt_if_missing(args)

    base = Path(args.inpath)
    if not base.exists():
        print(f"❌ Pfad nicht gefunden: {base}"); sys.exit(1)

    start_dt = dtparse.parse(args.start).date() if args.start else None
    end_dt   = dtparse.parse(args.end).date() if args.end else None
    try:
        wanted = parse_tf_input(args.tf)
    except Exception as e:
        print(f"❌ {e}"); sys.exit(1)

    files = []
    skipped = []

    if base.is_dir():
        # CSVs (nur “rohe” OANDA-Dateien, keine eigenen Outputs)
        for p in base.glob("**/*.csv"):
            if is_output_file(p): 
                skipped.append((p, "Output-Datei (ignored)")); 
                continue
            if not is_probably_raw_oanda_csv(p): 
                skipped.append((p, "kein OANDA-Input-Muster")); 
                continue
            tf = infer_tf_from_name(p.name)
            if tf and tf in wanted:
                files.append((p, tf))
        # XLSX (z. B. FX_1W_all_pairs(...).xlsx) -> alle Sheets lesen
        for p in base.glob("**/*.xlsx"):
            if is_output_file(p): 
                skipped.append((p, "Output-Datei (ignored)")); 
                continue
            tf = infer_tf_from_name(p.name)
            if tf is None and "W" in wanted and re.search(r"\b1W\b|_1W|1W_", p.name.upper()):
                tf = "W"
            if tf and tf in wanted:
                files.append((p, tf))
    else:
        if base.suffix.lower() == ".csv":
            if is_output_file(base) or not is_probably_raw_oanda_csv(base):
                print("❌ Datei entspricht nicht dem erwarteten OANDA-CSV-Muster (', 1W_' / ', 3D_')."); 
                sys.exit(1)
            tf = infer_tf_from_name(base.name)
            if tf and tf in wanted: 
                files.append((base, tf))
        elif base.suffix.lower() in {".xlsx", ".xls"}:
            tf = infer_tf_from_name(base.name)
            if tf is None and "W" in wanted: 
                tf = "W"
            if tf and tf in wanted: 
                files.append((base, tf))

    if not files:
        print("❌ Keine passenden Dateien für die gewählten Timeframes gefunden.")
        if skipped: 
            print(f"(Hinweis: {len(skipped)} Elemente übersprungen)")
        sys.exit(1)

    all_results = []
    processed_detail = []
    for path, tf in files:
        try:
            if path.suffix.lower() == ".csv":
                df = read_ohlc_from_csv(path)
                pair = infer_pair_from_filename(path)
                res = detect_pivots_for_pair(pair, tf, df, start_dt, end_dt)
                if not res.empty:
                    processed_detail.append(f"{pair} ({tf}) aus {path.name}")
                    all_results.append(res)
            else:
                sheets = read_ohlc_sheets_from_xlsx(path)
                any_ok = False
                for sh_name, df in sheets.items():
                    pair = infer_pair_from_text(sh_name) or infer_pair_from_filename(path)
                    try:
                        res = detect_pivots_for_pair(pair, tf, df, start_dt, end_dt)
                        if not res.empty:
                            processed_detail.append(f"{pair} ({tf}) aus {path.name} / {sh_name}")
                            all_results.append(res); 
                            any_ok = True
                    except Exception as e:
                        skipped.append((f"{path.name} / {sh_name}", f"Fehler: {e}"))
                if not any_ok:
                    skipped.append((path.name, "Keine OHLC-Sheets erkannt"))
        except Exception as e:
            skipped.append((path.name, f"Fehler: {e}"))

    if not all_results:
        print("Keine Pivots im angegebenen Zeitraum/TF gefunden.")
        if skipped: 
            print(f"(Hinweis: {len(skipped)} Elemente übersprungen)")
        return

    out = pd.concat(all_results, ignore_index=True)
    show = out.copy()
    for col in ["first_candle_time","second_candle_time","first_touch_time"]:
        show[col] = pd.to_datetime(show[col]).dt.strftime("%Y-%m-%d").fillna("—")

    print("\nPivot-Gap Output (Multi):")
    with pd.option_context("display.width", 120, "display.max_columns", None):
        print(show.head(40).to_string(index=False))
        if len(show) > 40: 
            print(f"... ({len(show)-40} weitere Zeilen)")

    # ---- Neuer Output-Block: in feste Ordnerstruktur schreiben ----
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_out = Path("outputs") / "pivots"
    outputs = []

    # Weekly-Pivots -> outputs/pivots/W/
    if "W" in wanted:
        dir_w = base_out / "W"
        dir_w.mkdir(parents=True, exist_ok=True)
        path_w = dir_w / f"pivots_gap_ALL_W_{stamp}.csv"
        out[out["timeframe"] == "W"].to_csv(path_w, index=False)
        outputs.append(path_w)

    # 3D-Pivots -> outputs/pivots/3D/
    if "3D" in wanted:
        dir_3d = base_out / "3D"
        dir_3d.mkdir(parents=True, exist_ok=True)
        path_3d = dir_3d / f"pivots_gap_ALL_3D_{stamp}.csv"
        out[out["timeframe"] == "3D"].to_csv(path_3d, index=False)
        outputs.append(path_3d)

    # Kombinierte Datei (falls Both gewählt) -> outputs/pivots/
    if wanted == {"W","3D"}:
        base_out.mkdir(parents=True, exist_ok=True)
        path_all = base_out / f"pivots_gap_ALL_Both_{stamp}.csv"
        out.to_csv(path_all, index=False)
        outputs.append(path_all)

    print("\n✅ Gespeichert:")
    for p in outputs:
        print(f" - {p.resolve()}")

    print(f"\nℹ️ Verarbeitet: {len(processed_detail)} Datenquellen")
    for line in processed_detail[:28]:
        print("   •", line)
    if len(processed_detail) > 28:
        print(f"   • ... (+{len(processed_detail)-28} weitere)")
    if skipped:
        print(f"ℹ️ Übersprungen: {len(skipped)} (z. B. Outputs / Non-OHLC)")

if __name__ == "__main__":
    main()
