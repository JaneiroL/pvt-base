#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, sys, re
from pathlib import Path
from datetime import datetime
from dateutil import parser as dtparse
import pandas as pd

# -------- Flexible Spaltenkandidaten --------
CANDIDATE_TIME_COLS  = ["time", "timestamp", "date", "datetime"]
CANDIDATE_OPEN_COLS  = ["open", "o"]
CANDIDATE_HIGH_COLS  = ["high", "h"]
CANDIDATE_LOW_COLS   = ["low", "l"]
CANDIDATE_CLOSE_COLS = ["close", "c"]

# 28-Pair-Universum (inkl. GBPNZD, ohne XAU/XAG)
PAIRS_28 = {
    "AUDCAD", "AUDCHF", "AUDJPY", "AUDNZD", "AUDUSD",
    "CADCHF", "CADJPY",
    "CHFJPY",
    "EURAUD", "EURCAD", "EURCHF", "EURGBP", "EURJPY", "EURNZD", "EURUSD",
    "GBPAUD", "GBPCAD", "GBPCHF", "GBPJPY", "GBPNZD", "GBPUSD",
    "NZDCAD", "NZDCHF", "NZDJPY", "NZDUSD",
    "USDCAD", "USDCHF", "USDJPY",
}

# NUR die eine Exception (wie bei dir)
SPECIAL_PAIR_FIX = {
    "OANDAG": "GBPNZD",
}

TF_ORDER = ["3D", "W", "2W", "M"]

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
    raise ValueError(
        f"Keine passende Spalte gefunden. Erwartet eine von {candidates}, "
        f"gefunden: {list(df.columns)}"
    )

def infer_pair_from_text(txt: str):
    """
    Robust, damit GBP-Paare nicht vermischt werden.

    PRIORITÄT:
    1) Exakter OANDA_{PAIR} Treffer
    2) Suche nach 28er-Paaren als Substring
    3) Fallback: 6-Letter-Blöcke; erst dann SPECIAL_PAIR_FIX anwenden
    """
    s = str(txt)

    # 1) OANDA_{PAIR} extrahieren (underscore/space/minus tolerant)
    m = re.search(r"OANDA[\s_\-]*([A-Z]{6})", s.upper())
    if m:
        code = m.group(1)
        if code in PAIRS_28:
            return code
        code2 = SPECIAL_PAIR_FIX.get(code, code)
        if code2 in PAIRS_28:
            return code2

    # 2) Normalisierte Buchstabenfolge
    up = re.sub(r"[^A-Z]", "", s.upper())
    for p in sorted(PAIRS_28):
        if p in up:
            return p

    # 3) 6-Letter-Fallback + Spezialfix
    m2 = re.search(r"([A-Z]{6})", up)
    if m2:
        code = m2.group(1)
        code = SPECIAL_PAIR_FIX.get(code, code)
        return code

    return None

def infer_pair_from_filename(path: Path) -> str:
    p = infer_pair_from_text(path.name)
    return p if p else path.stem

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
    try:
        return read_ohlc_from_df(df_raw)
    except Exception:
        pass

    df = df_raw.copy()
    df.columns = [str(c) for c in df.columns]
    best_row = None

    for i in range(min(25, len(df))):
        row_vals = [str(x).strip().lower() for x in list(df.iloc[i].values)]
        ohlc_hits = sum(any(re.search(rf"\b{k}\b", v) for v in row_vals) for k in ["open", "high", "low", "close"])
        time_hits = any(re.search(r"\b(time|timestamp|date|datetime)\b", v) for v in row_vals)
        if ohlc_hits >= 3 and time_hits:
            best_row = i
            break

    if best_row is None:
        return read_ohlc_from_df(df)

    new_cols = [str(x).strip() for x in list(df.iloc[best_row].values)]
    df2 = df.iloc[best_row + 1 :].reset_index(drop=True)
    df2.columns = new_cols
    return read_ohlc_from_df(df2)

def read_ohlc_from_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return try_read_ohlc_with_header_scan(df)

def read_ohlc_sheets_from_xlsx(xlsx_path: Path) -> dict:
    sheets = pd.read_excel(xlsx_path, sheet_name=None, engine="openpyxl", header=0)
    out = {}
    for sh, df in sheets.items():
        try:
            out[sh] = try_read_ohlc_with_header_scan(df)
        except Exception:
            continue
    return out

def color(o, c):
    if c > o:
        return "bull"
    if c < o:
        return "bear"
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
    if start_dt:
        df = df[df["time"] >= pd.Timestamp(start_dt)]
    if end_dt:
        df = df[df["time"] <= pd.Timestamp(end_dt)]
    df = df.reset_index(drop=True)

    rows = []
    for i in range(len(df) - 1):
        o1, h1, l1, c1, t1 = df.loc[i, ["open", "high", "low", "close", "time"]]
        o2, h2, l2, c2, t2 = df.loc[i + 1, ["open", "high", "low", "close", "time"]]
        col1, col2 = color(o1, c1), color(o2, c2)

        if col1 == "doji" or col2 == "doji":
            continue

        if col1 == "bear" and col2 == "bull":
            ptype = "long"
        elif col1 == "bull" and col2 == "bear":
            ptype = "short"
        else:
            continue

        chosen_price = float(o2)

        # --- Pivot-Zone (gap_low/gap_high) nach deiner Logik ---
        if ptype == "long":
            gap_low, gap_high = float(min(l1, l2)), chosen_price
        else:
            gap_low, gap_high = chosen_price, float(max(h1, h2))

        gap_low_n = float(min(gap_low, gap_high))
        gap_high_n = float(max(gap_low, gap_high))

        # --- Große Wick-Diff (deine Definition) ---
        if ptype == "long":
            wick_diff_low = gap_low_n
            wick_diff_high = float(max(l1, l2))  # weniger tiefer Dot
        else:
            wick_diff_high = gap_high_n
            wick_diff_low = float(min(h1, h2))   # weniger hoher Dot

        # clamp sicher in Pivot-Zone
        wick_diff_low = max(wick_diff_low, gap_low_n)
        wick_diff_high = min(wick_diff_high, gap_high_n)
        wick_diff_low, wick_diff_high = (min(wick_diff_low, wick_diff_high),
                                         max(wick_diff_low, wick_diff_high))

        # --- OUTSIDE-Zone innerhalb Pivot aber außerhalb Wick-Diff ---
        # Long:  [wick_diff_high .. gap_high]
        # Short: [gap_low .. wick_diff_low]
        if ptype == "long":
            outside_low = float(wick_diff_high)
            outside_high = float(gap_high_n)
        else:
            outside_low = float(gap_low_n)
            outside_high = float(wick_diff_low)

        outside_low, outside_high = (min(outside_low, outside_high), max(outside_low, outside_high))

        # first touch des großen Pivot-Gaps
        j_idx, j_time, j_price = first_touch_after(df, gap_low_n, gap_high_n, i + 2)
        candles_to_touch = (j_idx - (i + 1)) if j_idx is not None else None
        days_to_touch = ((pd.Timestamp(j_time) - pd.Timestamp(t2)).days if j_time is not None else None)

        # Optional: kompatibel zu deinem "weeks_to_first_touch" Feld (alte Variante 3)
        weeks_to_touch = candles_to_touch

        rows.append({
            "pair": pair,
            "timeframe": timeframe,
            "pivot_type": ptype,
            "first_candle_time": t1,
            "second_candle_time": t2,
            "chosen_price": chosen_price,
            "gap_low": gap_low_n,
            "gap_high": gap_high_n,
            "wick_diff_low": float(wick_diff_low),
            "wick_diff_high": float(wick_diff_high),
            "outside_low": float(outside_low),
            "outside_high": float(outside_high),
            "first_touch_time": j_time,
            "first_touch_price_repr": j_price,
            "candles_to_first_touch": candles_to_touch,
            "weeks_to_first_touch": weeks_to_touch,
            "days_to_first_touch": days_to_touch,
        })

    cols = [
        "pair","timeframe","pivot_type",
        "first_candle_time","second_candle_time",
        "chosen_price","gap_low","gap_high",
        "wick_diff_low","wick_diff_high",
        "outside_low","outside_high",
        "first_touch_time","first_touch_price_repr",
        "candles_to_first_touch","weeks_to_first_touch","days_to_first_touch",
    ]
    return pd.DataFrame(rows, columns=cols)

def sanitize_date_input(s):
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    return s.lstrip("> ").strip()

def parse_tf_input(s: str) -> set[str]:
    if s is None:
        raise ValueError("Ungültige Eingabe. Erlaubt: W, 3D, 2W, M, Both, All")
    raw = s.strip().lower()
    if not raw:
        raise ValueError("Ungültige Eingabe. Erlaubt: W, 3D, 2W, M, Both, All")

    if raw in {"both", "w+3d", "3d+w"}:
        return {"3D", "W"}
    if raw in {"all", "everything", "w+3d+2w+m"}:
        return {"3D", "W", "2W", "M"}

    parts = [p.strip() for p in re.split(r"[,\s+/|;]+", raw) if p.strip()]
    alias = {
        "w": "W", "week": "W", "weekly": "W", "1w": "W",
        "3d": "3D", "3": "3D", "3tage": "3D", "3-tage": "3D", "3day": "3D", "3days": "3D",
        "2w": "2W", "2week": "2W", "2weekly": "2W", "2-week": "2W", "2weeks": "2W", "2wöchig": "2W",
        "m": "M", "mon": "M", "month": "M", "monthly": "M", "1m": "M",
    }

    wanted = set()
    for p in parts:
        if p == "both":
            wanted |= {"3D", "W"}
            continue
        if p == "all":
            wanted |= {"3D", "W", "2W", "M"}
            continue
        if p not in alias:
            raise ValueError("Ungültige Eingabe. Erlaubt: W, 3D, 2W, M, Both, All")
        wanted.add(alias[p])

    if not wanted:
        raise ValueError("Ungültige Eingabe. Erlaubt: W, 3D, 2W, M, Both, All")
    return wanted

def parse_variant_input(s: str) -> str:
    if s is None:
        return "ALL"
    raw = s.strip().lower()
    if not raw:
        return "ALL"
    if raw in {"all", "full", "pivot", "gap", "normal"}:
        return "ALL"
    if raw in {"wd", "wigdiff", "wickdiff", "wick-diff", "wig-diff"}:
        return "WD"
    if raw in {"outside", "outer", "outerwd", "outer-wd", "outside-wd", "owd"}:
        return "OUTSIDE"
    raise ValueError("Ungültige Variante. Erlaubt: ALL | WD | OUTSIDE")

def prompt_if_missing(args):
    if not args.inpath:
        print("Pfad-Basis (leer = aktuelles Verzeichnis, ich suche darunter 'time frame data/...'):")
        ip = input("> ").strip('"').strip()
        args.inpath = ip if ip else "."

    if not args.start:
        print("Start-Datum (YYYY-MM-DD, leer = Anfang):")
        args.start = sanitize_date_input(input("> "))

    if not args.end:
        print("End-Datum (YYYY-MM-DD, leer = Ende):")
        args.end = sanitize_date_input(input("> "))

    if not args.tf:
        print("Timeframes wählen: 3D | W | 2W | M | Both | All (Kombis erlaubt, z.B. '3D, W, 2W'):")
        args.tf = input("> ").strip()

    if not args.variant:
        print("Variante wählen: ALL | WD | OUTSIDE:")
        args.variant = input("> ").strip()

    return args

def collect_files_for_tf(base: Path, wanted: set[str]):
    files: list[tuple[Path, str]] = []
    skipped: list[tuple[str, str]] = []

    # exakt wie bei dir: zuerst nach 'time frame data' unter base suchen
    tf_root = None
    for name in ["time frame data", "time_frame_data", "timeframe_data"]:
        cand = base / name
        if cand.exists() and cand.is_dir():
            tf_root = cand
            break
    if tf_root is None:
        tf_root = base

    tf_dirs = {
        "3D": [tf_root / "3D", tf_root / "3d", tf_root / "3Day", tf_root / "3Days"],
        "W":  [tf_root / "W", tf_root / "1W", tf_root / "Weekly"],
        "2W": [tf_root / "2Weekly", tf_root / "2W", tf_root / "2Week", tf_root / "2Weeks", tf_root / "2Weekly data"],
        "M":  [tf_root / "Monthly", tf_root / "M", tf_root / "1M", tf_root / "Month"],
    }

    for tf in TF_ORDER:
        if tf not in wanted:
            continue
        found_dir = None
        for cand in tf_dirs.get(tf, []):
            if cand.exists() and cand.is_dir():
                found_dir = cand
                break
        if not found_dir:
            skipped.append((tf, f"Kein Ordner für {tf} unter {tf_root} gefunden"))
            continue

        for p in found_dir.glob("**/*.csv"):
            files.append((p, tf))
        for p in found_dir.glob("**/*.xlsx"):
            files.append((p, tf))

    return files, skipped

def tf_to_outsub(tf: str) -> str:
    return {"W": "W", "3D": "3D", "2W": "2Weekly", "M": "Monthly"}[tf]

def tf_to_outtag(tf: str) -> str:
    return {"W": "W", "3D": "3D", "2W": "2W", "M": "M"}[tf]

def main():
    parser = argparse.ArgumentParser(
        description="Phase-1 Pivot Finder (3D / W / 2W / M) inkl. Wick-Diff + Outside. Variante steuerbar."
    )
    parser.add_argument("--inpath", type=str, help="Basis-Ordner (Standard: aktuelles Verzeichnis).")
    parser.add_argument("--start", type=str, help="Startdatum YYYY-MM-DD (optional)")
    parser.add_argument("--end", type=str, help="Enddatum YYYY-MM-DD (optional)")
    parser.add_argument("--tf", type=str, help="3D | W | 2W | M | Both | All (optional)")
    parser.add_argument("--variant", type=str, help="ALL | WD | OUTSIDE (optional)")
    parser.add_argument("--only-variant", action="store_true",
                        help="Wenn gesetzt: schreibe nur die Variant-Datei (statt zusätzlich ALL).")
    args = parser.parse_args()
    args = prompt_if_missing(args)

    base = Path(args.inpath)
    if not base.exists():
        print(f"❌ Pfad nicht gefunden: {base}")
        sys.exit(1)
    if base.is_file():
        print("❌ Bitte einen Basis-Ordner angeben, keine einzelne Datei.")
        sys.exit(1)

    start_dt = dtparse.parse(args.start).date() if args.start else None
    end_dt = dtparse.parse(args.end).date() if args.end else None

    try:
        wanted = parse_tf_input(args.tf)
    except Exception as e:
        print(f"❌ {e}")
        sys.exit(1)

    try:
        variant = parse_variant_input(args.variant)
    except Exception as e:
        print(f"❌ {e}")
        sys.exit(1)

    files, skipped = collect_files_for_tf(base, wanted)
    if not files:
        print("❌ Keine passenden Dateien für die gewählten Timeframes gefunden.")
        if skipped:
            print(f"(Hinweis: {len(skipped)} Hinweise)")
            for s in skipped:
                print("   •", s[0], "->", s[1])
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
                    processed_detail.append(f"{pair} ({tf}) aus {path.relative_to(base)}")
                    all_results.append(res)
            else:
                sheets = read_ohlc_sheets_from_xlsx(path)
                any_ok = False
                for sh_name, df in sheets.items():
                    pair = infer_pair_from_text(sh_name) or infer_pair_from_filename(path)
                    try:
                        res = detect_pivots_for_pair(pair, tf, df, start_dt, end_dt)
                        if not res.empty:
                            processed_detail.append(f"{pair} ({tf}) aus {path.relative_to(base)} / {sh_name}")
                            all_results.append(res)
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

    # ---------- Variant-View bauen (ohne die Pivot-Metas zu verlieren) ----------
    # Wir filtern NICHT die Pivot-Entstehung, sondern nur welche Arbeitszone "aktiv" ist.
    df_variant = out.copy()
    if variant == "WD":
        df_variant["work_low"] = df_variant["wick_diff_low"]
        df_variant["work_high"] = df_variant["wick_diff_high"]
    elif variant == "OUTSIDE":
        df_variant["work_low"] = df_variant["outside_low"]
        df_variant["work_high"] = df_variant["outside_high"]
    else:  # ALL
        df_variant["work_low"] = df_variant["gap_low"]
        df_variant["work_high"] = df_variant["gap_high"]
    df_variant["work_variant"] = variant

    # Console preview
    show = df_variant.copy()
    for col in ["first_candle_time", "second_candle_time", "first_touch_time"]:
        show[col] = pd.to_datetime(show[col]).dt.strftime("%Y-%m-%d").fillna("—")

    print("\nPivot Output (Phase-1 Unified) Preview:")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(show.head(30).to_string(index=False))
        if len(show) > 30:
            print(f"... ({len(show) - 30} weitere Zeilen)")

    # ---------- Output (Ordner/Strings bleiben wie bei dir) ----------
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_out = Path("outputs") / "pivots"
    outputs = []

    for tf in TF_ORDER:
        if tf not in wanted:
            continue

        sub = tf_to_outsub(tf)
        tag = tf_to_outtag(tf)
        dir_tf = base_out / sub
        dir_tf.mkdir(parents=True, exist_ok=True)

        # 1) Standard ALL (wie bisher) – optional abschaltbar via --only-variant
        if not args.only_variant:
            path_all = dir_tf / f"pivots_gap_ALL_{tag}_{stamp}.csv"
            out[out["timeframe"] == tf].to_csv(path_all, index=False)
            outputs.append(path_all)

        # 2) Variant-Datei (klar getrennt)
        #    Hinweis: OUTSIDE kann Breite 0 haben -> wie bei dir filtern wir raus
        df_tf_var = df_variant[df_variant["timeframe"] == tf].copy()
        if variant == "OUTSIDE":
            df_tf_var["outside_width"] = (df_tf_var["outside_high"] - df_tf_var["outside_low"]).astype(float)
            df_tf_var = df_tf_var[df_tf_var["outside_width"] > 0].drop(columns=["outside_width"])

        path_var = dir_tf / f"pivots_gap_{variant}_{tag}_{stamp}.csv"
        df_tf_var.to_csv(path_var, index=False)
        outputs.append(path_var)

    print("\n✅ Gespeichert:")
    for p in outputs:
        print(f" - {p.resolve()}")

    print(f"\nℹ️ Verarbeitet: {len(processed_detail)} Datenquellen")
    for line in processed_detail[:28]:
        print("   •", line)
    if len(processed_detail) > 28:
        print(f"   • ... (+{len(processed_detail) - 28} weitere)")

    if skipped:
        print(f"ℹ️ Hinweise/Übersprungen: {len(skipped)}")
        for s in skipped[:25]:
            print("   •", s[0], "->", s[1])

if __name__ == "__main__":
    main()
