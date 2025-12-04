diff --git a/tradesjournal1.7fool.py b/tradesjournal1.7fool.py
index 9b064539fb2cf865c6228fc64ace5bedfa2c4c3e..97511bb9efb699d1162c86f7c2615891fd00e174 100644
--- a/tradesjournal1.7fool.py
+++ b/tradesjournal1.7fool.py
@@ -1,27 +1,28 @@
 from __future__ import annotations
 
+import argparse
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
@@ -742,95 +743,126 @@ def summarize_trades(
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
 # Main – Variante A: kompletter Backtest
 # -----------------------------------
-def main_allpairs() -> None:
+def main_allpairs(
+    pivot_start: Optional[pd.Timestamp] = None,
+    pivot_end: Optional[pd.Timestamp] = None,
+) -> None:
     base = Path(__file__).resolve().parent
 
     out_base = base / "outputs" / "trades"
     out_base.mkdir(parents=True, exist_ok=True)
     stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
 
     # Weekly: W -> H4, alle Paare
     trades_w = run_mode(
         base=base,
         mode="W",
         ltf_dir=base / "time frame data" / "4h data",
         ltf_label="H4",
+        pivot_start=pivot_start,
+        pivot_end=pivot_end,
     )
     if not trades_w.empty:
         path_w = out_base / f"trades_W_{stamp}.csv"
         trades_w.to_csv(path_w, index=False)
         print(f"\n💾 Trades W gespeichert in: {path_w.resolve()}")
         summarize_trades(trades_w, "W", out_base, stamp)
 
     # 3D: 3D -> H1, alle Paare
     trades_3d = run_mode(
         base=base,
         mode="3D",
         ltf_dir=base / "time frame data" / "1h data",
         ltf_label="H1",
+        pivot_start=pivot_start,
+        pivot_end=pivot_end,
     )
     if not trades_3d.empty:
         path_3d = out_base / f"trades_3D_{stamp}.csv"
         trades_3d.to_csv(path_3d, index=False)
         print(f"\n💾 Trades 3D gespeichert in: {path_3d.resolve()}")
         summarize_trades(trades_3d, "3D", out_base, stamp)
 
 
 # -----------------------------------
 # Main – Variante B: nur EURCHF, 3D, 2015-01 bis 2015-08
 # -----------------------------------
 def main_eurchf_3d_2015() -> None:
     base = Path(__file__).resolve().parent
 
     out_base = base / "outputs" / "trades"
     out_base.mkdir(parents=True, exist_ok=True)
     stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
 
     start = pd.Timestamp(2015, 1, 1)
     end   = pd.Timestamp(2015, 9, 1)  # < 2015-09-01 → bis inkl. August
 
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
         path_3d = out_base / f"trades_3D_EURCHF_2015H1_{stamp}.csv"
         trades_3d.to_csv(path_3d, index=False)
         print(f"\n💾 Trades 3D EURCHF 2015 gespeichert in: {path_3d.resolve()}")
         summarize_trades(trades_3d, "3D_EURCHF_2015", out_base, stamp)
 
 
 if __name__ == "__main__":
+    parser = argparse.ArgumentParser(
+        description="Tradesjournal-Runner für alle Paare (W und 3D)."
+    )
+    parser.add_argument(
+        "--start",
+        type=str,
+        help="Startdatum YYYY-MM-DD (optional, filtert Pivots ab diesem Datum)",
+    )
+    parser.add_argument(
+        "--end",
+        type=str,
+        help="Enddatum YYYY-MM-DD (optional, filtert Pivots vor diesem Datum)",
+    )
+    args = parser.parse_args()
+
+    def _parse_date(val: Optional[str]) -> Optional[pd.Timestamp]:
+        if not val:
+            return None
+        ts = pd.to_datetime(val, errors="coerce")
+        return ts.tz_localize(None) if ts.tzinfo else ts
+
+    start_ts = _parse_date(args.start)
+    end_ts = _parse_date(args.end)
+
     # Standard: kompletter Backtest für alle Paare
-    main_allpairs()
+    main_allpairs(pivot_start=start_ts, pivot_end=end_ts)
 
     # Wenn du NUR EURCHF 3D von 2015-01 bis 2015-08 testen willst:
     # → obere Zeile auskommentieren und diese aktivieren:
     # main_eurchf_3d_2015()
