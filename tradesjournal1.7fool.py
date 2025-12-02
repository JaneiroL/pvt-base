trades: List[dict] = []
    used_pivots: Set[Tuple] = set()
    skipped_pairs: Set[str] = set()

    for _, row in wd.iterrows():
        pair6 = str(row["pair6"]).upper()
        if pair6 != "EURCHF":
            continue

        direction = str(row["pivot_type"]).lower()
        if direction not in {"long", "short"}:
            continue

        pivot_touch_time = row["pivot_touch_time"]
        if pd.isna(pivot_touch_time):
            continue

        # Zeitraumfilter: nur Pivots, deren FIRST TOUCH im gewünschten Zeitfenster liegt
        if not (FILTER_START <= pivot_touch_time <= FILTER_END):
            continue

        pivot_id = (
            pair6,
            direction,
            row["pivot_first_time"],
            row["pivot_second_time"],
        )
        if pivot_id in used_pivots:
            continue

        ltf_path = ltf_map.get(pair6)
        if not ltf_path:
            skipped_pairs.add(pair6)
            continue

        ltf_df = read_ohlc_file(ltf_path)
        if ltf_df is None or ltf_df.empty:
            skipped_pairs.add(pair6)
            continue

        rng = float(row["pivot_high"] - row["pivot_low"])
        if rng <= 0:
            continue

        if direction == "long":
            pivot_tp_price = float(row["pivot_high"] + rng)
        else:
            pivot_tp_price = float(row["pivot_low"] - rng)

        start_time = pivot_touch_time
        entry_idx, invalidated = find_entry_candle(
            ltf_df=ltf_df,
            zone_low=row["wd_low"],
            zone_high=row["wd_high"],
            direction=direction,
            start_time=start_time,
            pivot_tp_price=pivot_tp_price,
        )

        if invalidated:
            used_pivots.add(pivot_id)
            continue

        if entry_idx is None:
            continue

        entry_price = float(ltf_df.loc[entry_idx, "close"])
        tp_price, sl_price, rr_pos = compute_tp_sl(
            entry_price, row["pivot_low"], row["pivot_high"], direction, pair6
        )
        if tp_price is None:
            continue

        sim = simulate_trade(
            ltf_df, entry_idx, direction, entry_price, tp_price, sl_price, rr_pos
        )
        if sim is None:
            continue

        trades.append(
            {
                "mode": "3D",
                "ltf": ltf_label,
                "pair": pair6,
                "direction": direction,
                "pivot_first_time": row["pivot_first_time"],
                "pivot_second_time": row["pivot_second_time"],
                "pivot_low": row["pivot_low"],
                "pivot_high": row["pivot_high"],
                "pivot_range": float(row["pivot_high"] - row["pivot_low"]),
                "wd_first_time": row["wd_first_time"],
                "wd_second_time": row["wd_second_time"],
                "wd_low": row["wd_low"],
                "wd_high": row["wd_high"],
                "entry_time": sim["entry_time"],
                "entry_price": sim["entry_price"],
                "tp_price": tp_price,
                "sl_price": sl_price,
                "exit_time": sim["exit_time"],
                "exit_price": sim["exit_price"],
                "result": sim["result"],
                "rr_signed": sim["rr_signed"],
            }
        )
        used_pivots.add(pivot_id)

    if skipped_pairs:
        print(
            "ℹ️ Übersprungen (fehlende/ungültige LTF-Datei):",
            ", ".join(sorted(skipped_pairs)),
        )

    return pd.DataFrame(trades)


def summarize_trades(
    df_trades: pd.DataFrame, out_dir: Path, stamp: str
) -> None:
    if df_trades.empty:
        print("⚠️ Keine Trades (EURCHF 3D 2015-01 bis 2015-08).")
        return

    df = df_trades.copy()
    df["is_win"] = df["result"] == "win"

    n_tot = len(df)