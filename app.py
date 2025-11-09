with tab_compare:
    st.header("Actual vs Predicted (latest dated draw)")
    df_sorted = df.sort_values(by='date', ascending=False, na_position='last').reset_index(drop=True)

    # Find latest valid draw (has date and 8 numbers)
    latest_idx = df_sorted['date'].first_valid_index() or 0
    latest_row = df_sorted.iloc[latest_idx]
    actual = [int(latest_row[f"n{i}"]) for i in range(1, NUMBERS_PER_DRAW + 1)]

    def position_match_pct(pred, actual):
        same_pos = sum(1 for p, a in zip(pred, actual) if p == a)
        return 100.0 * same_pos / NUMBERS_PER_DRAW

    records = []
    best_pos = best_ov = best_dist = 0
    for i, pred in enumerate(predictions, start=1):
        pos_pct = round(position_match_pct(pred, actual), 2)
        ov_pct = round(closeness_overlap_pct(pred, actual), 2)
        dist_pct = round(avg_min_distance_score(pred, actual), 2)
        best_pos = max(best_pos, pos_pct)
        best_ov = max(best_ov, ov_pct)
        best_dist = max(best_dist, dist_pct)
        records.append({
            "Prediction Set": i,
            "Predicted": ", ".join(map(str, pred)),
            "Actual (latest)": ", ".join(map(str, actual)),
            "Position Match (%)": pos_pct,
            "Overlap (%)": ov_pct,
            "Distance Score (%)": dist_pct
        })

    comp_df = pd.DataFrame(records)
    st.dataframe(comp_df, use_container_width=True)

    # --- Summary card ---
    latest_date_str = (
        latest_row['date'].strftime('%Y-%m-%d')
        if pd.notna(latest_row['date'])
        else "Unknown"
    )
    st.markdown(
        f"""
        <div style='background:#f6f9ff;padding:15px;border-radius:10px;margin-top:10px'>
        <b>📅 Latest Draw:</b> {latest_date_str}<br>
        <b>🎯 Best Position Match:</b> {best_pos:.2f}%<br>
        <b>🔁 Best Overlap Match:</b> {best_ov:.2f}%<br>
        <b>📏 Closest Distance Score:</b> {best_dist:.2f}%
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Backtest (position-based) ---
    st.subheader("Backtest (last 20 draws • top-weighted set only)")
    N = min(20, len(df_sorted) - 1)
    bt_rows = []
    for k in range(N):
        history = df_sorted.iloc[k + 1:].reset_index(drop=True)
        if history.empty:
            continue
        wfreq_k = weighted_frequency(history, half_life_draws=decay)
        top_set_k = wfreq_k['Number'].head(NUMBERS_PER_DRAW).tolist()
        actual_k = [int(df_sorted.iloc[k][f"n{i}"]) for i in range(1, NUMBERS_PER_DRAW + 1)]
        bt_rows.append({
            "Draw Date": df_sorted.iloc[k]['date'].strftime('%Y-%m-%d') if pd.notna(df_sorted.iloc[k]['date']) else "",
            "Predicted (top-weighted)": ", ".join(map(str, top_set_k)),
            "Actual": ", ".join(map(str, actual_k)),
            "Position Match (%)": round(position_match_pct(top_set_k, actual_k), 2),
            "Overlap (%)": round(closeness_overlap_pct(top_set_k, actual_k), 2),
            "Distance Score (%)": round(avg_min_distance_score(top_set_k, actual_k), 2)
        })
    if bt_rows:
        st.dataframe(pd.DataFrame(bt_rows), use_container_width=True)
    else:
        st.info("Not enough dated draws to backtest.")
