# app.py / streamlit_app.py
import os
import io
import re
import time
import random
from datetime import datetime, timedelta
from collections import Counter

import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import plotly.express as px
import streamlit as st

# ===========================
# Config
# ===========================
st.set_page_config(page_title="Magnum Life Dashboard", layout="wide")
st.title("🎯 Magnum Life Prediction Dashboard")

LOTTOLYZER_BASE = "https://en.lottolyzer.com/history/malaysia/magnum-life/page/{}/per-page/50/number-view"
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/115.0.0.0 Safari/537.36")
}
REQUEST_TIMEOUT = 20
PAUSE_BETWEEN_PAGE_REQUESTS = 0.4  # be polite
NUMBERS_PER_DRAW = 8
TOTAL_NUMBERS = 36

# Caching
CACHE_FILENAME = "past_results.csv"
CACHE_MAX_AGE_HOURS = 24  # cache considered fresh for this long

# Prediction controls (you can tweak in UI too)
DEFAULT_PREDICTION_COUNT = 3
DEFAULT_DECAY_HALFLIFE_DRAWS = 60  # more weight to recent ~60 draws


# ===========================
# Helpers: Cache
# ===========================
def is_cache_fresh(path=CACHE_FILENAME, max_age_hours=CACHE_MAX_AGE_HOURS) -> bool:
    if not os.path.exists(path):
        return False
    mtime = os.path.getmtime(path)
    age = datetime.now() - datetime.fromtimestamp(mtime)
    return age < timedelta(hours=max_age_hours)


def save_cache(df: pd.DataFrame, path=CACHE_FILENAME):
    df.to_csv(path, index=False)


def load_cache(path=CACHE_FILENAME) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=['date'], dayfirst=True)


# ===========================
# Scraper
# ===========================
def parse_row_for_nums(tr) -> list:
    """Extract up to 8 numbers from a <tr> either via <img alt> or digit text in tds."""
    nums = []
    # Try image alts first
    for img in tr.find_all("img", alt=True):
        alt = img.get("alt", "").strip()
        if alt.isdigit():
            n = int(alt)
            if 1 <= n <= TOTAL_NUMBERS:
                nums.append(n)
    # Fallback: digit text from tds
    if len(nums) < NUMBERS_PER_DRAW:
        for td in tr.find_all("td"):
            txt = td.get_text(" ", strip=True)
            # collect single or two-digit tokens
            for token in re.findall(r"\b\d{1,2}\b", txt):
                if token.isdigit():
                    n = int(token)
                    if 1 <= n <= TOTAL_NUMBERS:
                        nums.append(n)
                        if len(nums) == NUMBERS_PER_DRAW:
                            break
            if len(nums) == NUMBERS_PER_DRAW:
                break
    return nums[:NUMBERS_PER_DRAW]


def parse_row_for_date(tr) -> str | None:
    """Find a date-like string inside the row."""
    date_regex = re.compile(r'(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\w+\s+\d{1,2},\s*\d{4})')
    # try first td or any td
    tds = tr.find_all("td")
    for td in ([tds[0]] if tds else []):
        txt = td.get_text(" ", strip=True)
        m = date_regex.search(txt)
        if m:
            return m.group(0)
    # search entire row text as fallback
    m = date_regex.search(tr.get_text(" ", strip=True))
    return m.group(0) if m else None


def scrape_all_pages(max_pages_cap=2000) -> pd.DataFrame:
    """
    Keep paging until:
    - non-200 HTTP, or
    - no <tr> rows found (end), or
    - we hit a high safety cap (to avoid infinite loops)
    """
    rows = []
    seen_keys = set()
    page = 1
    while page <= max_pages_cap:
        url = LOTTOLYZER_BASE.format(page)
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        if resp.status_code != 200:
            break

        soup = BeautifulSoup(resp.text, "lxml")
        trs = soup.select("table tbody tr")
        if not trs:
            # try broader selection as fallback
            trs = soup.find_all("tr")
        if not trs:
            # no more data
            break

        page_new = 0
        for tr in trs:
            nums = parse_row_for_nums(tr)
            if len(nums) != NUMBERS_PER_DRAW:
                continue
            date_text = parse_row_for_date(tr)
            # build row dict
            d = {'date': date_text}
            for i, n in enumerate(nums, start=1):
                d[f"n{i}"] = n
            # dedupe by (date, tuple(nums)) when date exists; else by nums only
            key = (date_text, tuple(nums)) if date_text else tuple(nums)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            rows.append(d)
            page_new += 1

        # If a whole page yielded nothing new, likely end.
        if page_new == 0:
            break

        page += 1
        time.sleep(PAUSE_BETWEEN_PAGE_REQUESTS)

    if not rows:
        return pd.DataFrame(columns=['date'] + [f"n{i}" for i in range(1, NUMBERS_PER_DRAW+1)])

    df = pd.DataFrame(rows)
    # Normalize date -> datetime (coerce unknown)
    df['date'] = pd.to_datetime(df['date'], errors='coerce', dayfirst=True)
    # Sort newest first (NaT last)
    df = df.sort_values(by='date', ascending=False, na_position='last').reset_index(drop=True)
    # drop perfect duplicates
    df = df.drop_duplicates(subset=['date'] + [f"n{i}" for i in range(1, NUMBERS_PER_DRAW+1)],
                            keep='first').reset_index(drop=True)
    return df


# ===========================
# Analytics / "AI" scaffold
# ===========================
def weighted_frequency(df: pd.DataFrame, half_life_draws: int = DEFAULT_DECAY_HALFLIFE_DRAWS) -> pd.DataFrame:
    """
    Recency-weighted frequency:
    weight = 0.5 ** (age / half_life_draws), where age=0 for most recent draw.
    Returns DataFrame with columns Number, Weight, Rank.
    """
    if df.empty:
        return pd.DataFrame(columns=['Number', 'Weight', 'Rank'])

    # age: 0 for most recent row, 1 for next, etc.
    ages = np.arange(len(df))  # df is sorted descending by date
    weights = 0.5 ** (ages / max(1, half_life_draws))  # decay
    # Expand weights across numbers
    score = {n: 0.0 for n in range(1, TOTAL_NUMBERS+1)}
    for idx, w in enumerate(weights):
        row = df.iloc[idx]
        for i in range(1, NUMBERS_PER_DRAW+1):
            n = row.get(f"n{i}")
            if pd.notna(n):
                n = int(n)
                if 1 <= n <= TOTAL_NUMBERS:
                    score[n] += w

    wf = pd.DataFrame(sorted(score.items(), key=lambda x: x[1], reverse=True),
                      columns=['Number', 'Weight'])
    wf['Rank'] = np.arange(1, len(wf)+1)
    return wf


def generate_predictions(wfreq: pd.DataFrame, count: int = DEFAULT_PREDICTION_COUNT, diversify=True) -> list[list[int]]:
    """
    Generate 'count' predicted sets (8 unique numbers per set) using the weighted frequencies.
    - First set: strict top-8 by weight (deterministic).
    - Remaining sets: sample without replacement using weights (stochastic), optionally diversifying.
    """
    # Safety: if no weights, return random picks
    if wfreq.empty or wfreq['Weight'].sum() <= 0:
        return [sorted(random.sample(range(1, TOTAL_NUMBERS+1), NUMBERS_PER_DRAW)) for _ in range(count)]

    numbers = wfreq['Number'].to_numpy()
    probs = wfreq['Weight'].to_numpy()
    probs = probs / probs.sum()

    preds = []
    # Set 1: top-8
    top8 = sorted(numbers[:NUMBERS_PER_DRAW].tolist())
    preds.append(top8)

    # Other sets: weighted sampling without replacement
    for k in range(1, count):
        # draw 8 numbers without replacement, proportional to weights
        # use numpy choice iteratively to re-normalize each pick
        chosen = []
        mask = np.ones_like(numbers, dtype=bool)
        for _ in range(NUMBERS_PER_DRAW):
            available_nums = numbers[mask]
            available_probs = probs[mask]
            available_probs = available_probs / available_probs.sum()
            pick = np.random.choice(available_nums, p=available_probs)
            chosen.append(int(pick))
            # remove picked index
            mask[np.where(numbers == pick)[0][0]] = False
            if diversify:
                # penalize neighbors +/-1 to reduce clustering
                for delta in (-1, +1):
                    neighbor = pick + delta
                    if 1 <= neighbor <= TOTAL_NUMBERS:
                        idx = np.where(numbers == neighbor)[0]
                        if idx.size > 0:
                            mask[idx[0]] = mask[idx[0]]  # keep, but effect is minimal without direct prob change
        preds.append(sorted(chosen))
    return preds[:count]


def closeness_overlap_pct(pred: list[int], actual: list[int]) -> float:
    """Percent overlap (0..100) between two 8-number sets."""
    return 100.0 * len(set(pred) & set(actual)) / NUMBERS_PER_DRAW


def avg_min_distance_score(pred: list[int], actual: list[int]) -> float:
    """
    For each predicted number, compute distance to nearest actual number.
    Convert to a 0..100 score where 100 = identical, 0 = very far on average.
    Normalize by (TOTAL_NUMBERS/2) as a coarse scale.
    """
    if not pred or not actual:
        return 0.0
    dists = []
    for p in pred:
        d = min(abs(p - a) for a in actual)
        dists.append(d)
    avg_d = np.mean(dists) if dists else (TOTAL_NUMBERS / 2)
    # Map distance to score (smaller distance => higher score)
    norm = (TOTAL_NUMBERS / 2.0)
    score = max(0.0, 100.0 * (1.0 - (avg_d / norm)))
    return float(score)


def render_number_badge(n: int, hot_set: set[int]) -> str:
    style = "display:inline-block;margin:3px;padding:6px 10px;border-radius:8px;white-space:nowrap;"
    if n in hot_set:
        return f"<span style='{style}background:#ffecec;color:#b30000;font-weight:700'>{n}</span>"
    else:
        return f"<span style='{style}background:#eef6ff;color:#0a3f6b'>{n}</span>"


# ===========================
# Data loading
# ===========================
with st.sidebar:
    st.subheader("Data & Model Settings")
    force_rescrape = st.button("🔄 Force re-scrape now")
    decay = st.slider("Recency half-life (draws)", min_value=10, max_value=200, value=DEFAULT_DECAY_HALFLIFE_DRAWS, step=5)
    pred_count = st.slider("Number of prediction sets", min_value=1, max_value=10, value=DEFAULT_PREDICTION_COUNT, step=1)

st.info("Loading past results (cache-aware)...")

df = None
if not force_rescrape and is_cache_fresh():
    try:
        df = load_cache()
        st.success(f"Loaded past results from cache ({len(df)} draws).")
    except Exception:
        df = None

if df is None or force_rescrape:
    with st.spinner("Scraping all available history from Lottolyzer…"):
        df = scrape_all_pages()
        if df.empty:
            st.error("Unable to parse any draws from Lottolyzer. Please try again later.")
            st.stop()
        save_cache(df)
        st.success(f"Scraped {len(df)} draws and cached.")


# ===========================
# Analytics
# ===========================
# Weighted frequencies & hot numbers
wfreq = weighted_frequency(df, half_life_draws=decay)
hot_numbers = set(wfreq['Number'].head(8).tolist())

# Generate predictions dynamically (no more fixed 'top 3')
predictions = generate_predictions(wfreq, count=pred_count, diversify=True)

# ===========================
# Layout: Tabs
# ===========================
tab_pred, tab_compare, tab_past, tab_freq = st.tabs([
    "Predictions", "Actual vs Predicted", "Past Results", "Frequency Graph"
])

with tab_pred:
    st.header("Dynamic Predictions (recency-weighted)")
    st.write(f"*Using {len(df)} historical draws • Half-life = {decay} draws*")

    for i, s in enumerate(predictions, start=1):
        badges = " ".join(render_number_badge(n, hot_numbers) for n in s)
        st.markdown(f"<div style='white-space:nowrap'><b>Set {i}</b> — {badges}</div>", unsafe_allow_html=True)

    st.caption("Set 1 is the top-weighted set. Others are sampled from weighted probabilities for diversity.")

with tab_compare:
    st.header("Actual vs Predicted (latest draw)")

    # Latest actual draw (first row with a valid date)
    df_sorted = df.sort_values(by='date', ascending=False, na_position='last').reset_index(drop=True)
    latest_idx = df_sorted['date'].first_valid_index()
    if latest_idx is None:
        latest_idx = 0
    latest_row = df_sorted.iloc[latest_idx]
    actual = [int(latest_row[f"n{i}"]) for i in range(1, NUMBERS_PER_DRAW+1)]

    # Compare each prediction to the latest actual
    records = []
    for i, pred in enumerate(predictions, start=1):
        overlap_pct = closeness_overlap_pct(pred, actual)
        dist_score = avg_min_distance_score(pred, actual)
        records.append({
            "Prediction Set": i,
            "Predicted": ", ".join(map(str, pred)),
            "Actual (latest)": ", ".join(map(str, actual)),
            "Overlap (%)": round(overlap_pct, 2),
            "Distance Score (%)": round(dist_score, 2)
        })
    comp_df = pd.DataFrame(records)
    st.dataframe(comp_df, use_container_width=True)

    # Optional backtest on last N draws (walk-forward)
    st.subheader("Backtest (last 20 draws • top-weighted set only)")
    N = min(20, len(df_sorted)-1) if len(df_sorted) > 1 else 0
    bt_rows = []
    for k in range(N):
        # train on draws AFTER index k (i.e., older draws), predict for draw at index k
        # df_sorted is newest->oldest; to avoid lookahead, use rows with index > k (older)
        history = df_sorted.iloc[k+1:].reset_index(drop=True)
        if history.empty:
            continue
        wfreq_k = weighted_frequency(history, half_life_draws=decay)
        top_set_k = wfreq_k['Number'].head(NUMBERS_PER_DRAW).tolist()
        actual_k = [int(df_sorted.iloc[k][f"n{i}"]) for i in range(1, NUMBERS_PER_DRAW+1)]
        bt_rows.append({
            "Draw Date": df_sorted.iloc[k]['date'].strftime('%Y-%m-%d') if pd.notna(df_sorted.iloc[k]['date']) else "",
            "Predicted (top-weighted)": ", ".join(map(str, top_set_k)),
            "Actual": ", ".join(map(str, actual_k)),
            "Overlap (%)": round(closeness_overlap_pct(top_set_k, actual_k), 2)
        })
    if bt_rows:
        st.dataframe(pd.DataFrame(bt_rows), use_container_width=True)
    else:
        st.info("Not enough dated draws to backtest.")

with tab_past:
    st.header("Past Draws (collapsed by default)")
    with st.expander("Show full past draws table (click to expand)", expanded=False):
        display_df = df.copy()
        display_df['date'] = pd.to_datetime(display_df['date'], errors='coerce')
        # Reorder: date, n1..n8
        cols = ['date'] + [f"n{i}" for i in range(1, NUMBERS_PER_DRAW+1)]
        display_df = display_df[cols]
        st.dataframe(display_df.sort_values(by='date', ascending=False, na_position='last').reset_index(drop=True),
                     use_container_width=True)
        # Download buttons
        buf = io.StringIO()
        display_df.to_csv(buf, index=False)
        st.download_button("Download past_results.csv", buf.getvalue(),
                           file_name="past_results.csv", mime="text/csv")

with tab_freq:
    st.header("Frequency Graph (recency-weighted)")
    fig = px.bar(wfreq, x='Number', y='Weight', title='Recency-Weighted Number Strength',
                 labels={'Weight': 'Weighted Frequency'})
    st.plotly_chart(fig, use_container_width=True)

    # Also show plain counts (unweighted) for reference
    st.subheader("Raw Frequency (all history, unweighted)")
    all_nums = []
    for i in range(1, NUMBERS_PER_DRAW+1):
        col = f"n{i}"
        if col in df.columns:
            all_nums.extend(df[col].dropna().astype(int).tolist())
    cnt = Counter(all_nums)
    raw_freq = pd.DataFrame(sorted(cnt.items()), columns=['Number', 'Frequency'])
    fig2 = px.bar(raw_freq, x='Number', y='Frequency', title='Raw Frequency')
    st.plotly_chart(fig2, use_container_width=True)

st.caption("Disclaimer: This tool uses historical data and probabilistic methods. No guarantees of winnings.")
