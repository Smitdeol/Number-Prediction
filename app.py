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

# Known pagination has 24 pages (user-confirmed). We'll iterate 1..24 and also stop early if empty.
LAST_PAGE = 24
BASE_URL = "https://en.lottolyzer.com/history/malaysia/magnum-life/page/{}/per-page/50/number-view"
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/115.0.0.0 Safari/537.36")
}
REQUEST_TIMEOUT = 20
PAUSE_BETWEEN_PAGE_REQUESTS = 0.35  # be polite to the site

NUMBERS_PER_DRAW = 8
TOTAL_NUMBERS = 36

# Local CSV cache (in Streamlit Cloud working dir)
CACHE_FILENAME = "past_results.csv"
CACHE_MAX_AGE_HOURS = 24

# Prediction defaults
DEFAULT_PREDICTION_COUNT = 3
DEFAULT_DECAY_HALFLIFE_DRAWS = 60  # recency weighting half-life (draws)


# ===========================
# Helpers: file cache
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
    # We parse 'date' column as datetime; tolerate mixed formats.
    return pd.read_csv(path, parse_dates=['date'], dayfirst=True)


# ===========================
# Scraper (robust selectors)
# ===========================
DATE_REGEX = re.compile(
    r'(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|'
    r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},\s*\d{4}|'
    r'\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4})',
    re.I
)

def _parse_row_for_numbers(tr) -> list[int]:
    """Extract exactly 8 numbers from a <tr> using multiple strategies."""
    nums = []

    # Strategy A: images with alt digits (common in 'number-view')
    for img in tr.find_all("img", alt=True):
        alt = (img.get("alt") or "").strip()
        if alt.isdigit():
            n = int(alt)
            if 1 <= n <= TOTAL_NUMBERS:
                nums.append(n)

    # Strategy B: digits contained in elements with class containing 'number' or 'ball'
    if len(nums) < NUMBERS_PER_DRAW:
        for td in tr.find_all(["td", "div", "span"]):
            classes = " ".join(td.get("class", [])) if td.get("class") else ""
            if any(key in classes.lower() for key in ["number", "num", "ball"]):
                txt = td.get_text(" ", strip=True)
                for token in re.findall(r"\b\d{1,2}\b", txt):
                    n = int(token)
                    if 1 <= n <= TOTAL_NUMBERS:
                        nums.append(n)
                        if len(nums) == NUMBERS_PER_DRAW:
                            break
            if len(nums) == NUMBERS_PER_DRAW:
                break

    # Strategy C: last resort - any digits in the row (kept safe by range check)
    if len(nums) < NUMBERS_PER_DRAW:
        txt = tr.get_text(" ", strip=True)
        for token in re.findall(r"\b\d{1,2}\b", txt):
            n = int(token)
            if 1 <= n <= TOTAL_NUMBERS:
                nums.append(n)
                if len(nums) == NUMBERS_PER_DRAW:
                    break

    return nums[:NUMBERS_PER_DRAW]


def _parse_row_for_date(tr) -> str | None:
    """Find a date-like string inside the row (prefer first cell)."""
    tds = tr.find_all(["td", "div", "span"])
    # Prefer first cell-like element
    if tds:
        txt0 = tds[0].get_text(" ", strip=True)
        m0 = DATE_REGEX.search(txt0)
        if m0:
            return m0.group(0)

    # Fallback: any date in the row
    m = DATE_REGEX.search(tr.get_text(" ", strip=True))
    return m.group(0) if m else None


def _fallback_grouping_from_images(soup) -> list[dict]:
    """If no table rows found, group all <img alt=digit> by 8 sequentially."""
    rows = []
    imgs = soup.find_all("img", alt=lambda a: a and a.strip().isdigit())
    digits = [int(img['alt'].strip()) for img in imgs if img.get('alt', '').strip().isdigit()]
    for i in range(0, len(digits), NUMBERS_PER_DRAW):
        group = digits[i:i+NUMBERS_PER_DRAW]
        if len(group) == NUMBERS_PER_DRAW:
            rows.append({
                "date": None,
                **{f"n{j+1}": group[j] for j in range(NUMBERS_PER_DRAW)}
            })
    return rows


def scrape_all_history(last_page: int = LAST_PAGE) -> pd.DataFrame:
    """
    Scrape all pages (1..last_page), robustly parsing date + 8 numbers per draw.
    Stops early if a page yields no valid rows.
    """
    rows = []
    seen = set()

    for page in range(1, last_page + 1):
        url = BASE_URL.format(page)
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        if resp.status_code != 200:
            # If a mid-series page fails (e.g., 500), try next page anyway.
            time.sleep(PAUSE_BETWEEN_PAGE_REQUESTS)
            continue

        soup = BeautifulSoup(resp.text, "lxml")

        # Primary: table rows
        trs = soup.select("table tbody tr")
        if not trs:
            trs = soup.find_all("tr")

        page_new = 0
        for tr in trs:
            nums = _parse_row_for_numbers(tr)
            if len(nums) != NUMBERS_PER_DRAW:
                continue
            date_text = _parse_row_for_date(tr)

            # Build row dict
            d = {'date': date_text}
            for i, n in enumerate(nums, start=1):
                d[f"n{i}"] = n

            # Deduplicate by (date, tuple(nums)) when date found; else by nums only
            key = (date_text, tuple(nums)) if date_text else tuple(nums)
            if key in seen:
                continue
            seen.add(key)
            rows.append(d)
            page_new += 1

        # Fallback if table parsing failed completely
        if page_new == 0:
            fallback_rows = _fallback_grouping_from_images(soup)
            for d in fallback_rows:
                key = (d.get('date'), tuple(d[f"n{i}"] for i in range(1, NUMBERS_PER_DRAW+1)))
                if key in seen:
                    continue
                seen.add(key)
                rows.append(d)
                page_new += 1

        # If a whole page yields nothing, likely the end
        if page_new == 0:
            break

        time.sleep(PAUSE_BETWEEN_PAGE_REQUESTS)

    if not rows:
        return pd.DataFrame(columns=['date'] + [f"n{i}" for i in range(1, NUMBERS_PER_DRAW+1)])

    df = pd.DataFrame(rows)

    # Normalize date: multiple formats tolerated
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce', dayfirst=True, infer_datetime_format=True)

    # Sort newest first (NaT last)
    df = df.sort_values(by='date', ascending=False, na_position='last').reset_index(drop=True)

    # Ensure correct columns order & dtypes
    for i in range(1, NUMBERS_PER_DRAW+1):
        col = f"n{i}"
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

    # Drop perfect duplicates
    df = df.drop_duplicates(subset=['date'] + [f"n{i}" for i in range(1, NUMBERS_PER_DRAW+1)],
                            keep='first').reset_index(drop=True)
    return df


# ===========================
# Analytics / Prediction (“AI-like”)
# ===========================
def weighted_frequency(df: pd.DataFrame, half_life_draws: int) -> pd.DataFrame:
    """
    Exponential recency weighting:
      weight = 0.5 ** (age / half_life_draws), where age=0 for most recent row.
    Returns DataFrame(Number, Weight, Rank).
    """
    if df.empty:
        return pd.DataFrame(columns=['Number', 'Weight', 'Rank'])

    # Age by row index (df already sorted newest->oldest)
    ages = np.arange(len(df))
    weights = 0.5 ** (ages / max(1, half_life_draws))

    # Accumulate weights per number across all draws
    scores = {n: 0.0 for n in range(1, TOTAL_NUMBERS + 1)}
    for idx, w in enumerate(weights):
        row = df.iloc[idx]
        for i in range(1, NUMBERS_PER_DRAW + 1):
            val = row.get(f"n{i}")
            if pd.notna(val):
                v = int(val)
                if 1 <= v <= TOTAL_NUMBERS:
                    scores[v] += w

    wf = pd.DataFrame(sorted(scores.items(), key=lambda x: x[1], reverse=True),
                      columns=['Number', 'Weight'])
    wf['Rank'] = np.arange(1, len(wf) + 1)
    return wf


def generate_predictions(wfreq: pd.DataFrame, count: int) -> list[list[int]]:
    """
    Create `count` predicted sets:
      - Set 1: strict top-8 by weight (deterministic)
      - Sets 2..count: weighted sampling without replacement (diversified)
    """
    if wfreq.empty or wfreq['Weight'].sum() <= 0:
        return [sorted(random.sample(range(1, TOTAL_NUMBERS + 1), NUMBERS_PER_DRAW)) for _ in range(count)]

    numbers = wfreq['Number'].to_numpy()
    probs = wfreq['Weight'].to_numpy()
    probs = probs / probs.sum()

    preds = []
    # Set 1: top-8
    preds.append(sorted(numbers[:NUMBERS_PER_DRAW].tolist()))

    # Sets 2..count: sample without replacement using current probs
    for _ in range(1, count):
        chosen = []
        mask = np.ones_like(numbers, dtype=bool)
        for _pick in range(NUMBERS_PER_DRAW):
            avail_nums = numbers[mask]
            avail_probs = probs[mask]
            avail_probs = avail_probs / avail_probs.sum()
            pick = int(np.random.choice(avail_nums, p=avail_probs))
            chosen.append(pick)
            # remove picked index
            mask[np.where(numbers == pick)[0][0]] = False
        preds.append(sorted(chosen))
    return preds


def closeness_overlap_pct(pred: list[int], actual: list[int]) -> float:
    """Simple overlap: how many of the 8 match (0..100%)."""
    return 100.0 * len(set(pred) & set(actual)) / NUMBERS_PER_DRAW


def avg_min_distance_score(pred: list[int], actual: list[int]) -> float:
    """
    Distance-based score: for each predicted number, distance to nearest actual.
    Convert to 0..100 where 100=identical, 0=far away on average.
    """
    if not pred or not actual:
        return 0.0
    dists = []
    for p in pred:
        dists.append(min(abs(p - a) for a in actual))
    avg_d = float(np.mean(dists)) if dists else (TOTAL_NUMBERS / 2.0)
    norm = (TOTAL_NUMBERS / 2.0)
    return max(0.0, 100.0 * (1.0 - (avg_d / norm)))


def render_badge(n: int, hot: set[int]) -> str:
    style = "display:inline-block;margin:3px;padding:6px 10px;border-radius:8px;white-space:nowrap;"
    if n in hot:
        return f"<span style='{style}background:#ffecec;color:#b30000;font-weight:700'>{n}</span>"
    return f"<span style='{style}background:#eef6ff;color:#0a3f6b'>{n}</span>"


# ===========================
# Sidebar controls
# ===========================
with st.sidebar:
    st.subheader("Data & Model Settings")
    force_rescrape = st.button("🔄 Force re-scrape now")
    decay = st.slider("Recency half-life (draws)", min_value=10, max_value=200,
                      value=DEFAULT_DECAY_HALFLIFE_DRAWS, step=5)
    pred_count = st.slider("Number of prediction sets", min_value=1, max_value=10,
                           value=DEFAULT_PREDICTION_COUNT, step=1)


# ===========================
# Data loading (cache-aware)
# ===========================
st.info("Loading past results (cache-aware)…")

df = None
if not force_rescrape and is_cache_fresh():
    try:
        df = load_cache()
        st.success(f"Loaded past results from cache ({len(df)} draws).")
    except Exception:
        df = None

if df is None or force_rescrape:
    with st.spinner("Scraping all 24 pages from Lottolyzer…"):
        df = scrape_all_history(last_page=LAST_PAGE)
        if df.empty:
            st.error("Unable to parse any draws from Lottolyzer. Please try again later.")
            st.stop()
        save_cache(df)
        st.success(f"Scraped {len(df)} draws and cached.")
else:
    # Ensure types are correct even if coming from cache
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce', dayfirst=True)


# ===========================
# Analytics & Predictions
# ===========================
wfreq = weighted_frequency(df, half_life_draws=decay)
hot_numbers = set(wfreq['Number'].head(8).tolist())
predictions = generate_predictions(wfreq, count=pred_count)

# ===========================
# Layout: Tabs
# ===========================
tab_pred, tab_compare, tab_past, tab_freq = st.tabs([
    "Predictions", "Actual vs Predicted", "Past Results", "Frequency Graphs"
])

with tab_pred:
    st.header("Dynamic Predictions (recency-weighted)")
    st.write(f"*Using {len(df)} historical draws • Half-life = {decay} draws*")
    for i, s in enumerate(predictions, start=1):
        badges = " ".join(render_badge(n, hot_numbers) for n in s)
        st.markdown(f"<div style='white-space:nowrap'><b>Set {i}</b> — {badges}</div>",
                    unsafe_allow_html=True)
    st.caption("Set 1 is the deterministic top-weighted set. Others are weighted samples for diversity.")

with tab_compare:
    st.header("Actual vs Predicted (latest dated draw)")
    df_sorted = df.sort_values(by='date', ascending=False, na_position='last').reset_index(drop=True)
    # choose latest row that has a valid date & 8 numbers
    latest_idx = df_sorted['date'].first_valid_index()
    if latest_idx is None:
        latest_idx = 0
    latest_row = df_sorted.iloc[latest_idx]
    actual = [int(latest_row[f"n{i}"]) for i in range(1, NUMBERS_PER_DRAW + 1)]

    records = []
    for i, pred in enumerate(predictions, start=1):
        records.append({
            "Prediction Set": i,
            "Predicted": ", ".join(map(str, pred)),
            "Actual (latest)": ", ".join(map(str, actual)),
            "Overlap (%)": round(closeness_overlap_pct(pred, actual), 2),
            "Distance Score (%)": round(avg_min_distance_score(pred, actual), 2)
        })
    comp_df = pd.DataFrame(records)
    st.dataframe(comp_df, use_container_width=True)

    # Backtest on last N draws (top-weighted only, walk-forward without look-ahead)
    st.subheader("Backtest (last 20 draws • top-weighted set only)")
    N = min(20, len(df_sorted) - 1) if len(df_sorted) > 1 else 0
    bt_rows = []
    for k in range(N):
        history = df_sorted.iloc[k+1:].reset_index(drop=True)
        if history.empty:
            continue
        wfreq_k = weighted_frequency(history, half_life_draws=decay)
        top_set_k = wfreq_k['Number'].head(NUMBERS_PER_DRAW).tolist()
        actual_k = [int(df_sorted.iloc[k][f"n{i}"]) for i in range(1, NUMBERS_PER_DRAW + 1)]
        bt_rows.append({
            "Draw Date": df_sorted.iloc[k]['date'].strftime('%Y-%m-%d') if pd.notna(df_sorted.iloc[k]['date']) else "",
            "Predicted (top-weighted)": ", ".join(map(str, top_set_k)),
            "Actual": ", ".join(map(str, actual_k)),
            "Overlap (%)": round(closeness_overlap_pct(top_set_k, actual_k), 2),
            "Distance Score (%)": round(avg_min_distance_score(top_set_k, actual_k), 2)
        })
    if bt_rows:
        st.dataframe(pd.DataFrame(bt_rows), use_container_width=True)
    else:
        st.info("Not enough dated draws to backtest.")

with tab_past:
    st.header("Past Draws (collapsed by default)")
    with st.expander("Show full past draws table (click to expand)", expanded=False):
        display_df = df.copy()
        display_df['date'] = pd.to_datetime(display_df['date'], errors='coerce', dayfirst=True)
        display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
        cols = ['date'] + [f"n{i}" for i in range(1, NUMBERS_PER_DRAW + 1)]
        display_df = display_df[cols]
        st.dataframe(display_df.sort_values(by='date', ascending=False, na_position='last').reset_index(drop=True),
                     use_container_width=True)

        # Download CSV
        buf = io.StringIO()
        display_df.to_csv(buf, index=False)
        st.download_button("Download past_results.csv", buf.getvalue(),
                           file_name="past_results.csv", mime="text/csv")

with tab_freq:
    st.header("Recency-Weighted Number Strength")
    fig = px.bar(wfreq, x='Number', y='Weight', title='Recency-Weighted Number Strength',
                 labels={'Weight': 'Weighted Frequency'})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Raw Frequency (all history, unweighted)")
    all_nums = []
    for i in range(1, NUMBERS_PER_DRAW + 1):
        col = f"n{i}"
        if col in df.columns:
            all_nums.extend(df[col].dropna().astype(int).tolist())
    cnt = Counter(all_nums)
    raw_freq = pd.DataFrame(sorted(cnt.items()), columns=['Number', 'Frequency'])
    fig2 = px.bar(raw_freq, x='Number', y='Frequency', title='Raw Frequency')
    st.plotly_chart(fig2, use_container_width=True)

st.caption("Disclaimer: This tool uses historical data and probabilistic methods. No guarantees of winnings.")
