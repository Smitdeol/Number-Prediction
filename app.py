# streamlit_app.py
import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
from collections import Counter
import random
import os
import time
import re
import io
import plotly.express as px
from datetime import datetime, timedelta

# ---------- Config ----------
st.set_page_config(page_title="Magnum Life Dashboard", layout="wide")
st.title("🎯 Magnum Life Prediction Dashboard")

LOTTOLYZER_BASE = "https://en.lottolyzer.com/history/malaysia/magnum-life/page/{}/per-page/50/number-view"
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/115.0.0.0 Safari/537.36")
}
CACHE_FILENAME = "past_results.csv"
CACHE_MAX_AGE_HOURS = 24
PAGES_TO_SCRAPE = 5
NUMBERS_PER_DRAW = 8
TOTAL_NUMBERS = 36

# ---------- Helpers ----------
def is_cache_fresh(path=CACHE_FILENAME, max_age_hours=CACHE_MAX_AGE_HOURS):
    if not os.path.exists(path):
        return False
    mtime = os.path.getmtime(path)
    age = datetime.now() - datetime.fromtimestamp(mtime)
    return age < timedelta(hours=max_age_hours)

def save_cache(df, path=CACHE_FILENAME):
    df.to_csv(path, index=False)

def load_cache(path=CACHE_FILENAME):
    return pd.read_csv(path, parse_dates=['date'], dayfirst=True, infer_datetime_format=True)

def scrape_lottolyzer_pages(pages=1):
    rows = []
    date_regex = re.compile(r'(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\w+\s+\d{1,2},\s*\d{4})')
    for page in range(1, pages+1):
        url = LOTTOLYZER_BASE.format(page)
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        found = False
        for tr in soup.select("tr"):
            date_text = None
            td_date = tr.find("td", class_=re.compile("date", re.I)) or tr.find("td")
            if td_date:
                txt = td_date.get_text(" ", strip=True)
                m = date_regex.search(txt)
                if m:
                    date_text = m.group(0)
            nums = []
            for img in tr.find_all("img", alt=True):
                alt = img.get("alt", "").strip()
                if alt.isdigit():
                    nums.append(int(alt))
            if not nums:
                for td in tr.find_all("td"):
                    if td.get("class") and any("number" in c.lower() for c in td.get("class")):
                        t = td.get_text(strip=True)
                        if t.isdigit():
                            nums.append(int(t))
            if nums:
                found = True
                if len(nums) >= NUMBERS_PER_DRAW:
                    rows.append({
                        "date": date_text,
                        **{f"n{i+1}": nums[i] for i in range(NUMBERS_PER_DRAW)}
                    })
        if not found:
            imgs = soup.find_all("img", alt=lambda a: a and a.strip().isdigit())
            nums = [int(img['alt'].strip()) for img in imgs]
            for i in range(0, len(nums), NUMBERS_PER_DRAW):
                group = nums[i:i+NUMBERS_PER_DRAW]
                if len(group) == NUMBERS_PER_DRAW:
                    rows.append({
                        "date": None,
                        **{f"n{i+1}": group[i] for i in range(NUMBERS_PER_DRAW)}
                    })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce', dayfirst=True, infer_datetime_format=True)
    df = df.drop_duplicates().reset_index(drop=True)
    return df

def build_frequency(df):
    nums = []
    for i in range(1, NUMBERS_PER_DRAW+1):
        col = f"n{i}"
        if col in df.columns:
            nums.extend(df[col].dropna().astype(int).tolist())
    cnt = Counter(nums)
    freq_df = pd.DataFrame([(n, cnt.get(n,0)) for n in range(1, TOTAL_NUMBERS+1)], columns=["Number","Frequency"])
    freq_df = freq_df.sort_values(by="Frequency", ascending=False).reset_index(drop=True)
    return freq_df

def top_three_sets_by_freq(freq_df):
    top24 = freq_df['Number'].tolist()[:24]
    sets = []
    for i in range(3):
        start = i*8
        subset = sorted(top24[start:start+8])
        sets.append(subset)
    return sets

def render_number_badge(n, hot_set):
    # keep in single line with white-space: nowrap
    style = "display:inline-block;margin:3px;padding:6px 10px;border-radius:8px;white-space:nowrap;"
    if n in hot_set:
        return f"<span style='{style}background:#ffecec;color:#b30000;font-weight:700'>{n}</span>"
    else:
        return f"<span style='{style}background:#eef6ff;color:#0a3f6b'>{n}</span>"

# ---------- Load or scrape data ----------
st.info("Loading past results (cache-aware)...")
df = None
try:
    if is_cache_fresh():
        try:
            df = load_cache()
            st.success(f"Loaded past results from cache ({len(df)} draws).")
        except Exception:
            df = None

    if df is None:
        with st.spinner("Scraping history from Lottolyzer (this may take a few seconds)..."):
            scraped = scrape_lottolyzer_pages(pages=PAGES_TO_SCRAPE)
            if scraped.empty:
                st.error("Unable to parse any draws from Lottolyzer. Please check source or try again later.")
                st.stop()
            for i in range(1, NUMBERS_PER_DRAW+1):
                col = f"n{i}"
                if col not in scraped.columns:
                    scraped[col] = None
            scraped = scraped[['date'] + [f"n{i}" for i in range(1, NUMBERS_PER_DRAW+1)]]
            scraped.to_csv(CACHE_FILENAME, index=False)
            df = scraped
            st.success(f"Scraped and cached {len(df)} draws.")
except Exception as e:
    st.error(f"Failed to load or scrape data: {e}")
    st.stop()

# ---------- Build analytics ----------
freq_df = build_frequency(df)
sets = top_three_sets_by_freq(freq_df)
hot_numbers = set(freq_df['Number'].tolist()[:8])

# ---------- UI Layout ----------
tab1, tab2, tab3 = st.tabs(["Predictions", "Past Results", "Frequency Graph"])

with tab1:
    st.header("Top 3 Predicted Sets (based on historical frequency)")
    st.write(f"*Based on {len(df)} draws | Cache age: {time.ctime(os.path.getmtime(CACHE_FILENAME))}*")
    for i, s in enumerate(sets, start=1):
        badges = " ".join(render_number_badge(n, hot_numbers) for n in s)
        st.markdown(f"<div style='white-space:nowrap'><b>Set {i}</b> — {badges}</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("Quick Picks (randomized — for variety)")
    quicks = [sorted(random.sample(range(1, TOTAL_NUMBERS+1), NUMBERS_PER_DRAW)) for _ in range(5)]
    for i, q in enumerate(quicks, start=1):
        badges = " ".join(render_number_badge(n, hot_numbers) for n in q)
        st.markdown(f"<div style='white-space:nowrap'><b>Quick {i}</b> — {badges}</div>", unsafe_allow_html=True)

with tab2:
    st.header("Past Draws (collapsed by default)")
    with st.expander("Show past draws table (click to expand)", expanded=False):
        display_df = df.copy()
        if 'date' in display_df.columns:
            display_df['date'] = pd.to_datetime(display_df['date'], errors='coerce')
        display_df = display_df.rename(columns={f"n{i}": f"{i}" for i in range(1, NUMBERS_PER_DRAW+1)})
        cols = ['date'] + [str(i) for i in range(1, NUMBERS_PER_DRAW+1)]
        available_cols = [c for c in cols if c in display_df.columns]
        st.dataframe(display_df[available_cols].sort_values(by='date', ascending=False).reset_index(drop=True), use_container_width=True)
        csv_buf = io.StringIO()
        display_df.to_csv(csv_buf, index=False)
        st.download_button("Download past_results.csv", csv_buf.getvalue(), file_name="past_results.csv", mime="text/csv")

with tab3:
    st.header("Frequency Graph")
    st.write("Top numbers by historical frequency (higher = hotter)")
    fig = px.bar(freq_df, x='Number', y='Frequency', title='Number Frequency', labels={'Frequency':'Count','Number':'Number'})
    st.plotly_chart(fig, use_container_width=True)
    freq_buf = io.StringIO()
    freq_df.to_csv(freq_buf, index=False)
    st.download_button("Download frequency.csv", freq_buf.getvalue(), file_name="frequency.csv", mime="text/csv")

st.caption("Disclaimer: predictions are probabilistic suggestions based on past draws. No guarantee of winnings.")
