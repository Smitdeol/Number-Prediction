# streamlit_app.py
import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
from collections import Counter
import random
import re
import time
from datetime import datetime
import math

# ---------- Config ----------
st.set_page_config(page_title="Magnum Life - Full History", layout="wide")
st.title("🎯 Magnum Life — Full History Predictor")

LOTTOLYZER_PAGE_TEMPLATE = "https://en.lottolyzer.com/history/malaysia/magnum-life/page/{}/per-page/50/number-view"
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/115.0.0.0 Safari/537.36")
}
REQUEST_TIMEOUT = 15
PAUSE_BETWEEN_REQUESTS = 0.6  # seconds (be polite)
NUMBERS_PER_DRAW = 8
TOTAL_NUMBERS = 36
MAX_PAGES_SAFEGUARD = 200  # safety cap in case pagination is huge or site loops

# ---------- Helpers ----------
def fetch(url):
    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        return resp
    except Exception as e:
        raise RuntimeError(f"Request failed: {e}")

def detect_total_pages(soup):
    """Try to detect total pages from pagination links; fallback to scanning hrefs for /page/N/"""
    max_page = 1
    # look for common pagination containers
    for a in soup.select("a[href]"):
        href = a['href']
        m = re.search(r"/page/(\d+)", href)
        if m:
            try:
                p = int(m.group(1))
                if p > max_page:
                    max_page = p
            except:
                continue
    # also try text like 'Page 1 of 5'
    txt = soup.get_text(" ", strip=True)
    m2 = re.search(r"Page\s*\d+\s*of\s*(\d+)", txt, re.I)
    if m2:
        try:
            p = int(m2.group(1))
            if p > max_page:
                max_page = p
        except:
            pass
    # safety cap
    if max_page > MAX_PAGES_SAFEGUARD:
        max_page = MAX_PAGES_SAFEGUARD
    return max_page

def parse_draws_from_page(soup):
    """
    Best-effort parse: Lottolyzer draws often have date text and image tags where alt="13".
    We'll find draw containers (rows or list items) that include both a date and 8 images (alts).
    Fallback: group sequential img alts into draws of 8.
    """
    date_regex = re.compile(r"\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\w+\s+\d{1,2},\s*\d{4}")
    draws = []

    # attempt: find table rows (<tr>) that represent draws
    for tr in soup.select("tr"):
        # gather imgs alt numbers in this tr
        imgs = tr.find_all("img", alt=True)
        nums = [int(img['alt']) for img in imgs if img.get('alt','').strip().isdigit()]
        # try to find a date in any td/th text within the tr
        date_text = None
        for td in tr.find_all(["td","th"]):
            t = td.get_text(" ", strip=True)
            m = date_regex.search(t)
            if m:
                date_text = m.group(0)
                break
        if nums and len(nums) >= NUMBERS_PER_DRAW:
            draws.append({"date": date_text, "numbers": nums[:NUMBERS_PER_DRAW]})

    # fallback: look for div/list entries that look like draw containers
    if not draws:
        for entry in soup.select("div, li"):
            imgs = entry.find_all("img", alt=True)
            nums = [int(img['alt']) for img in imgs if img.get('alt','').strip().isdigit()]
            if nums and len(nums) >= NUMBERS_PER_DRAW:
                # find date nearby in entry text
                t = entry.get_text(" ", strip=True)
                m = date_regex.search(t)
                date_text = m.group(0) if m else None
                draws.append({"date": date_text, "numbers": nums[:NUMBERS_PER_DRAW]})

    # last fallback: gather all image alts on page and group sequentially
    if not draws:
        imgs = soup.find_all("img", alt=True)
        nums = [int(img['alt']) for img in imgs if img.get('alt','').strip().isdigit()]
        for i in range(0, len(nums), NUMBERS_PER_DRAW):
            group = nums[i:i+NUMBERS_PER_DRAW]
            if len(group) == NUMBERS_PER_DRAW:
                draws.append({"date": None, "numbers": group})

    return draws

def scrape_all_pages():
    """Scrape all available pages from Lottolyzer, detect page count automatically."""
    aggregated = []
    # fetch page 1 first to detect total pages
    url1 = LOTTOLYZER_PAGE_TEMPLATE.format(1)
    resp1 = fetch(url1)
    if resp1.status_code == 403 or resp1.status_code == 429:
        raise PermissionError(f"Access blocked by remote host (HTTP {resp1.status_code}). Try again later or run locally.")
    if resp1.status_code != 200:
        raise RuntimeError(f"Failed to fetch page 1 (HTTP {resp1.status_code})")
    soup1 = BeautifulSoup(resp1.text, "lxml")
    total_pages = detect_total_pages(soup1)
    # safety: if detection fails, default to 1
    if total_pages < 1:
        total_pages = 1

    # guard: if total_pages is huge, cap it (safeguard already in detect)
    total_pages = min(total_pages, MAX_PAGES_SAFEGUARD)

    # parse page1
    page_draws = parse_draws_from_page(soup1)
    aggregated.extend(page_draws)

    # iterate remaining pages
    if total_pages > 1:
        for p in range(2, total_pages + 1):
            url = LOTTOLYZER_PAGE_TEMPLATE.format(p)
            time.sleep(PAUSE_BETWEEN_REQUESTS)
            resp = fetch(url)
            if resp.status_code == 403 or resp.status_code == 429:
                # remote blocked us; fail gracefully and return what we have
                st.warning(f"Scraping blocked at page {p} (HTTP {resp.status_code}). Returning partial data.")
                break
            if resp.status_code != 200:
                st.warning(f"Failed to fetch page {p} (HTTP {resp.status_code}). Stopping pagination.")
                break
            soup = BeautifulSoup(resp.text, "lxml")
            page_draws = parse_draws_from_page(soup)
            if not page_draws:
                # if a page yields nothing, we continue but stop if many consecutive empty pages
                st.info(f"No draws parsed on page {p}; continuing.")
            aggregated.extend(page_draws)
    # deduplicate: some pages might duplicate draws; we remove exact duplicates by date+nums if date exists, else by nums sequence
    seen = set()
    unique = []
    for d in aggregated:
        key = None
        if d.get("date"):
            key = (d["date"], tuple(d["numbers"]))
        else:
            key = tuple(d["numbers"])
        if key not in seen:
            seen.add(key)
            unique.append(d)
    return unique

def draws_to_dataframe(draws):
    rows = []
    for d in draws:
        row = {}
        # parse/normalize date field to ISO if possible
        dt = None
        if d.get("date"):
            try:
                dt = pd.to_datetime(d["date"], errors='coerce', dayfirst=False, infer_datetime_format=True)
            except:
                dt = None
        row['date'] = dt
        for i in range(NUMBERS_PER_DRAW):
            row[f"n{i+1}"] = d["numbers"][i]
        rows.append(row)
    df = pd.DataFrame(rows)
    # if date column exists, sort descending by date (put NaT at bottom)
    if 'date' in df.columns:
        df = df.sort_values(by='date', ascending=False, na_position='last').reset_index(drop=True)
    return df

def build_frequency(df):
    all_nums = []
    for i in range(1, NUMBERS_PER_DRAW+1):
        col = f"n{i}"
        if col in df.columns:
            all_nums.extend(df[col].dropna().astype(int).tolist())
    cnt = Counter(all_nums)
    freq_list = [(n, cnt.get(n, 0)) for n in range(1, TOTAL_NUMBERS+1)]
    freq_df = pd.DataFrame(freq_list, columns=["Number", "Frequency"]).sort_values(by="Frequency", ascending=False).reset_index(drop=True)
    return freq_df

def top3_sets_from_freq(freq_df):
    top24 = freq_df['Number'].tolist()[:24]
    sets = []
    for i in range(3):
        subset = sorted(top24[i*8:(i+1)*8])
        sets.append(subset)
    return sets

def render_ticket(numbers, hot_set=None):
    """Return HTML for two rows x 4 columns ticket style. hot_set is a set of numbers to highlight."""
    hot_set = hot_set or set()
    def badge(n):
        if n in hot_set:
            return f"<span style='display:inline-block;margin:4px;padding:8px 12px;background:#ffecec;color:#b30000;font-weight:700;border-radius:6px'>{n}</span>"
        else:
            return f"<span style='display:inline-block;margin:4px;padding:8px 12px;background:#eef6ff;color:#0a3f6b;border-radius:6px'>{n}</span>"
    row1 = " ".join(badge(n) for n in numbers[:4])
    row2 = " ".join(badge(n) for n in numbers[4:8])
    html = f"<div style='padding:6px 0'>{row1}<br>{row2}</div>"
    return html

# ---------- Main flow ----------
st.markdown("**Status:** scraping full history from Lottolyzer (may take a few seconds).")
try:
    all_draws_raw = scrape_all_pages()
    if not all_draws_raw:
        st.error("No draws parsed. The source may have changed or blocked scraping.")
        st.stop()
except PermissionError as pe:
    st.error(str(pe))
    st.stop()
except Exception as e:
    st.error(f"Unexpected scraping error: {e}")
    st.stop()

# Convert to DataFrame
df = draws_to_dataframe(all_draws_raw)

# Build frequency & predictions
freq_df = build_frequency(df)
top3_sets = top3_sets_from_freq(freq_df)
hot_numbers = set(freq_df['Number'].tolist()[:8])

# Layout: Tabs
tab_pred, tab_past, tab_freq = st.tabs(["Predictions", "Past Results", "Frequency Graph"])

with tab_pred:
    st.header("Top 3 Predicted Sets (unchanged algorithm)")
    st.write(f"*Based on {len(df)} draws scraped*")
    for idx, s in enumerate(top3_sets, start=1):
        html = render_ticket(s, hot_set=hot_numbers)
        st.markdown(f"**Set {idx}**", unsafe_allow_html=False)
        st.markdown(html, unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("Quick Picks (3 randomized sets, same layout)")
    for i in range(3):
        q = sorted(random.sample(range(1, TOTAL_NUMBERS+1), NUMBERS_PER_DRAW))
        html = render_ticket(q, hot_set=hot_numbers)
        st.markdown(f"Quick {i+1}", unsafe_allow_html=False)
        st.markdown(html, unsafe_allow_html=True)

with tab_past:
    st.header("Past Draws (collapsed by default)")
    with st.expander("Show full past draws (click to expand)", expanded=False):
        # show date and eight numbers as columns (n1..n8)
        display_df = df.copy()
        # format date nicely
        if 'date' in display_df.columns:
            display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
        # reorder to date, n1..n8
        cols = ['date'] + [f"n{i}" for i in range(1, NUMBERS_PER_DRAW+1) if f"n{i}" in display_df.columns]
        st.dataframe(display_df[cols], use_container_width=True)
        # Also render a few rows in ticket style for quick visual check
        st.markdown("### Visual (first 25 draws):")
        for _, row in display_df.head(25).iterrows():
            d = row.get('date') if 'date' in row.index else ''
            nums = [int(row[f"n{i}"]) for i in range(1, NUMBERS_PER_DRAW+1)]
            html = f"<strong>{d}</strong><br>" + render_ticket(nums, hot_set=hot_numbers)
            st.markdown(html, unsafe_allow_html=True)

with tab_freq:
    st.header("Frequency Graph")
    st.write("Numbers sorted by historical frequency (higher = hotter)")
    # plot with streamlit native altair or plotly
    try:
        import plotly.express as px
        fig = px.bar(freq_df, x='Number', y='Frequency', title='Number Frequency', labels={'Frequency':'Count'})
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.dataframe(freq_df)

st.success("Done — full history scraped and analyzed.")
