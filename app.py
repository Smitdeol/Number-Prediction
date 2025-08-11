# app.py
import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
from collections import Counter
import random
import re
import time
from datetime import datetime

# --------- Config ---------
st.set_page_config(page_title="Magnum Life Predictor (Full History)", layout="wide")
st.title("🎯 Magnum Life — Full History Predictor")

LOTTOLYZER_PAGE_TEMPLATE = "https://en.lottolyzer.com/history/malaysia/magnum-life/page/{}/per-page/50/number-view"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/115.0.0.0 Safari/537.36"
    )
}
REQUEST_TIMEOUT = 15
PAUSE_BETWEEN_REQUESTS = 0.5
NUMBERS_PER_DRAW = 8
TOTAL_NUMBERS = 36
MAX_PAGES_SAFE = 200  # safety cap


# --------- Scraper & parsers ---------
def fetch(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        return r
    except Exception as e:
        raise RuntimeError(f"Request failed: {e}")


def detect_total_pages(soup):
    max_page = 1
    # Search links for /page/N/
    for a in soup.find_all("a", href=True):
        m = re.search(r"/page/(\d+)", a["href"])
        if m:
            try:
                p = int(m.group(1))
                if p > max_page:
                    max_page = p
            except:
                pass
    # Look for "Page X of Y" text
    txt = soup.get_text(" ", strip=True)
    m2 = re.search(r"Page\s*\d+\s*of\s*(\d+)", txt, re.I)
    if m2:
        try:
            p = int(m2.group(1))
            if p > max_page:
                max_page = p
        except:
            pass
    if max_page > MAX_PAGES_SAFE:
        max_page = MAX_PAGES_SAFE
    return max_page


def parse_draws_from_page(soup):
    """
    Parse draws from a page: try table rows first (date + img alts),
    fallback to grouping all img alts sequentially.
    Returns list of dicts: {"date": date_or_none, "numbers": [n1..n8]}
    """
    date_regex = re.compile(r"\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\w+\s+\d{1,2},\s*\d{4}")
    draws = []

    # Try parsing table rows
    for tr in soup.select("table tbody tr"):
        # attempt date from first cell
        tds = tr.find_all("td")
        date_text = None
        if tds:
            for td in tds[:2]:
                txt = td.get_text(" ", strip=True)
                m = date_regex.search(txt)
                if m:
                    date_text = m.group(0)
                    break
        # collect numbers from img alts within the row
        nums = []
        for img in tr.find_all("img", alt=True):
            alt = img.get("alt", "").strip()
            if alt.isdigit():
                nums.append(int(alt))
        # fallback: if not images, try to parse digit tokens from subsequent tds
        if not nums:
            # assume first column is date, next columns are numbers (text)
            if len(tds) >= NUMBERS_PER_DRAW + 1:
                for td in tds[1:1 + NUMBERS_PER_DRAW]:
                    txt = td.get_text(" ", strip=True)
                    # maybe contains number like '13'
                    m = re.search(r"\b(\d{1,2})\b", txt)
                    if m:
                        nums.append(int(m.group(1)))
        if nums and len(nums) >= NUMBERS_PER_DRAW:
            draws.append({"date": date_text, "numbers": nums[:NUMBERS_PER_DRAW]})

    # If no table rows parsed, try searching div/li blocks
    if not draws:
        for entry in soup.select("div, li"):
            imgs = entry.find_all("img", alt=True)
            nums = [int(img["alt"]) for img in imgs if img.get("alt", "").strip().isdigit()]
            if nums and len(nums) >= NUMBERS_PER_DRAW:
                txt = entry.get_text(" ", strip=True)
                m = date_regex.search(txt)
                date_text = m.group(0) if m else None
                draws.append({"date": date_text, "numbers": nums[:NUMBERS_PER_DRAW]})

    # Final fallback: group all img alts sequentially, no dates
    if not draws:
        imgs = soup.find_all("img", alt=True)
        nums = [int(img["alt"]) for img in imgs if img.get("alt", "").strip().isdigit()]
        for i in range(0, len(nums), NUMBERS_PER_DRAW):
            group = nums[i:i + NUMBERS_PER_DRAW]
            if len(group) == NUMBERS_PER_DRAW:
                draws.append({"date": None, "numbers": group})

    return draws


def scrape_all_history():
    # Fetch page 1 first
    url1 = LOTTOLYZER_PAGE_TEMPLATE.format(1)
    resp1 = fetch(url1)
    if resp1.status_code in (403, 429):
        raise PermissionError(f"Access blocked by remote host (HTTP {resp1.status_code}).")
    if resp1.status_code != 200:
        raise RuntimeError(f"Failed to fetch page 1 (HTTP {resp1.status_code})")
    soup1 = BeautifulSoup(resp1.text, "lxml")
    total_pages = detect_total_pages(soup1) or 1
    total_pages = min(total_pages, MAX_PAGES_SAFE)

    aggregated = []
    # parse page1
    aggregated.extend(parse_draws_from_page(soup1))

    # parse remaining pages
    for p in range(2, total_pages + 1):
        time.sleep(PAUSE_BETWEEN_REQUESTS)
        url = LOTTOLYZER_PAGE_TEMPLATE.format(p)
        resp = fetch(url)
        if resp.status_code in (403, 429):
            # remote blocked us; stop and return partial data
            st.warning(f"Scraping blocked at page {p} (HTTP {resp.status_code}). Returning partial data.")
            break
        if resp.status_code != 200:
            st.warning(f"Failed to fetch page {p} (HTTP {resp.status_code}). Stopping.")
            break
        soup = BeautifulSoup(resp.text, "lxml")
        page_draws = parse_draws_from_page(soup)
        if not page_draws:
            # continue but warn
            st.info(f"No draws parsed on page {p}; continuing.")
        aggregated.extend(page_draws)

    # deduplicate preserves order of first occurrence
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


# --------- Conversion helpers & analytics ---------
def draws_to_df(draws):
    rows = []
    for d in draws:
        row = {}
        dt = None
        if d.get("date"):
            try:
                dt = pd.to_datetime(d["date"], errors="coerce", dayfirst=False, infer_datetime_format=True)
            except:
                dt = None
        row["date"] = dt
        for i in range(NUMBERS_PER_DRAW):
            row[f"n{i+1}"] = d["numbers"][i]
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["date"] + [f"n{i}" for i in range(1, NUMBERS_PER_DRAW+1)])
    df = pd.DataFrame(rows)
    # sort by date if possible (NaT will go last)
    if "date" in df.columns:
        try:
            df = df.sort_values(by="date", ascending=False, na_position="last").reset_index(drop=True)
        except Exception:
            pass
    return df


def build_frequency(df):
    all_nums = []
    for i in range(1, NUMBERS_PER_DRAW + 1):
        col = f"n{i}"
        if col in df.columns:
            all_nums.extend(df[col].dropna().astype(int).tolist())
    cnt = Counter(all_nums)
    freq_list = [(n, cnt.get(n, 0)) for n in range(1, TOTAL_NUMBERS + 1)]
    freq_df = pd.DataFrame(freq_list, columns=["Number", "Frequency"]).sort_values("Frequency", ascending=False).reset_index(drop=True)
    return freq_df


def top3_sets_by_freq(freq_df):
    # take top 24 numbers (or fewer if not available) and split into 3 sets of 8
    top_nums = freq_df["Number"].tolist()
    sets = []
    used = set()
    # prepare pool of remaining numbers for filling gaps
    remaining_pool = [n for n in range(1, TOTAL_NUMBERS + 1)]
    for i in range(3):
        start = i * 8
        subset = top_nums[start:start + 8]
        # if subset too small, fill from top_nums (remaining) then random from remaining_pool
        if len(subset) < 8:
            # add from top_nums leftovers
            needed = 8 - len(subset)
            # pick from top_nums that are not already used and not in subset
            extras = [n for n in top_nums if n not in subset and n not in used]
            take = extras[:needed]
            subset.extend(take)
            needed -= len(take)
            if needed > 0:
                fill_pool = [n for n in remaining_pool if n not in subset and n not in used]
                if len(fill_pool) >= needed:
                    subset.extend(random.sample(fill_pool, needed))
                else:
                    subset.extend(fill_pool[:needed])
        used.update(subset)
        sets.append(sorted(subset))
    return sets


def render_ticket_html(numbers, hot_set=None):
    hot_set = hot_set or set()
    def badge(n):
        if n in hot_set:
            return f"<span style='display:inline-block;margin:4px;padding:8px 12px;background:#ffecec;color:#b30000;font-weight:700;border-radius:6px'>{n}</span>"
        else:
            return f"<span style='display:inline-block;margin:4px;padding:8px 12px;background:#eef6ff;color:#0a3f6b;border-radius:6px'>{n}</span>"
    row1 = " ".join(badge(n) for n in numbers[:4])
    row2 = " ".join(badge(n) for n in numbers[4:8])
    return f"{row1}<br>{row2}"


# --------- Main app flow ---------
with st.spinner("Scraping full history from Lottolyzer (this may take a few seconds)..."):
    try:
        raw_draws = scrape_all_history()
    except PermissionError as pe:
        st.error(str(pe))
        st.stop()
    except Exception as e:
        st.error(f"Unexpected scraping error: {e}")
        st.stop()

if not raw_draws:
    st.error("No draws parsed. The source may be temporarily unavailable or its structure changed.")
    st.stop()

df = draws_to_df(raw_draws)

# Build frequency and predictions
freq_df = build_frequency(df)
hot_numbers = set(freq_df["Number"].tolist()[:8])
top3 = top3_sets_by_freq(freq_df)

# UI layout: Tabs
tab1, tab2, tab3 = st.tabs(["Predictions", "Past Results", "Frequency Graph"])

with tab1:
    st.header("Top 3 Predicted Sets (based on full history)")
    st.write(f"*Based on {len(df)} draws scraped*")
    for idx, s in enumerate(top3, start=1):
        html = render_ticket_html(s, hot_set=hot_numbers)
        st.markdown(f"**Set {idx}**", unsafe_allow_html=False)
        st.markdown(html, unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("Quick Picks (3 randomized sets)")
    for i in range(3):
        qp = sorted(random.sample(range(1, TOTAL_NUMBERS + 1), NUMBERS_PER_DRAW))
        html = render_ticket_html(qp, hot_set=hot_numbers)
        st.markdown(f"Quick {i+1}", unsafe_allow_html=False)
        st.markdown(html, unsafe_allow_html=True)

with tab2:
    st.header("Past Results (collapsed)")
    with st.expander("Show full past results (click to expand)", expanded=False):
        display_df = df.copy()
        # safe date formatting
        if "date" in display_df.columns:
            safe_dates = pd.to_datetime(display_df["date"], errors="coerce")
            display_df["date"] = safe_dates.dt.strftime("%Y-%m-%d").fillna("")
        # ensure numeric columns exist and are properly typed (keep as integers where possible)
        for i in range(1, NUMBERS_PER_DRAW + 1):
            col = f"n{i}"
            if col in display_df.columns:
                display_df[col] = pd.to_numeric(display_df[col], errors="coerce").astype("Int64")
            else:
                display_df[col] = pd.NA
        cols = ["date"] + [f"n{i}" for i in range(1, NUMBERS_PER_DRAW + 1)]
        st.dataframe(display_df[cols], use_container_width=True)

with tab3:
    st.header("Frequency Graph")
    try:
        import plotly.express as px
        fig = px.bar(freq_df, x="Number", y="Frequency", title="Number Frequency (historical)", labels={"Frequency": "Count", "Number": "Number"})
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.dataframe(freq_df)

st.success("Analysis complete.")
```0
