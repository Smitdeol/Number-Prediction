# ============================================================
#  Magnum Life AI Predictor — Streamlit App
#  FREE VERSION — Google Gemini 2.0 Flash
#
#  Scraper targets lottolyzer.com number-view page
#  which renders numbers as individual <td> cells in a table
# ============================================================

import streamlit as st
import requests
import json, re, time
import pandas as pd
from bs4 import BeautifulSoup
from collections import Counter

st.set_page_config(page_title="Magnum Life AI Predictor", page_icon="🎱", layout="centered")

# ── Constants ─────────────────────────────────────────────────
BASE_URL    = "https://en.lottolyzer.com/history/malaysia/magnum-life/page/{page}/per-page/50/number-view"
LAST_PAGE   = 24   # 24 pages × 50 draws = ~1,200 draws total
GEMINI_URL  = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={key}"
HEADERS     = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Referer": "https://en.lottolyzer.com/",
}

# ── Styling ───────────────────────────────────────────────────
st.markdown("""
<style>
body, .stApp { background-color: #0C0C14; color: #ffffff; }
.ball { display:inline-flex; align-items:center; justify-content:center;
        width:40px; height:40px; border-radius:50%;
        font-weight:900; font-size:0.95rem; margin:3px; }
.b-normal { background:#1E1E30; color:#fff; }
.b-gold   { background:linear-gradient(135deg,#b8860b,#FFD700); color:#000; box-shadow:0 3px 10px rgba(255,215,0,.3); }
.b-red    { background:linear-gradient(135deg,#FF3B3B,#ff6b6b); color:#fff; box-shadow:0 3px 10px rgba(255,59,59,.4); }
.b-hot    { background:#FF6B3520; border:1.5px solid #FF6B35; color:#FF6B35; }
.b-cold   { background:#4FC3F720; border:1.5px solid #4FC3F7; color:#4FC3F7; }
.pred-best { background:linear-gradient(135deg,#1a1500,#13131F); border:1.5px solid #FFD70055; border-radius:14px; padding:16px; margin-bottom:12px; }
.pred-card { background:#13131F; border:1.5px solid #1E1E30; border-radius:14px; padding:16px; margin-bottom:12px; }
.tag-h { background:#1a3a1a; color:#4CAF50; padding:2px 8px; border-radius:5px; font-size:0.65rem; font-weight:700; }
.tag-m { background:#2a2010; color:#FFA726; padding:2px 8px; border-radius:5px; font-size:0.65rem; font-weight:700; }
.draw-row { background:#13131F; border:1px solid #1E1E30; border-radius:8px; padding:8px 12px; margin-bottom:5px; }
.sec-label { color:#444; font-size:0.62rem; letter-spacing:3px; text-transform:uppercase; margin:16px 0 8px; }
.disclaimer { background:#090909; border:1px solid #181818; border-radius:8px; padding:12px; color:#333; font-size:0.65rem; text-align:center; margin-top:20px; }
.stat-box { background:#13131F; border:1px solid #1E1E30; border-radius:10px; padding:14px; text-align:center; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────
def balls(numbers, style="normal"):
    c = {"normal":"b-normal","gold":"b-gold","red":"b-red","hot":"b-hot","cold":"b-cold"}.get(style,"b-normal")
    return "".join(f'<span class="ball {c}">{n}</span>' for n in numbers)

def extract_json(raw):
    s = re.sub(r'```json|```', '', raw, flags=re.IGNORECASE).strip()
    a, b = s.find('{'), s.rfind('}')
    if a == -1 or b == -1:
        raise ValueError("No JSON found in AI response")
    return json.loads(s[a:b+1])

# ── Scraper — lottolyzer number-view layout ───────────────────
def scrape_page(page: int) -> list:
    """
    Lottolyzer number-view layout:
    Each draw = one <tr> row
    First <td> = draw number or date
    Remaining <td> cells = individual numbers (1-36)
    Last number in each row = Life Ball
    """
    rows   = []
    url    = BASE_URL.format(page=page)
    resp   = requests.get(url, headers=HEADERS, timeout=15)

    if resp.status_code != 200:
        return rows

    soup = BeautifulSoup(resp.text, "html.parser")

    # Find the results table — lottolyzer uses class 'table' or similar
    table = (
        soup.find("table", {"class": re.compile(r"table|result|history", re.I)})
        or soup.find("table")
    )
    if not table:
        return rows

    for tr in table.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 5:
            continue

        # Extract date — first cell
        date_text = tds[0].get_text(strip=True)

        # Extract all numbers from remaining cells
        nums = []
        for td in tds[1:]:
            txt = td.get_text(strip=True)
            # Handle cells that may contain multiple numbers separated by space
            for token in txt.split():
                if token.isdigit():
                    n = int(token)
                    if 1 <= n <= 36:
                        nums.append(n)

        # Need at least 9 numbers (8 main + 1 life ball)
        if len(nums) >= 9:
            rows.append({
                "date":     date_text,
                "numbers":  sorted(nums[:8]),
                "lifeBall": nums[8],
            })

    return rows

def scrape_all(last_page: int = LAST_PAGE, progress_bar=None) -> pd.DataFrame:
    all_rows = []
    for page in range(1, last_page + 1):
        page_rows = scrape_page(page)
        all_rows.extend(page_rows)
        if progress_bar:
            progress_bar.progress(page / last_page, text=f"Scraping page {page}/{last_page} — {len(all_rows)} draws found")
        time.sleep(0.4)   # polite delay between requests

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date", ascending=False)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df = df.drop_duplicates(subset=["date"])
    return df.reset_index(drop=True)

# ── Local statistics (pure Python — no AI needed) ─────────────
def compute_stats(df: pd.DataFrame) -> dict:
    all_nums   = [n for nums in df["numbers"] for n in nums]
    all_lbs    = list(df["lifeBall"])
    freq       = Counter(all_nums)
    lb_freq    = Counter(all_lbs)

    hot        = [n for n, _ in freq.most_common(8)]
    cold_all   = [n for n, _ in freq.most_common()]
    cold       = [n for n, _ in freq.most_common()[:-9:-1]]  # bottom 8

    # Gap analysis — draws since each number last appeared
    all_numbers_flat = list(df["numbers"])
    gaps = {}
    for n in range(1, 37):
        gap = next((i for i, nums in enumerate(all_numbers_flat) if n in nums), len(df))
        gaps[n] = gap
    most_overdue = sorted(gaps, key=gaps.get, reverse=True)[:8]

    # Odd / even average
    odd_counts  = [sum(1 for n in nums if n % 2 != 0) for nums in df["numbers"]]
    even_counts = [sum(1 for n in nums if n % 2 == 0) for nums in df["numbers"]]

    return {
        "hotNumbers":        hot,
        "coldNumbers":       cold,
        "overdueNumbers":    most_overdue,
        "hotLifeBalls":      [n for n, _ in lb_freq.most_common(3)],
        "totalDraws":        len(df),
        "avgOdd":            round(sum(odd_counts)  / len(odd_counts),  1),
        "avgEven":           round(sum(even_counts) / len(even_counts), 1),
        "numberFrequency":   dict(freq),
        "gaps":              gaps,
    }

# ── Gemini AI — predict using real stats ──────────────────────
def call_gemini(prompt: str, api_key: str, retry: bool = True):
    url  = GEMINI_URL.format(key=api_key)
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 4000},
    }
    r = requests.post(url, json=body, timeout=90)
    if r.status_code == 429:
        return None, "rate_limit"
    if r.status_code == 401:
        raise ValueError("Invalid Gemini API key")
    if not r.ok:
        raise ValueError(f"Gemini error {r.status_code}: {r.text[:150]}")
    return r.json()["candidates"][0]["content"]["parts"][0]["text"], "ok"

def ai_predict(stats: dict, api_key: str) -> dict:
    prompt = f"""You are a Magnum Life lottery prediction expert for Malaysia.

Magnum Life rules: 8 main numbers from 1–36, 1 Life Ball from 1–36. Weekly Wednesday draw.

REAL HISTORICAL STATISTICS (scraped from {stats['totalDraws']} actual draws):

Hot numbers (most frequent): {stats['hotNumbers']}
Cold numbers (least frequent): {stats['coldNumbers']}
Most overdue numbers (draws since last appeared): {stats['overdueNumbers']}
Hot Life Balls: {stats['hotLifeBalls']}
Average odd per draw: {stats['avgOdd']}
Average even per draw: {stats['avgEven']}
Full frequency table: {stats['numberFrequency']}
Gap table (draws since last seen): {stats['gaps']}

Using ALL the above real data, apply these strategies to generate 3 prediction sets:
1. Hot strategy — favour most frequent numbers
2. Balanced — mix hot + cold + correct odd/even ratio
3. Due strategy — favour most overdue numbers

Return ONLY raw JSON, no markdown:
{{
  "predictions":[
    {{"strategy":"🔥 Hot Numbers","numbers":[8 sorted unique ints 1-36],"lifeBall":N,"confidence":"High","reason":"one line"}},
    {{"strategy":"⚖️ Balanced Mix","numbers":[8 sorted unique ints 1-36],"lifeBall":N,"confidence":"High","reason":"one line"}},
    {{"strategy":"❄️ Due Numbers","numbers":[8 sorted unique ints 1-36],"lifeBall":N,"confidence":"Medium","reason":"one line"}}
  ],
  "nextDraw":"YYYY-MM-DD"
}}"""

    raw, status = call_gemini(prompt, api_key)
    if status == "rate_limit":
        return None, "rate_limit"
    return extract_json(raw), "ok"

# ── UI ─────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:24px 0 10px">
  <div style="color:#FFD700;font-size:.7rem;letter-spacing:4px;text-transform:uppercase;margin-bottom:6px">Malaysia</div>
  <h1 style="margin:0;font-size:2.6rem;font-weight:900;letter-spacing:2px">MAGNUM <span style="color:#FFD700">LIFE</span></h1>
  <div style="color:#555;font-size:.65rem;letter-spacing:3px;text-transform:uppercase;margin-top:6px">AI Number Predictor</div>
  <div style="color:#333;font-size:.6rem;margin-top:4px">Real data from lottolyzer.com · Google Gemini AI (Free)</div>
</div>
<hr style="border-color:#1E1E30">
""", unsafe_allow_html=True)

# ── API Key check ─────────────────────────────────────────────
api_key = st.secrets.get("GEMINI_API_KEY", "")
if not api_key:
    st.markdown("""
<div style="background:#1a1a00;border:1px solid #FFD70044;border-radius:12px;padding:20px;margin:10px 0">
  <div style="color:#FFD700;font-weight:700;margin-bottom:10px">⚙️ One-time Free Setup</div>
  <div style="color:#aaa;font-size:0.82rem;line-height:2">
    1️⃣ Go to <b style="color:#4FC3F7">aistudio.google.com/apikey</b><br>
    2️⃣ Sign in with Google (100% free)<br>
    3️⃣ Click <b>Create API Key</b> → copy it<br>
    4️⃣ Streamlit → <b>Manage App → Settings → Secrets</b><br>
    5️⃣ Add: <code>GEMINI_API_KEY = "AIzaSy..."</code><br>
    6️⃣ Save ✅
  </div>
</div>""", unsafe_allow_html=True)
    st.stop()

st.markdown('<div style="background:#0d1f0d;border:1px solid #4CAF5033;border-radius:8px;padding:8px 14px;font-size:0.72rem;color:#4CAF50;margin-bottom:16px;text-align:center">✅ Free Tier Active — Google Gemini 2.0 Flash</div>', unsafe_allow_html=True)

if st.button("⚡  FETCH & PREDICT", use_container_width=True, type="primary"):
    with st.status("Running...", expanded=True) as status_ui:

        # ── Step 1: Scrape real data ──────────────────────────
        st.write("🌐 Scraping all pages from Lottolyzer.com...")
        pbar = st.progress(0, text="Starting...")
        df   = scrape_all(last_page=LAST_PAGE, progress_bar=pbar)
        pbar.empty()

        if df.empty:
            st.error("❌ Could not scrape Lottolyzer. The site may be temporarily down. Try again in a few minutes.")
            st.stop()

        st.write(f"✅ {len(df)} real draws scraped from {LAST_PAGE} pages")

        # ── Step 2: Compute stats locally (no AI needed) ──────
        st.write("🔍 Computing frequency, gaps & patterns...")
        stats = compute_stats(df)
        st.write(f"✅ Stats computed — {stats['totalDraws']} draws analysed")

        # ── Step 3: AI predictions using real stats ───────────
        st.write("🧠 AI generating predictions from real data...")
        result, ai_status = ai_predict(stats, api_key)

        if ai_status == "rate_limit":
            st.write("⏳ Rate limit — auto-retrying in 65 seconds...")
            prog = st.progress(0)
            for i in range(65):
                time.sleep(1)
                prog.progress((i+1)/65, text=f"Waiting {64-i}s...")
            prog.empty()
            result, _ = ai_predict(stats, api_key)

        if not result:
            st.error("❌ AI prediction failed. Try again in 1 minute.")
            st.stop()

        preds     = result.get("predictions", [])
        next_draw = result.get("nextDraw", "Next Wednesday")
        status_ui.update(label="✅ Done!", state="complete")

        # ── Stats overview ────────────────────────────────────
        st.markdown('<div class="sec-label">📊 Real Data Overview</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Draws Analysed", stats["totalDraws"])
        c2.metric("Avg Odd / Even",  f"{stats['avgOdd']} / {stats['avgEven']}")
        c3.metric("Next Draw",       next_draw)

        # ── Hot, Cold, Overdue ────────────────────────────────
        st.markdown('<div class="sec-label">🔥 Hot Numbers</div>', unsafe_allow_html=True)
        st.markdown(balls(stats["hotNumbers"], "hot"), unsafe_allow_html=True)

        st.markdown('<div class="sec-label">❄️ Cold Numbers</div>', unsafe_allow_html=True)
        st.markdown(balls(stats["coldNumbers"], "cold"), unsafe_allow_html=True)

        st.markdown('<div class="sec-label">⏰ Most Overdue Numbers</div>', unsafe_allow_html=True)
        st.markdown(balls(stats["overdueNumbers"], "normal"), unsafe_allow_html=True)

        st.markdown('<div class="sec-label">🔴 Hot Life Balls</div>', unsafe_allow_html=True)
        st.markdown(balls(stats["hotLifeBalls"], "red"), unsafe_allow_html=True)

        # ── Predictions ───────────────────────────────────────
        st.markdown('<div class="sec-label">🎯 AI Predictions — Based on Real Data</div>', unsafe_allow_html=True)
        for i, p in enumerate(preds):
            card = "pred-best" if i == 1 else "pred-card"
            ctag = "tag-h" if p.get("confidence") == "High" else "tag-m"
            best = '<span style="background:#FFD700;color:#000;font-size:.55rem;font-weight:900;padding:2px 7px;border-radius:4px;margin-left:6px">BEST BET</span>' if i == 1 else ""
            tc   = "#FFD700" if i == 1 else "#cccccc"
            mb   = balls(p.get("numbers",[]), "gold" if i == 1 else "normal")
            lb   = balls([p.get("lifeBall","?")], "red")

            st.markdown(f"""
<div class="{card}">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
    <span style="font-weight:700;font-size:.95rem;color:{tc}">{p.get('strategy','')} {best}</span>
    <span class="{ctag}">{p.get('confidence','')} Confidence</span>
  </div>
  <div style="margin-bottom:10px">{mb}</div>
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
    <span style="color:#444;font-size:.6rem;letter-spacing:1px;text-transform:uppercase">Life Ball</span>
    {lb}
    <span style="color:#333;font-size:.72rem;font-style:italic">{p.get('reason','')}</span>
  </div>
</div>""", unsafe_allow_html=True)
            st.code(f"Numbers: {', '.join(map(str, p.get('numbers',[])))}  |  Life Ball: {p.get('lifeBall','?')}", language=None)

        # ── Recent draws ──────────────────────────────────────
        st.markdown('<div class="sec-label">📅 Recent Draw History</div>', unsafe_allow_html=True)
        for _, row in df.head(10).iterrows():
            nb = balls(row["numbers"], "normal")
            lb = balls([row["lifeBall"]], "red")
            st.markdown(f"""
<div class="draw-row">
  <span style="color:#444;font-size:.68rem;margin-right:8px">{row['date']}</span>
  {nb} {lb}
</div>""", unsafe_allow_html=True)

        st.markdown('<div class="disclaimer">⚠️ For entertainment only. Lottery results are random. Play responsibly.</div>', unsafe_allow_html=True)
