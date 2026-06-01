# ============================================================
#  Magnum Life AI Predictor — Streamlit App
#  FREE VERSION — Google Gemini 2.0 Flash
#
#  Scraper handles lottolyzer card-based layout:
#  Each draw = a card block with draw number, date, balls
#  Numbers in colored circle elements (span/div)
# ============================================================

import streamlit as st
import requests
import json, re, time
import pandas as pd
from bs4 import BeautifulSoup
from collections import Counter

st.set_page_config(page_title="Magnum Life AI Predictor", page_icon="🎱", layout="centered")

# ── Constants ─────────────────────────────────────────────────
# From screenshot: URL is /history not /number-view
# Using per-page/10 to be safe, then loop more pages
HISTORY_URL = "https://en.lottolyzer.com/history/malaysia/magnum-life/page/{page}/per-page/50/number-view"
LAST_PAGE   = 26   # screenshot shows 26 pages
GEMINI_URL  = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={key}"
HEADERS     = {
    "User-Agent": "Mozilla/5.0 (Linux; Android 13; SM-G991B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Mobile Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Referer": "https://en.lottolyzer.com/",
    "Upgrade-Insecure-Requests": "1",
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
.debug-box { background:#0a0a1a; border:1px solid #2a2a4a; border-radius:8px; padding:12px; font-size:0.7rem; color:#666; margin:8px 0; }
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
        raise ValueError("No JSON found")
    return json.loads(s[a:b+1])

# ── Scraper — handles lottolyzer card layout ──────────────────
def parse_page(html: str) -> list:
    """
    Lottolyzer layout from screenshot:

    History Summary Table — card view
    Each draw card contains:
      - Header: "Draw 375/26  31 May 2026"
      - Row of 8 colored ball spans/divs
      - Second row: 2 more numbers (Life Ball numbers)

    Numbers are inside elements like:
      <span class="...ball...">8</span>
      or <div class="num">8</div>
      or <td>8</td> in number-view mode
    """
    soup  = BeautifulSoup(html, "html.parser")
    rows  = []

    # ── Strategy 1: number-view table format ──────────────────
    # Each <tr> = one draw, cells contain individual numbers
    tables = soup.find_all("table")
    for table in tables:
        for tr in table.find_all("tr"):
            tds = tr.find_all("td")
            if len(tds) < 5:
                continue
            date_text = tds[0].get_text(strip=True)
            nums = []
            for td in tds[1:]:
                for token in td.get_text(strip=True).split():
                    if token.isdigit() and 1 <= int(token) <= 36:
                        nums.append(int(token))
            if len(nums) >= 9:
                rows.append({
                    "date":     date_text,
                    "numbers":  sorted(nums[:8]),
                    "lifeBall": nums[8],
                })

    if rows:
        return rows

    # ── Strategy 2: Card/block layout (what screenshot shows) ──
    # Find draw header blocks then extract numbers near them
    # Draw headers look like: "Draw 375/26" with a date
    draw_headers = soup.find_all(
        string=re.compile(r'Draw\s+\d+/\d+', re.I)
    )

    for header in draw_headers:
        parent = header.parent
        # Walk up to find the card container
        card = parent
        for _ in range(5):
            if card.parent:
                card = card.parent
            else:
                break

        # Extract date from header text
        date_match = re.search(
            r'(\d{1,2}\s+\w+\s+\d{4}|\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4})',
            card.get_text()
        )
        date_text = date_match.group(1) if date_match else ""

        # Extract all numbers 1-36 from this card
        all_text = card.get_text(separator=" ")
        tokens   = re.findall(r'\b(\d{1,2})\b', all_text)
        nums     = []
        seen     = set()
        for t in tokens:
            n = int(t)
            if 1 <= n <= 36 and n not in seen:
                nums.append(n)
                seen.add(n)

        if len(nums) >= 9:
            rows.append({
                "date":     date_text,
                "numbers":  sorted(nums[:8]),
                "lifeBall": nums[8],
            })

    if rows:
        return rows

    # ── Strategy 3: Find any elements with ball/number classes ─
    # Handles any CSS class names used for lottery balls
    ball_els = soup.find_all(
        class_=re.compile(r'ball|num|number|lotto|draw', re.I)
    )

    # Group by proximity — collect runs of 9+ numbers
    current_group = []
    current_date  = ""
    for el in ball_els:
        txt = el.get_text(strip=True)
        if re.match(r'\d{1,2}$', txt):
            n = int(txt)
            if 1 <= n <= 36:
                current_group.append(n)
        # Date-like element resets the group
        elif re.search(r'\d{4}', txt) and len(txt) > 6:
            if len(current_group) >= 9:
                rows.append({
                    "date":     current_date,
                    "numbers":  sorted(current_group[:8]),
                    "lifeBall": current_group[8],
                })
            current_group = []
            current_date  = txt

        if len(current_group) >= 10:
            rows.append({
                "date":     current_date,
                "numbers":  sorted(current_group[:8]),
                "lifeBall": current_group[8],
            })
            current_group = []

    return rows


def scrape_all(last_page=LAST_PAGE, progress_bar=None) -> tuple:
    """Returns (DataFrame, debug_info)"""
    all_rows   = []
    debug_info = []

    for page in range(1, last_page + 1):
        url  = HISTORY_URL.format(page=page)
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            status = resp.status_code
            html   = resp.text if resp.ok else ""

            page_rows = parse_page(html) if html else []
            all_rows.extend(page_rows)

            debug_info.append(f"Page {page}: HTTP {status}, "
                              f"HTML {len(html)} bytes, "
                              f"{len(page_rows)} draws parsed")
        except Exception as e:
            debug_info.append(f"Page {page}: ERROR — {e}")

        if progress_bar:
            progress_bar.progress(
                page / last_page,
                text=f"Scraping page {page}/{last_page} — {len(all_rows)} draws so far"
            )
        time.sleep(0.4)

    if not all_rows:
        return pd.DataFrame(), debug_info

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date", ascending=False)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df = df.drop_duplicates(subset=["date"])
    return df.reset_index(drop=True), debug_info


# ── Local statistics ──────────────────────────────────────────
def compute_stats(df: pd.DataFrame) -> dict:
    all_nums = [n for nums in df["numbers"] for n in nums]
    all_lbs  = list(df["lifeBall"])
    freq     = Counter(all_nums)
    lb_freq  = Counter(all_lbs)

    hot  = [n for n, _ in freq.most_common(8)]
    cold = [n for n, _ in freq.most_common()[:-9:-1]]

    all_number_lists = list(df["numbers"])
    gaps = {}
    for n in range(1, 37):
        gap = next((i for i, nums in enumerate(all_number_lists) if n in nums), len(df))
        gaps[n] = gap
    overdue = sorted(gaps, key=gaps.get, reverse=True)[:8]

    odd_counts  = [sum(1 for n in nums if n % 2 != 0) for nums in df["numbers"]]
    even_counts = [sum(1 for n in nums if n % 2 == 0) for nums in df["numbers"]]

    return {
        "hotNumbers":     hot,
        "coldNumbers":    cold,
        "overdueNumbers": overdue,
        "hotLifeBalls":   [n for n, _ in lb_freq.most_common(3)],
        "totalDraws":     len(df),
        "avgOdd":         round(sum(odd_counts)  / len(odd_counts),  1),
        "avgEven":        round(sum(even_counts) / len(even_counts), 1),
        "numberFrequency": dict(freq),
        "gaps":           gaps,
    }


# ── Gemini AI ─────────────────────────────────────────────────
def call_gemini(prompt, api_key):
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


def ai_predict(stats, api_key):
    prompt = f"""You are a Magnum Life lottery prediction expert for Malaysia.
Magnum Life: 8 numbers from 1–36 + 1 Life Ball from 1–36. Weekly Wednesday draw.

REAL DATA from {stats['totalDraws']} actual draws:
Hot numbers: {stats['hotNumbers']}
Cold numbers: {stats['coldNumbers']}
Overdue numbers: {stats['overdueNumbers']}
Hot Life Balls: {stats['hotLifeBalls']}
Avg odd/even: {stats['avgOdd']} / {stats['avgEven']}
Full frequency: {stats['numberFrequency']}
Draws since last seen (gap): {stats['gaps']}

Generate 3 prediction sets using the real stats above.
Return ONLY raw JSON no markdown:
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


# ── UI ────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:24px 0 10px">
  <div style="color:#FFD700;font-size:.7rem;letter-spacing:4px;text-transform:uppercase;margin-bottom:6px">Malaysia</div>
  <h1 style="margin:0;font-size:2.6rem;font-weight:900;letter-spacing:2px">MAGNUM <span style="color:#FFD700">LIFE</span></h1>
  <div style="color:#555;font-size:.65rem;letter-spacing:3px;text-transform:uppercase;margin-top:6px">AI Number Predictor</div>
  <div style="color:#333;font-size:.6rem;margin-top:4px">Real data · lottolyzer.com · Google Gemini Free</div>
</div>
<hr style="border-color:#1E1E30">
""", unsafe_allow_html=True)

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

        st.write("🌐 Scraping all pages from Lottolyzer.com...")
        pbar = st.progress(0, text="Starting...")
        df, debug = scrape_all(last_page=LAST_PAGE, progress_bar=pbar)
        pbar.empty()

        # Always show debug so we can see what happened
        with st.expander("🔍 Scrape debug log (click to expand)"):
            for line in debug[:10]:
                st.markdown(f'<div class="debug-box">{line}</div>', unsafe_allow_html=True)

        if df.empty:
            st.error("❌ Scraper could not parse draw data. See debug log above for details.")
            st.info("The debug log shows HTTP status codes and HTML sizes — share this with the developer to fix the parser.")
            st.stop()

        st.write(f"✅ {len(df)} real draws loaded from {LAST_PAGE} pages")

        st.write("🔍 Computing frequency, gaps & patterns from real data...")
        stats = compute_stats(df)
        st.write(f"✅ {stats['totalDraws']} draws fully analysed")

        st.write("🧠 AI generating predictions using real statistics...")
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

        # Stats
        st.markdown('<div class="sec-label">📊 Real Data Overview</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Draws Analysed", stats["totalDraws"])
        c2.metric("Avg Odd / Even",  f"{stats['avgOdd']} / {stats['avgEven']}")
        c3.metric("Next Draw",       next_draw)

        st.markdown('<div class="sec-label">🔥 Hot Numbers</div>', unsafe_allow_html=True)
        st.markdown(balls(stats["hotNumbers"], "hot"), unsafe_allow_html=True)
        st.markdown('<div class="sec-label">❄️ Cold Numbers</div>', unsafe_allow_html=True)
        st.markdown(balls(stats["coldNumbers"], "cold"), unsafe_allow_html=True)
        st.markdown('<div class="sec-label">⏰ Most Overdue</div>', unsafe_allow_html=True)
        st.markdown(balls(stats["overdueNumbers"], "normal"), unsafe_allow_html=True)
        st.markdown('<div class="sec-label">🔴 Hot Life Balls</div>', unsafe_allow_html=True)
        st.markdown(balls(stats["hotLifeBalls"], "red"), unsafe_allow_html=True)

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
            st.code(f"Numbers: {', '.join(map(str,p.get('numbers',[])))}  |  Life Ball: {p.get('lifeBall','?')}", language=None)

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
