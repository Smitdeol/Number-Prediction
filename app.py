# ============================================================
#  Magnum Life AI Predictor — Streamlit App
#  FREE VERSION — Google Gemini 2.0 Flash
#
#  Flow: Scrape ALL data → Stats locally → AI predicts once
# ============================================================

import streamlit as st
import requests, json, re, time
import pandas as pd
from bs4 import BeautifulSoup, Tag
from collections import Counter

st.set_page_config(page_title="Magnum Life AI Predictor", page_icon="🎱", layout="centered")

LAST_PAGE  = 26
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={key}"

# Lottolyzer URLs — try multiple formats
URLS = [
    "https://en.lottolyzer.com/history/malaysia/magnum-life/page/{p}/per-page/50/number-view",
    "https://en.lottolyzer.com/history/malaysia/magnum-life/page/{p}/per-page/50",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Linux; Android 13; SM-G991B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Mobile Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://en.lottolyzer.com/",
    "Connection": "keep-alive",
}

# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
body,.stApp{background:#0C0C14;color:#fff}
.ball{display:inline-flex;align-items:center;justify-content:center;
      width:40px;height:40px;border-radius:50%;font-weight:900;font-size:.95rem;margin:3px}
.bn{background:#1E1E30;color:#fff}
.bg{background:linear-gradient(135deg,#b8860b,#FFD700);color:#000;box-shadow:0 3px 10px rgba(255,215,0,.3)}
.br{background:linear-gradient(135deg,#c0392b,#e74c3c);color:#fff;box-shadow:0 3px 10px rgba(255,59,59,.4)}
.bh{background:#FF6B3520;border:1.5px solid #FF6B35;color:#FF6B35}
.bc{background:#4FC3F720;border:1.5px solid #4FC3F7;color:#4FC3F7}
.bo{background:#9B59B620;border:1.5px solid #9B59B6;color:#9B59B6}
.pb{background:linear-gradient(135deg,#1a1500,#13131F);border:1.5px solid #FFD70055;border-radius:14px;padding:16px;margin-bottom:12px}
.pc{background:#13131F;border:1.5px solid #1E1E30;border-radius:14px;padding:16px;margin-bottom:12px}
.th{background:#1a3a1a;color:#4CAF50;padding:2px 8px;border-radius:5px;font-size:.65rem;font-weight:700}
.tm{background:#2a2010;color:#FFA726;padding:2px 8px;border-radius:5px;font-size:.65rem;font-weight:700}
.dr{background:#13131F;border:1px solid #1E1E30;border-radius:8px;padding:8px 12px;margin-bottom:5px}
.sl{color:#444;font-size:.62rem;letter-spacing:3px;text-transform:uppercase;margin:16px 0 8px}
.disc{background:#090909;border:1px solid #181818;border-radius:8px;padding:12px;color:#2a2a2a;font-size:.65rem;text-align:center;margin-top:20px}
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────
def B(nums, s="n"):
    m={"n":"bn","g":"bg","r":"br","h":"bh","c":"bc","o":"bo"}
    c=m.get(s,"bn")
    return "".join(f'<span class="ball {c}">{n}</span>' for n in nums)

def xj(raw):
    s=re.sub(r'```json|```','',raw,flags=re.I).strip()
    a,b=s.find('{'),s.rfind('}')
    if a<0 or b<0: raise ValueError("No JSON")
    return json.loads(s[a:b+1])

# ── SCRAPER ───────────────────────────────────────────────────
def extract_draws_from_html(html: str) -> list:
    """
    Handles lottolyzer layout from screenshot:
    Cards showing "Draw 375/26  31 May 2026"
    with 8 colored balls + 2 life ball numbers below
    """
    soup  = BeautifulSoup(html, "html.parser")
    draws = []

    # ── Method A: number-view table ───────────────────────────
    for table in soup.find_all("table"):
        for tr in table.find_all("tr"):
            cells = tr.find_all("td")
            if len(cells) < 5:
                continue
            date_raw = cells[0].get_text(strip=True)
            nums = []
            for td in cells[1:]:
                t = td.get_text(strip=True)
                if t.isdigit():
                    n = int(t)
                    if 1 <= n <= 36:
                        nums.append(n)
            if len(nums) >= 9:
                draws.append({
                    "date": date_raw,
                    "numbers": sorted(nums[:8]),
                    "lifeBall": nums[8]
                })

    if draws:
        return draws

    # ── Method B: Card layout (what screenshot shows) ─────────
    # Find all elements containing "Draw" + number pattern
    draw_pattern = re.compile(r'Draw\s+\d+/\d+', re.I)

    # Get all text nodes or elements with Draw header
    for el in soup.find_all(string=draw_pattern):
        # Get the card container (walk up 3-6 levels)
        container = el.parent
        for _ in range(6):
            if not container or not hasattr(container, 'parent'):
                break
            text = container.get_text(" ", strip=True)
            # Check if container has enough numbers
            found = re.findall(r'\b([1-9]|[12][0-9]|3[0-6])\b', text)
            if len(found) >= 9:
                break
            container = container.parent

        if not container:
            continue

        card_text = container.get_text(" ", strip=True)

        # Extract date
        date_m = re.search(
            r'(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{4})',
            card_text, re.I
        )
        date_raw = date_m.group(1) if date_m else ""

        # Extract numbers — avoid draw number like 375/26
        # Remove the draw header part first
        clean = re.sub(r'Draw\s+\d+/\d+', '', card_text)
        clean = re.sub(r'\d{4}', '', clean)  # remove year
        tokens = re.findall(r'\b(\d{1,2})\b', clean)

        nums = []
        seen = set()
        for t in tokens:
            n = int(t)
            if 1 <= n <= 36 and n not in seen:
                nums.append(n)
                seen.add(n)

        if len(nums) >= 9:
            draws.append({
                "date": date_raw,
                "numbers": sorted(nums[:8]),
                "lifeBall": nums[8]
            })

    if draws:
        return draws

    # ── Method C: Find numbered ball elements by CSS class ─────
    # Try common class names lottolyzer might use
    for cls in ['ball', 'num', 'number', 'lotto-ball', 'draw-number',
                'winning', 'result', 'n-', 'no-']:
        els = soup.find_all(class_=re.compile(cls, re.I))
        if len(els) >= 9:
            nums_run, date_run = [], ""
            for el in els:
                t = el.get_text(strip=True)
                if re.match(r'^\d{1,2}$', t):
                    n = int(t)
                    if 1 <= n <= 36:
                        nums_run.append(n)
                        if len(nums_run) == 10:
                            draws.append({
                                "date": date_run,
                                "numbers": sorted(nums_run[:8]),
                                "lifeBall": nums_run[8]
                            })
                            nums_run = []
            break

    return draws


def scrape_all(last_page=LAST_PAGE, pbar=None):
    all_draws  = []
    debug_log  = []
    url_format = URLS[0]   # start with number-view

    for page in range(1, last_page + 1):
        url  = url_format.format(p=page)
        success = False

        for attempt_url in [url_format.format(p=page), URLS[1].format(p=page)]:
            try:
                r = requests.get(attempt_url, headers=HEADERS, timeout=15)
                if r.status_code == 200 and len(r.text) > 500:
                    page_draws = extract_draws_from_html(r.text)
                    all_draws.extend(page_draws)
                    debug_log.append(
                        f"p{page}: HTTP {r.status_code}, "
                        f"{len(r.text)}B, {len(page_draws)} draws"
                    )
                    success = True
                    break
                else:
                    debug_log.append(f"p{page}: HTTP {r.status_code} ({attempt_url[-20:]})")
            except Exception as e:
                debug_log.append(f"p{page}: ERR {str(e)[:40]}")

        if pbar:
            pbar.progress(page/last_page,
                text=f"Page {page}/{last_page} — {len(all_draws)} draws collected")
        time.sleep(0.35)

    if not all_draws:
        return pd.DataFrame(), debug_log

    df = pd.DataFrame(all_draws)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date", ascending=False)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df = df.drop_duplicates(subset=["date"])
    return df.reset_index(drop=True), debug_log


# ── LOCAL STATS ENGINE ────────────────────────────────────────
def compute_stats(df: pd.DataFrame) -> dict:
    all_nums = [n for row in df["numbers"] for n in row]
    all_lbs  = list(df["lifeBall"])
    freq     = Counter(all_nums)
    lb_freq  = Counter(all_lbs)

    hot   = [n for n,_ in freq.most_common(8)]
    cold  = [n for n,_ in freq.most_common()[:-9:-1]]

    lists = list(df["numbers"])
    gaps  = {}
    for n in range(1, 37):
        gaps[n] = next((i for i,row in enumerate(lists) if n in row), len(df))
    overdue = sorted(gaps, key=gaps.get, reverse=True)[:8]

    odd_c  = [sum(1 for n in row if n%2!=0) for row in df["numbers"]]
    even_c = [sum(1 for n in row if n%2==0) for row in df["numbers"]]

    # Pair frequency
    pairs = Counter()
    for row in df["numbers"]:
        nums = list(row)
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                pairs[(nums[i], nums[j])] += 1
    hot_pairs = [list(p) for p,_ in pairs.most_common(5)]

    return {
        "hotNumbers":     hot,
        "coldNumbers":    cold,
        "overdueNumbers": overdue,
        "hotLifeBalls":   [n for n,_ in lb_freq.most_common(3)],
        "totalDraws":     len(df),
        "avgOdd":         round(sum(odd_c)/len(odd_c), 1),
        "avgEven":        round(sum(even_c)/len(even_c), 1),
        "frequency":      dict(freq.most_common()),
        "gaps":           gaps,
        "hotPairs":       hot_pairs,
    }


# ── GEMINI AI — called ONCE after all data is ready ──────────
def call_gemini(prompt, api_key):
    r = requests.post(
        GEMINI_URL.format(key=api_key),
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 2000},
        },
        timeout=90
    )
    if r.status_code == 429: return None, "rl"
    if r.status_code == 401: raise ValueError("Invalid API key")
    if not r.ok: raise ValueError(f"Gemini {r.status_code}")
    return r.json()["candidates"][0]["content"]["parts"][0]["text"], "ok"


def ai_predict(stats, api_key):
    prompt = f"""Magnum Life Malaysia lottery expert.
Rules: Pick 8 numbers from 1-36 + 1 Life Ball from 1-36. Weekly Wednesday draw.

REAL STATISTICS from {stats['totalDraws']} actual draws:
- Hot (most frequent 8): {stats['hotNumbers']}
- Cold (least frequent 8): {stats['coldNumbers']}
- Most overdue 8 (longest gap): {stats['overdueNumbers']}
- Hot Life Balls top 3: {stats['hotLifeBalls']}
- Avg odd per draw: {stats['avgOdd']}, avg even: {stats['avgEven']}
- Top 5 number pairs: {stats['hotPairs']}
- Full frequency: {stats['frequency']}

Generate 3 prediction sets using the real stats.
Return ONLY raw JSON (no markdown, no text before/after):
{{"predictions":[
  {{"strategy":"🔥 Hot Numbers","numbers":[8 sorted unique ints 1-36],"lifeBall":N,"confidence":"High","reason":"short"}},
  {{"strategy":"⚖️ Balanced Mix","numbers":[8 sorted unique ints 1-36],"lifeBall":N,"confidence":"High","reason":"short"}},
  {{"strategy":"❄️ Due Numbers","numbers":[8 sorted unique ints 1-36],"lifeBall":N,"confidence":"Medium","reason":"short"}}
],"nextDraw":"YYYY-MM-DD"}}"""

    raw, status = call_gemini(prompt, api_key)
    if status == "rl": return None, "rl"
    return xj(raw), "ok"


# ── UI ────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:20px 0 8px">
  <div style="color:#FFD700;font-size:.7rem;letter-spacing:4px;text-transform:uppercase;margin-bottom:6px">Malaysia</div>
  <h1 style="margin:0;font-size:2.4rem;font-weight:900;letter-spacing:2px">MAGNUM <span style="color:#FFD700">LIFE</span></h1>
  <div style="color:#555;font-size:.62rem;letter-spacing:3px;text-transform:uppercase;margin-top:5px">AI Number Predictor</div>
  <div style="color:#2a2a2a;font-size:.58rem;margin-top:3px">All history from lottolyzer.com · Free AI</div>
</div>
<hr style="border-color:#1E1E30">
""", unsafe_allow_html=True)

api_key = st.secrets.get("GEMINI_API_KEY","")
if not api_key:
    st.warning("⚙️ Add `GEMINI_API_KEY` in Streamlit → Manage App → Settings → Secrets")
    st.code('GEMINI_API_KEY = "AIzaSy..."')
    st.stop()

st.markdown('<div style="background:#0d1f0d;border:1px solid #4CAF5030;border-radius:8px;padding:7px 14px;font-size:.7rem;color:#4CAF50;text-align:center;margin-bottom:14px">✅ Google Gemini Free Tier Active</div>',unsafe_allow_html=True)

if st.button("⚡  FETCH & PREDICT", use_container_width=True, type="primary"):
    with st.status("Running...", expanded=True) as sui:

        # ── PHASE 1: SCRAPE ALL DATA ──────────────────────────
        st.write("🌐 Fetching all draw history from Lottolyzer.com...")
        st.caption(f"Scraping {LAST_PAGE} pages (~{LAST_PAGE*50} draws total)")
        pb = st.progress(0, text="Starting scrape...")
        df, dbg = scrape_all(last_page=LAST_PAGE, pbar=pb)
        pb.empty()

        with st.expander(f"📋 Scrape log ({LAST_PAGE} pages)"):
            st.code("\n".join(dbg))

        if df.empty:
            st.error("❌ No draws scraped. Check the scrape log above.")
            st.info("The log shows HTTP status per page. If all show 200 but 0 draws, the HTML structure changed — share the log here to fix.")
            st.stop()

        st.write(f"✅ **{len(df)} draws** collected across {LAST_PAGE} pages")

        # ── PHASE 2: LOCAL STATS (no AI) ─────────────────────
        st.write("🔢 Computing statistics from real draw history...")
        stats = compute_stats(df)
        st.write(f"✅ Stats ready — {stats['totalDraws']} draws analysed")

        # ── PHASE 3: AI PREDICTION (1 call only) ─────────────
        st.write("🧠 Sending real stats to AI for prediction...")
        result, ai_st = ai_predict(stats, api_key)

        if ai_st == "rl":
            st.write("⏳ Rate limit — waiting 65s then retrying once...")
            pb2 = st.progress(0)
            for i in range(65):
                time.sleep(1)
                pb2.progress((i+1)/65, text=f"Retrying in {64-i}s...")
            pb2.empty()
            result, _ = ai_predict(stats, api_key)

        if not result:
            st.error("❌ AI call failed after retry. Wait 1 min and try again.")
            st.stop()

        preds     = result.get("predictions", [])
        next_draw = result.get("nextDraw", "Next Wednesday")
        sui.update(label="✅ Predictions ready!", state="complete")

        # ── RESULTS ───────────────────────────────────────────
        st.markdown('<div class="sl">📊 Overview</div>', unsafe_allow_html=True)
        c1,c2,c3 = st.columns(3)
        c1.metric("Draws Analysed", stats["totalDraws"])
        c2.metric("Avg Odd/Even", f"{stats['avgOdd']}/{stats['avgEven']}")
        c3.metric("Next Draw", next_draw)

        st.markdown('<div class="sl">🔥 Hot Numbers</div>', unsafe_allow_html=True)
        st.markdown(B(stats["hotNumbers"],"h"), unsafe_allow_html=True)

        st.markdown('<div class="sl">❄️ Cold Numbers</div>', unsafe_allow_html=True)
        st.markdown(B(stats["coldNumbers"],"c"), unsafe_allow_html=True)

        st.markdown('<div class="sl">⏰ Most Overdue</div>', unsafe_allow_html=True)
        st.markdown(B(stats["overdueNumbers"],"o"), unsafe_allow_html=True)

        st.markdown('<div class="sl">🔴 Hot Life Balls</div>', unsafe_allow_html=True)
        st.markdown(B(stats["hotLifeBalls"],"r"), unsafe_allow_html=True)

        st.markdown('<div class="sl">🎯 AI Predictions — Next Draw</div>', unsafe_allow_html=True)
        for i, p in enumerate(preds):
            card = "pb" if i==1 else "pc"
            ctag = "th" if p.get("confidence")=="High" else "tm"
            best = '<span style="background:#FFD700;color:#000;font-size:.55rem;font-weight:900;padding:2px 7px;border-radius:4px;margin-left:6px">BEST BET</span>' if i==1 else ""
            tc   = "#FFD700" if i==1 else "#ccc"
            mb   = B(p.get("numbers",[]),"g" if i==1 else "n")
            lb   = B([p.get("lifeBall","?")],"r")
            st.markdown(f"""
<div class="{card}">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
    <span style="font-weight:700;font-size:.95rem;color:{tc}">{p.get('strategy','')} {best}</span>
    <span class="{ctag}">{p.get('confidence','')} Confidence</span>
  </div>
  <div style="margin-bottom:10px">{mb}</div>
  <div style="display:flex;align-items:center;gap:10px">
    <span style="color:#444;font-size:.6rem;letter-spacing:1px;text-transform:uppercase">Life Ball</span>
    {lb}
    <span style="color:#2a2a2a;font-size:.72rem;font-style:italic">{p.get('reason','')}</span>
  </div>
</div>""", unsafe_allow_html=True)
            st.code(f"Numbers: {', '.join(map(str,p.get('numbers',[])))}  |  Life Ball: {p.get('lifeBall','?')}", language=None)

        st.markdown('<div class="sl">📅 Recent Draw History</div>', unsafe_allow_html=True)
        for _, row in df.head(10).iterrows():
            st.markdown(f"""
<div class="dr">
  <span style="color:#555;font-size:.68rem;margin-right:8px">{row['date']}</span>
  {B(row['numbers'],'n')} {B([row['lifeBall']],'r')}
</div>""", unsafe_allow_html=True)

        st.markdown('<div class="disc">⚠️ Entertainment only. Lottery is random. Play responsibly.</div>', unsafe_allow_html=True)
