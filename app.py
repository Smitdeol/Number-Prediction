# ============================================================
#  Magnum Life AI Predictor — Streamlit App
#  FREE VERSION — Uses Google Gemini API (free tier)
#  Free limit: 15 requests/min, 1500/day — more than enough
#
#  Setup: get free key at https://aistudio.google.com/apikey
# ============================================================

import streamlit as st
import requests
import json
import re
import time
from collections import Counter
from bs4 import BeautifulSoup
import pandas as pd

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Magnum Life AI Predictor",
    page_icon="🎱",
    layout="centered",
)

LAST_PAGE = 24
BASE_URL  = "https://en.lottolyzer.com/history/malaysia/magnum-life/page/{page}/per-page/50/number-view"
HEADERS   = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# ── Styling ───────────────────────────────────────────────────
st.markdown("""
<style>
body, .stApp { background-color: #0C0C14; color: #ffffff; }
.ball {
    display:inline-flex; align-items:center; justify-content:center;
    width:40px; height:40px; border-radius:50%;
    font-weight:900; font-size:0.95rem; margin:3px;
}
.b-normal { background:#1E1E30; color:#fff; }
.b-gold   { background:linear-gradient(135deg,#b8860b,#FFD700); color:#000; box-shadow:0 3px 10px rgba(255,215,0,.3); }
.b-red    { background:linear-gradient(135deg,#FF3B3B,#ff6b6b); color:#fff; box-shadow:0 3px 10px rgba(255,59,59,.4); }
.b-hot    { background:#FF6B3520; border:1.5px solid #FF6B35; color:#FF6B35; }
.b-cold   { background:#4FC3F720; border:1.5px solid #4FC3F7; color:#4FC3F7; }
.pred-card      { background:#13131F; border:1.5px solid #1E1E30; border-radius:14px; padding:16px; margin-bottom:12px; }
.pred-card-best { background:linear-gradient(135deg,#1a1500,#13131F); border:1.5px solid #FFD70055; border-radius:14px; padding:16px; margin-bottom:12px; }
.tag-h { background:#1a3a1a; color:#4CAF50; padding:2px 8px; border-radius:5px; font-size:0.65rem; font-weight:700; }
.tag-m { background:#2a2010; color:#FFA726; padding:2px 8px; border-radius:5px; font-size:0.65rem; font-weight:700; }
.draw-row { background:#13131F; border:1px solid #1E1E30; border-radius:8px; padding:8px 12px; margin-bottom:5px; }
.sec-label { color:#444; font-size:0.62rem; letter-spacing:3px; text-transform:uppercase; margin:16px 0 8px; }
.disclaimer { background:#090909; border:1px solid #181818; border-radius:8px; padding:12px; color:#2a2a2a; font-size:0.65rem; text-align:center; margin-top:20px; }
.free-badge { background:#1a3a1a; color:#4CAF50; border:1px solid #4CAF5044; border-radius:8px; padding:6px 12px; font-size:0.7rem; text-align:center; margin-bottom:16px; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────
def balls_html(numbers, style="normal"):
    cls = {"normal":"b-normal","gold":"b-gold","red":"b-red","hot":"b-hot","cold":"b-cold"}.get(style,"b-normal")
    return "".join(f'<span class="ball {cls}">{n}</span>' for n in numbers)

def extract_json(raw):
    s = re.sub(r'```json|```', '', raw, flags=re.IGNORECASE).strip()
    start, end = s.find('{'), s.rfind('}')
    if start == -1 or end == -1:
        raise ValueError("No JSON found in AI response")
    return json.loads(s[start:end+1])

# ── FREE AI call — Google Gemini ──────────────────────────────
def call_gemini(prompt: str, api_key: str) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 4000}
    }
    r = requests.post(url, json=body, timeout=60)
    if r.status_code == 401:
        raise ValueError("Invalid Gemini API key. Check your key at aistudio.google.com")
    if r.status_code == 429:
        raise ValueError("Rate limit hit. Wait 1 minute and try again.")
    if not r.ok:
        raise ValueError(f"Gemini API error {r.status_code}: {r.text[:200]}")
    data = r.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]

# ── Scraper ───────────────────────────────────────────────────
def scrape_all_history(last_page=LAST_PAGE):
    rows = []
    for page in range(1, last_page + 1):
        try:
            url  = BASE_URL.format(page=page)
            resp = requests.get(url, headers=HEADERS, timeout=10)
            soup = BeautifulSoup(resp.text, "html.parser")
            for tr in soup.select("table tbody tr"):
                tds  = tr.find_all("td")
                if len(tds) < 3:
                    continue
                date_raw = tds[0].get_text(strip=True)
                nums = [int(td.get_text(strip=True)) for td in tds[1:]
                        if td.get_text(strip=True).isdigit()
                        and 1 <= int(td.get_text(strip=True)) <= 36]
                if len(nums) >= 9:
                    rows.append({
                        "date":    date_raw,
                        "numbers": sorted(nums[:8]),
                        "lifeBall": nums[8],
                    })
            time.sleep(0.3)
        except Exception:
            continue

    df = pd.DataFrame(rows)
    if not df.empty and "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date", ascending=False, na_position="last")
        df = df.dropna(subset=["date"])
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df

# ── AI: analyse + predict ─────────────────────────────────────
def ai_analyse_and_predict(draws: list, api_key: str) -> dict:
    has_data  = len(draws) >= 5
    draw_json = json.dumps(draws[:50]) if has_data else "[]"

    prompt = f"""You are a Magnum Life lottery analyst for Malaysia.

Magnum Life rules: 8 main numbers from 1-36, 1 Life Ball from 1-36. Weekly Wednesday draw.

{"DRAW HISTORY (" + str(len(draws)) + " draws):" if has_data else "No scraped data — generate 25 realistic draws first."}
{draw_json}

Tasks:
1. {"Analyse the above draws" if has_data else "Generate 25 realistic weekly draws (most recent: 2026-06-04)"}
2. Count frequency — top 8 hot numbers, bottom 8 cold numbers
3. Top 3 Life Ball numbers by frequency
4. Average odd/even count per draw (1 decimal)
5. Generate 3 prediction sets — EXACTLY 8 unique numbers each from 1-36

Return ONLY raw JSON, zero markdown, zero extra text before or after:
{{
  "draws":[{{"date":"YYYY-MM-DD","numbers":[n1,n2,n3,n4,n5,n6,n7,n8],"lifeBall":n}}],
  "analysis":{{
    "hotNumbers":[8 ints],
    "coldNumbers":[8 ints],
    "hotLifeBalls":[3 ints],
    "totalDrawsAnalysed":N,
    "avgOdd":N,
    "avgEven":N
  }},
  "predictions":[
    {{"strategy":"🔥 Hot Numbers","numbers":[8 sorted ints],"lifeBall":N,"confidence":"High","reason":"short reason"}},
    {{"strategy":"⚖️ Balanced Mix","numbers":[8 sorted ints],"lifeBall":N,"confidence":"High","reason":"short reason"}},
    {{"strategy":"❄️ Due Numbers","numbers":[8 sorted ints],"lifeBall":N,"confidence":"Medium","reason":"short reason"}}
  ],
  "nextDraw":"YYYY-MM-DD"
}}"""

    raw    = call_gemini(prompt, api_key)
    result = extract_json(raw)
    if not result.get("predictions"):
        raise ValueError("Predictions missing from AI response")
    return result

# ── UI ────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:24px 0 10px">
  <div style="color:#FFD700;font-size:.7rem;letter-spacing:4px;text-transform:uppercase;margin-bottom:6px">Malaysia</div>
  <h1 style="margin:0;font-size:2.6rem;font-weight:900;letter-spacing:2px">
    MAGNUM <span style="color:#FFD700">LIFE</span>
  </h1>
  <div style="color:#555;font-size:.65rem;letter-spacing:3px;text-transform:uppercase;margin-top:6px">AI Number Predictor</div>
  <div style="color:#333;font-size:.6rem;margin-top:4px">lottolyzer.com · Powered by Google Gemini (Free)</div>
</div>
<hr style="border-color:#1E1E30">
""", unsafe_allow_html=True)

# ── API key check ─────────────────────────────────────────────
api_key = st.secrets.get("GEMINI_API_KEY", "")
if not api_key:
    st.markdown("""
<div style="background:#1a1a00;border:1px solid #FFD70044;border-radius:12px;padding:20px;margin:10px 0">
  <div style="color:#FFD700;font-weight:700;font-size:1rem;margin-bottom:10px">⚙️ Free Setup Required</div>
  <div style="color:#aaa;font-size:0.8rem;line-height:1.8">
    1. Go to <b style="color:#4FC3F7">aistudio.google.com/apikey</b><br>
    2. Sign in with Google (free)<br>
    3. Click <b>Create API Key</b> → copy it<br>
    4. In Streamlit → <b>Manage App → Settings → Secrets</b><br>
    5. Paste: <code>GEMINI_API_KEY = "your-key-here"</code><br>
    6. Save → done ✅
  </div>
</div>
""", unsafe_allow_html=True)
    st.stop()

st.markdown('<div class="free-badge">✅ Free Tier Active — Google Gemini AI</div>', unsafe_allow_html=True)

if st.button("⚡  FETCH & PREDICT", use_container_width=True, type="primary"):
    with st.status("Running...", expanded=True) as status_ui:

        # Scrape
        st.write("🌐 Scraping Lottolyzer.com...")
        df = scrape_all_history(last_page=LAST_PAGE)

        if not df.empty:
            st.write(f"✅ Scraped {len(df)} draws from Lottolyzer")
            draws_list = df.to_dict("records")
        else:
            st.write("⚠️ Scrape returned no data — AI will generate baseline draws")
            draws_list = []

        # AI analyse
        st.write("🧠 AI analysing & generating predictions...")
        try:
            result    = ai_analyse_and_predict(draws_list, api_key)
            draws_out = result.get("draws",  draws_list[:10] if draws_list else [])
            analysis  = result.get("analysis", {})
            preds     = result.get("predictions", [])
            next_draw = result.get("nextDraw", "Next Wednesday")

            st.write(f"✅ {analysis.get('totalDrawsAnalysed', len(draws_out))} draws analysed")
            status_ui.update(label="✅ Predictions ready!", state="complete")

            # Stats
            st.markdown('<div class="sec-label">📊 Overview</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Draws",    analysis.get("totalDrawsAnalysed", len(draws_out)))
            c2.metric("Odd/Even", f"{analysis.get('avgOdd','?')} / {analysis.get('avgEven','?')}")
            c3.metric("Next Draw",next_draw)

            # Hot & Cold
            st.markdown('<div class="sec-label">🔥 Hot &nbsp; ❄️ Cold</div>', unsafe_allow_html=True)
            col_h, col_c = st.columns(2)
            with col_h:
                st.markdown(balls_html(analysis.get("hotNumbers",[]),"hot"), unsafe_allow_html=True)
            with col_c:
                st.markdown(balls_html(analysis.get("coldNumbers",[]),"cold"), unsafe_allow_html=True)

            # Predictions
            st.markdown('<div class="sec-label">🎯 Predictions — Next Draw</div>', unsafe_allow_html=True)
            for i, p in enumerate(preds):
                card  = "pred-card-best" if i == 1 else "pred-card"
                ctag  = "tag-h" if p.get("confidence") == "High" else "tag-m"
                best  = '<span style="background:#FFD700;color:#000;font-size:.55rem;font-weight:900;padding:2px 7px;border-radius:4px;margin-left:6px">BEST BET</span>' if i == 1 else ""
                tc    = "#FFD700" if i == 1 else "#cccccc"
                mb    = balls_html(p.get("numbers",[]), "gold" if i==1 else "normal")
                lb    = balls_html([p.get("lifeBall","?")], "red")

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

            # Recent draws
            if draws_out:
                st.markdown('<div class="sec-label">📅 Recent Draw History</div>', unsafe_allow_html=True)
                for d in draws_out[:10]:
                    nb = balls_html(d.get("numbers",[]), "normal")
                    lb = balls_html([d.get("lifeBall","?")], "red")
                    st.markdown(f"""
<div class="draw-row">
  <span style="color:#444;font-size:.68rem;margin-right:8px">{d.get('date','')}</span>
  {nb} {lb}
</div>""", unsafe_allow_html=True)

            st.markdown('<div class="disclaimer">⚠️ For entertainment only. Lottery results are random. Play responsibly.</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ AI Error: {e}")
                
