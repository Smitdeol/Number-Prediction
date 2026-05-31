# ============================================================
#  Magnum Life AI Predictor — Streamlit App
#  Fixed for: Python 3.13 + Pandas 3.x + Streamlit 1.58
#
#  FIXES APPLIED:
#  1. Removed infer_datetime_format (dropped in Pandas 2.0)
#  2. Added anthropic to requirements.txt
#  3. Single AI call (no rate limit 429)
#  4. Robust JSON extraction
# ============================================================

import streamlit as st
import anthropic
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

LAST_PAGE   = 24
BASE_URL    = "https://en.lottolyzer.com/history/malaysia/magnum-life/page/{page}/per-page/50/number-view"
HEADERS     = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# ── Styling ───────────────────────────────────────────────────
st.markdown("""
<style>
body, .stApp { background-color: #0C0C14; color: #ffffff; }
.gold  { color: #FFD700; }
.red   { color: #FF3B3B; }
.ball  {
    display:inline-flex; align-items:center; justify-content:center;
    width:40px; height:40px; border-radius:50%;
    font-weight:900; font-size:0.95rem; margin:3px;
}
.b-normal { background:#1E1E30; color:#fff; }
.b-gold   { background:linear-gradient(135deg,#b8860b,#FFD700); color:#000; }
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
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────
def balls_html(numbers: list, style: str = "normal") -> str:
    cls = {"normal":"b-normal","gold":"b-gold","red":"b-red","hot":"b-hot","cold":"b-cold"}.get(style,"b-normal")
    return "".join(f'<span class="ball {cls}">{n}</span>' for n in numbers)

def extract_json(raw: str) -> dict:
    s = re.sub(r'```json|```', '', raw, flags=re.IGNORECASE).strip()
    start, end = s.find('{'), s.rfind('}')
    if start == -1 or end == -1:
        raise ValueError("No JSON found in AI response")
    return json.loads(s[start:end+1])

# ── Scraper ────────────────────────────────────────────────────
def scrape_all_history(last_page: int = LAST_PAGE) -> pd.DataFrame:
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
                        "date":     date_raw,
                        "numbers":  sorted(nums[:8]),
                        "lifeBall": nums[8],
                    })
            time.sleep(0.3)          # polite crawl delay
        except Exception:
            continue

    df = pd.DataFrame(rows)
    if not df.empty and "date" in df.columns:
        # ✅ FIX: removed infer_datetime_format (deprecated in Pandas 2.0)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date", ascending=False, na_position="last")
        df = df.dropna(subset=["date"])
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df

# ── AI: analyse + predict (single API call) ───────────────────
def ai_analyse_and_predict(draws: list) -> dict:
    client = anthropic.Anthropic()

    has_data = len(draws) >= 5
    draw_json = json.dumps(draws[:50]) if has_data else "[]"

    prompt = f"""You are a Magnum Life lottery analyst for Malaysia.

Magnum Life rules: 8 main numbers from 1–36, 1 Life Ball from 1–36. Weekly Wednesday draw.

{"DRAW HISTORY (" + str(len(draws)) + " draws):" if has_data else "No scraped data. Generate 25 realistic draws first."}
{draw_json}

Tasks:
1. {"Analyse the above draws" if has_data else "Generate 25 realistic weekly draws (most recent: 2026-06-04)"} 
2. Count number frequency — identify top 8 hot, bottom 8 cold
3. Count Life Ball frequency — top 3
4. Calculate average odd/even per draw (round to 1 decimal)
5. Generate 3 prediction sets — each with EXACTLY 8 unique numbers from 1–36

Return ONLY raw JSON, zero markdown, zero extra text:
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

    msg = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}],
    )
    return extract_json(msg.content[0].text)

# ── UI ─────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:24px 0 12px">
  <div style="color:#FFD700;font-size:.7rem;letter-spacing:4px;text-transform:uppercase;margin-bottom:6px">Malaysia</div>
  <h1 style="margin:0;font-size:2.6rem;font-weight:900;letter-spacing:2px">
    MAGNUM <span style="color:#FFD700">LIFE</span>
  </h1>
  <div style="color:#555;font-size:.65rem;letter-spacing:3px;text-transform:uppercase;margin-top:6px">
    AI Number Predictor
  </div>
  <div style="color:#333;font-size:.6rem;margin-top:4px">lottolyzer.com · auto-fetch</div>
</div>
<hr style="border-color:#1E1E30;margin-bottom:8px">
""", unsafe_allow_html=True)

if st.button("⚡  FETCH & PREDICT", use_container_width=True, type="primary"):
    with st.status("Running...", expanded=True) as status_ui:

        # Step 1 — scrape
        st.write("🌐 Scraping Lottolyzer.com...")
        df = scrape_all_history(last_page=LAST_PAGE)

        if not df.empty:
            st.write(f"✅ Scraped {len(df)} draws from Lottolyzer")
            draws_list = df.to_dict("records")
        else:
            st.write("⚠️ Scrape returned no data — AI will generate baseline draws")
            draws_list = []

        # Step 2 — AI analyse + predict (1 call)
        st.write("🧠 AI analysing patterns & generating predictions...")
        try:
            result     = ai_analyse_and_predict(draws_list)
            draws_out  = result.get("draws",  draws_list[:10] if draws_list else [])
            analysis   = result.get("analysis",  {})
            preds      = result.get("predictions", [])
            next_draw  = result.get("nextDraw", "Next Wednesday")

            st.write(f"✅ {analysis.get('totalDrawsAnalysed', len(draws_out))} draws analysed")
            status_ui.update(label="✅ Done!", state="complete")

            # ── Stats ──────────────────────────────────────────
            st.markdown('<div class="sec-label">📊 Overview</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Draws",     analysis.get("totalDrawsAnalysed", len(draws_out)))
            c2.metric("Odd/Even",  f"{analysis.get('avgOdd','?')} / {analysis.get('avgEven','?')}")
            c3.metric("Next Draw", next_draw)

            # ── Hot & Cold ─────────────────────────────────────
            st.markdown('<div class="sec-label">🔥 Hot &nbsp; ❄️ Cold</div>', unsafe_allow_html=True)
            col_h, col_c = st.columns(2)
            with col_h:
                st.markdown(balls_html(analysis.get("hotNumbers",[]),"hot"), unsafe_allow_html=True)
            with col_c:
                st.markdown(balls_html(analysis.get("coldNumbers",[]),"cold"), unsafe_allow_html=True)

            # ── Predictions ────────────────────────────────────
            st.markdown('<div class="sec-label">🎯 Predictions — Next Draw</div>', unsafe_allow_html=True)
            for i, p in enumerate(preds):
                card  = "pred-card-best" if i == 1 else "pred-card"
                ctag  = "tag-h" if p.get("confidence") == "High" else "tag-m"
                best  = '<span style="background:#FFD700;color:#000;font-size:.55rem;font-weight:900;padding:2px 7px;border-radius:4px;margin-left:6px">BEST BET</span>' if i == 1 else ""
                title_color = "#FFD700" if i == 1 else "#cccccc"
                main_balls  = balls_html(p.get("numbers",[]), "gold" if i==1 else "normal")
                lb_ball     = balls_html([p.get("lifeBall","?")], "red")

                st.markdown(f"""
<div class="{card}">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
    <span style="font-weight:700;font-size:.95rem;color:{title_color}">{p.get('strategy','')} {best}</span>
    <span class="{ctag}">{p.get('confidence','')} Confidence</span>
  </div>
  <div style="margin-bottom:10px">{main_balls}</div>
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
    <span style="color:#444;font-size:.6rem;letter-spacing:1px;text-transform:uppercase">Life Ball</span>
    {lb_ball}
    <span style="color:#333;font-size:.72rem;font-style:italic">{p.get('reason','')}</span>
  </div>
</div>""", unsafe_allow_html=True)
                st.code(
                    f"Numbers: {', '.join(map(str, p.get('numbers',[])))}  |  Life Ball: {p.get('lifeBall','?')}",
                    language=None
                )

            # ── Recent draws ───────────────────────────────────
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
            st.info("Make sure ANTHROPIC_API_KEY is set in Streamlit → Settings → Secrets")
             
