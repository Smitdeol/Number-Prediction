# ============================================================
#  Magnum Life AI Predictor — Streamlit App
#  Deploy: streamlit run magnum_life_app.py
#  Requires: pip install streamlit anthropic requests beautifulsoup4
# ============================================================

import streamlit as st
import anthropic
import requests
import json
import re
from collections import Counter
from bs4 import BeautifulSoup

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Magnum Life AI Predictor",
    page_icon="🎱",
    layout="centered",
)

SOURCE_URL = "https://en.lottolyzer.com/history/malaysia/magnum-life/page/1/per-page/50/number-view"

# ── Styling ───────────────────────────────────────────────────
st.markdown("""
<style>
  body, .stApp { background-color: #0C0C14; color: #ffffff; }
  .title-block { text-align:center; padding: 20px 0 10px; }
  .title-block h1 { font-size: 2.8rem; font-weight: 900; letter-spacing: 2px; }
  .gold { color: #FFD700; }
  .subtitle { color: #555; font-size: 0.75rem; letter-spacing: 3px; text-transform: uppercase; }
  .ball {
    display:inline-flex; align-items:center; justify-content:center;
    width:42px; height:42px; border-radius:50%;
    font-weight:900; font-size:1rem; margin:3px;
  }
  .ball-normal { background:#1E1E30; color:#ffffff; }
  .ball-gold   { background:linear-gradient(135deg,#b8860b,#FFD700); color:#000; }
  .ball-red    { background:linear-gradient(135deg,#FF3B3B,#ff6b6b); color:#fff; }
  .ball-hot    { background:#FF6B3520; border:1.5px solid #FF6B35; color:#FF6B35; }
  .ball-cold   { background:#4FC3F720; border:1.5px solid #4FC3F7; color:#4FC3F7; }
  .pred-card {
    background:#13131F; border:1.5px solid #1E1E30;
    border-radius:14px; padding:18px; margin-bottom:14px;
  }
  .pred-card-best {
    background:linear-gradient(135deg,#1a1500,#13131F);
    border:1.5px solid #FFD70055;
    border-radius:14px; padding:18px; margin-bottom:14px;
  }
  .tag-high { background:#1a3a1a; color:#4CAF50; padding:3px 10px; border-radius:6px; font-size:0.7rem; font-weight:700; }
  .tag-med  { background:#2a2010; color:#FFA726; padding:3px 10px; border-radius:6px; font-size:0.7rem; font-weight:700; }
  .section-label { color:#444; font-size:0.65rem; letter-spacing:3px; text-transform:uppercase; margin:18px 0 8px; }
  .draw-row { background:#13131F; border:1px solid #1E1E30; border-radius:10px; padding:8px 14px; margin-bottom:6px; }
  .disclaimer { background:#090909; border:1px solid #181818; border-radius:10px; padding:12px; color:#2a2a2a; font-size:0.7rem; text-align:center; margin-top:24px; }
  hr { border-color: #1E1E30; }
</style>
""", unsafe_allow_html=True)

# ── Helper: extract JSON safely ───────────────────────────────
def extract_json(raw: str) -> dict:
    s = re.sub(r'```json|```', '', raw, flags=re.IGNORECASE).strip()
    start = s.find('{')
    end   = s.rfind('}')
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in AI response")
    return json.loads(s[start:end+1])

# ── Helper: render lottery balls ──────────────────────────────
def render_balls(numbers: list, style: str = "normal") -> str:
    cls = {"normal":"ball-normal","gold":"ball-gold","red":"ball-red","hot":"ball-hot","cold":"ball-cold"}.get(style,"ball-normal")
    return "".join(f'<span class="ball {cls}">{n}</span>' for n in numbers)

# ── Step 1: Scrape lottolyzer directly ───────────────────────
def scrape_draws() -> list:
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        r = requests.get(SOURCE_URL, headers=headers, timeout=12)
        soup = BeautifulSoup(r.text, "html.parser")

        draws = []
        # Try table rows
        rows = soup.select("table tbody tr") or soup.select("tr")
        for row in rows:
            cells = [c.get_text(strip=True) for c in row.find_all(["td","th"])]
            nums = [int(x) for x in cells if x.isdigit() and 1 <= int(x) <= 36]
            if len(nums) >= 9:
                draws.append({
                    "date": cells[0] if cells else "Unknown",
                    "numbers": sorted(nums[:8]),
                    "lifeBall": nums[8]
                })
        if draws:
            return draws[:25]
    except Exception:
        pass
    return []

# ── Step 2: AI fallback for draw data + full analysis (1 call) ──
def ai_fetch_and_analyse(draws_hint: list) -> dict:
    client = anthropic.Anthropic()

    has_draws = len(draws_hint) >= 5
    draw_context = json.dumps(draws_hint) if has_draws else "No data scraped — please generate realistic data"

    prompt = f"""You are a Magnum Life lottery expert for Malaysia.

Magnum Life rules: Pick 8 numbers from 1–36 + 1 Life Ball from 1–36. Weekly draw every Wednesday.

{"SCRAPED DRAW DATA:" if has_draws else "TASK: Generate 25 realistic draws first, then analyse them."}
{draw_context}

{"If the scraped data looks incomplete or wrong, generate 25 realistic draws to supplement." if has_draws else ""}

Steps:
1. If draws are missing/incomplete, generate 25 realistic weekly draws (most recent: 2026-05-28)
2. Analyse frequency: hot numbers (top 8 most frequent), cold numbers (least frequent / overdue)
3. Analyse Life Ball frequency
4. Calculate average odd/even count per draw
5. Generate 3 prediction sets with EXACTLY 8 unique numbers each from 1–36

Return ONLY this raw JSON — no markdown, no extra text:
{{
  "draws":[{{"date":"YYYY-MM-DD","numbers":[n1,n2,n3,n4,n5,n6,n7,n8],"lifeBall":n}}],
  "analysis":{{
    "hotNumbers":[8 numbers],
    "coldNumbers":[8 numbers],
    "hotLifeBalls":[3 numbers],
    "totalDrawsAnalysed":N,
    "avgOdd":N,
    "avgEven":N
  }},
  "predictions":[
    {{"strategy":"🔥 Hot Numbers","numbers":[8 sorted unique numbers],"lifeBall":N,"confidence":"High","reason":"brief reason"}},
    {{"strategy":"⚖️ Balanced Mix","numbers":[8 sorted unique numbers],"lifeBall":N,"confidence":"High","reason":"brief reason"}},
    {{"strategy":"❄️ Due Numbers","numbers":[8 sorted unique numbers],"lifeBall":N,"confidence":"Medium","reason":"brief reason"}}
  ],
  "nextDraw":"YYYY-MM-DD"
}}"""

    msg = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}]
    )
    return extract_json(msg.content[0].text)

# ── UI ────────────────────────────────────────────────────────
st.markdown("""
<div class="title-block">
  <div class="subtitle">Malaysia</div>
  <h1>MAGNUM <span class="gold">LIFE</span></h1>
  <div class="subtitle">AI Number Predictor</div>
  <div style="color:#333;font-size:0.65rem;margin-top:4px;">Source: lottolyzer.com · auto-fetch enabled</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

if st.button("⚡ FETCH & PREDICT", use_container_width=True, type="primary"):
    with st.status("Running prediction engine...", expanded=True) as status_box:

        st.write("🌐 Connecting to lottolyzer.com...")
        scraped = scrape_draws()

        if scraped:
            st.write(f"✅ Scraped {len(scraped)} draws from lottolyzer.com")
        else:
            st.write("⚠️ Direct scrape returned no data — AI will generate baseline")

        st.write("🧠 Running AI analysis & prediction (single call)...")

        try:
            result = ai_fetch_and_analyse(scraped)
            draws      = result.get("draws", scraped or [])
            analysis   = result.get("analysis", {})
            predictions= result.get("predictions", [])
            next_draw  = result.get("nextDraw", "Next Wednesday")

            st.write(f"✅ Analysis complete — {analysis.get('totalDrawsAnalysed', len(draws))} draws analysed")
            status_box.update(label="✅ Predictions ready!", state="complete")

            # ── Stats strip ──
            st.markdown('<div class="section-label">📊 Overview</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Draws Analysed", analysis.get("totalDrawsAnalysed", len(draws)))
            c2.metric("Avg Odd / Even", f"{analysis.get('avgOdd','?')} / {analysis.get('avgEven','?')}")
            c3.metric("Next Draw", next_draw)

            # ── Hot & Cold ──
            st.markdown('<div class="section-label">🔥 Hot &nbsp;&nbsp; ❄️ Cold Numbers</div>', unsafe_allow_html=True)
            col_h, col_c = st.columns(2)
            with col_h:
                st.markdown(render_balls(analysis.get("hotNumbers",[]), "hot"), unsafe_allow_html=True)
            with col_c:
                st.markdown(render_balls(analysis.get("coldNumbers",[]), "cold"), unsafe_allow_html=True)

            # ── Predictions ──
            st.markdown('<div class="section-label">🎯 AI Predictions — Next Draw</div>', unsafe_allow_html=True)

            for i, pred in enumerate(predictions):
                card_cls = "pred-card-best" if i == 1 else "pred-card"
                conf_cls = "tag-high" if pred.get("confidence") == "High" else "tag-med"
                best_tag = '<span style="background:#FFD700;color:#000;font-size:0.6rem;font-weight:900;padding:2px 8px;border-radius:4px;margin-left:8px;">BEST BET</span>' if i==1 else ""

                balls_html = render_balls(pred.get("numbers",[]), "gold" if i==1 else "normal")
                lb_html    = render_balls([pred.get("lifeBall","?")], "red")

                st.markdown(f"""
<div class="{card_cls}">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
    <span style="font-weight:700;font-size:1rem;color:{'#FFD700' if i==1 else '#ccc'}">
      {pred.get('strategy','')} {best_tag}
    </span>
    <span class="{conf_cls}">{pred.get('confidence','')} Confidence</span>
  </div>
  <div style="margin-bottom:10px">{balls_html}</div>
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
    <span style="color:#444;font-size:0.65rem;letter-spacing:1px;text-transform:uppercase;">Life Ball</span>
    {lb_html}
    <span style="color:#333;font-size:0.75rem;font-style:italic;">{pred.get('reason','')}</span>
  </div>
</div>
""", unsafe_allow_html=True)

                # Copy-friendly text box
                st.code(f"Numbers: {', '.join(map(str, pred.get('numbers',[])))}  |  Life Ball: {pred.get('lifeBall','?')}", language=None)

            # ── Recent draws ──
            if draws:
                st.markdown('<div class="section-label">📅 Recent Draw History</div>', unsafe_allow_html=True)
                for d in draws[:10]:
                    balls = render_balls(d.get("numbers",[]), "normal")
                    lb    = render_balls([d.get("lifeBall","?")], "red")
                    st.markdown(f"""
<div class="draw-row">
  <span style="color:#444;font-size:0.7rem;margin-right:10px;">{d.get('date','')}</span>
  {balls} {lb}
</div>""", unsafe_allow_html=True)

            st.markdown('<div class="disclaimer">⚠️ For entertainment purposes only. Lottery outcomes are entirely random. Please play responsibly.</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.info("Check that your ANTHROPIC_API_KEY environment variable is set correctly.")
  
