import os, io, re, time, random, json, itertools
from datetime import datetime, timedelta
from collections import Counter, defaultdict

import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh   # pip install streamlit-autorefresh

# ───────────────────────────────────────────────
# Config
# ───────────────────────────────────────────────
st.set_page_config(page_title="Magnum Life AI Dashboard", layout="wide")
st.title("🎯 Magnum Life — Full AI Prediction Dashboard")

LAST_PAGE            = 24
BASE_URL             = ("https://en.lottolyzer.com/history/malaysia/magnum-life"
                        "/page/{}/per-page/50/number-view")
HEADERS              = {"User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/115.0.0.0 Safari/537.36")}
REQUEST_TIMEOUT      = 20
PAUSE_BETWEEN_PAGES  = 0.35

NUMBERS_PER_DRAW     = 8
TOTAL_NUMBERS        = 36

CACHE_FILENAME       = "past_results.csv"
FEEDBACK_FILENAME    = "feedback_weights.json"
CACHE_MAX_AGE_HOURS  = 6      # shorter = more live

DEFAULT_PRED_COUNT   = 3
DEFAULT_HALF_LIFE    = 60
DEFAULT_W_RECENCY    = 40
DEFAULT_W_GAP        = 20
DEFAULT_W_COOCCUR    = 20
DEFAULT_W_TREND      = 20
MONTE_CARLO_RUNS     = 10_000
AUTOREFRESH_MINS     = 30

ANTHROPIC_API_URL    = "https://api.anthropic.com/v1/messages"
CLAUDE_MODEL         = "claude-sonnet-4-20250514"

# ───────────────────────────────────────────────
# Cache helpers
# ───────────────────────────────────────────────
def is_cache_fresh(path=CACHE_FILENAME, max_age_hours=CACHE_MAX_AGE_HOURS) -> bool:
    if not os.path.exists(path):
        return False
    age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(path))
    return age < timedelta(hours=max_age_hours)

def save_cache(df: pd.DataFrame, path=CACHE_FILENAME):
    df.to_csv(path, index=False)

def load_cache(path=CACHE_FILENAME) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"], dayfirst=True)

# ───────────────────────────────────────────────
# Adaptive feedback
# ───────────────────────────────────────────────
def load_feedback(path=FEEDBACK_FILENAME) -> dict:
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return {str(n): 0.0 for n in range(1, TOTAL_NUMBERS + 1)}

def save_feedback(fb: dict, path=FEEDBACK_FILENAME):
    with open(path, "w") as f:
        json.dump(fb, f)

def apply_feedback_from_draw(fb: dict, predicted: list, actual: list,
                              lr: float = 0.05) -> dict:
    pred_set, actual_set = set(predicted), set(actual)
    fb = dict(fb)
    for n in range(1, TOTAL_NUMBERS + 1):
        key = str(n)
        if n in pred_set and n not in actual_set:
            fb[key] = fb.get(key, 0.0) - lr
        elif n in actual_set and n not in pred_set:
            fb[key] = fb.get(key, 0.0) + lr
        elif n in pred_set and n in actual_set:
            fb[key] = fb.get(key, 0.0) + lr * 0.3
        fb[key] = max(-1.0, min(1.0, fb[key]))
    return fb

# ───────────────────────────────────────────────
# Scraper
# ───────────────────────────────────────────────
DATE_REGEX = re.compile(
    r"(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},\s*\d{4}|"
    r"\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4})",
    re.I,
)

def _parse_numbers(tr):
    nums = []
    for img in tr.find_all("img", alt=True):
        alt = (img.get("alt") or "").strip()
        if alt.isdigit():
            n = int(alt)
            if 1 <= n <= TOTAL_NUMBERS:
                nums.append(n)
    if len(nums) < NUMBERS_PER_DRAW:
        for td in tr.find_all(["td", "div", "span"]):
            cls = " ".join(td.get("class", []))
            if any(k in cls.lower() for k in ["number", "num", "ball"]):
                for tok in re.findall(r"\b\d{1,2}\b", td.get_text(" ", strip=True)):
                    n = int(tok)
                    if 1 <= n <= TOTAL_NUMBERS:
                        nums.append(n)
                        if len(nums) == NUMBERS_PER_DRAW:
                            break
            if len(nums) == NUMBERS_PER_DRAW:
                break
    if len(nums) < NUMBERS_PER_DRAW:
        for tok in re.findall(r"\b\d{1,2}\b", tr.get_text(" ", strip=True)):
            n = int(tok)
            if 1 <= n <= TOTAL_NUMBERS:
                nums.append(n)
                if len(nums) == NUMBERS_PER_DRAW:
                    break
    return nums[:NUMBERS_PER_DRAW]

def _parse_date(tr):
    tds = tr.find_all(["td", "div", "span"])
    if tds:
        m = DATE_REGEX.search(tds[0].get_text(" ", strip=True))
        if m:
            return m.group(0)
    m = DATE_REGEX.search(tr.get_text(" ", strip=True))
    return m.group(0) if m else None

def _fallback(soup):
    rows, imgs = [], soup.find_all("img", alt=lambda a: a and a.strip().isdigit())
    digits = [int(img["alt"].strip()) for img in imgs]
    for i in range(0, len(digits), NUMBERS_PER_DRAW):
        g = digits[i:i + NUMBERS_PER_DRAW]
        if len(g) == NUMBERS_PER_DRAW:
            rows.append({"date": None, **{f"n{j+1}": g[j] for j in range(NUMBERS_PER_DRAW)}})
    return rows

def scrape_all_history(last_page: int = LAST_PAGE) -> pd.DataFrame:
    rows, seen = [], set()
    for page in range(1, last_page + 1):
        url = BASE_URL.format(page)
        try:
            resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        except Exception:
            time.sleep(PAUSE_BETWEEN_PAGES); continue
        if resp.status_code != 200:
            time.sleep(PAUSE_BETWEEN_PAGES); continue
        soup = BeautifulSoup(resp.text, "lxml")
        trs  = soup.select("table tbody tr") or soup.find_all("tr")
        page_new = 0
        for tr in trs:
            nums = _parse_numbers(tr)
            if len(nums) != NUMBERS_PER_DRAW:
                continue
            d   = {"date": _parse_date(tr), **{f"n{i+1}": nums[i] for i in range(NUMBERS_PER_DRAW)}}
            key = (d["date"], tuple(nums)) if d["date"] else tuple(nums)
            if key in seen: continue
            seen.add(key); rows.append(d); page_new += 1
        if page_new == 0:
            for d in _fallback(soup):
                key = (d.get("date"), tuple(d[f"n{i}"] for i in range(1, NUMBERS_PER_DRAW + 1)))
                if key in seen: continue
                seen.add(key); rows.append(d); page_new += 1
        if page_new == 0:
            break
        time.sleep(PAUSE_BETWEEN_PAGES)
    if not rows:
        return pd.DataFrame(columns=["date"] + [f"n{i}" for i in range(1, NUMBERS_PER_DRAW + 1)])
    df = pd.DataFrame(rows)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True, infer_datetime_format=True)
    df = df.sort_values("date", ascending=False, na_position="last").reset_index(drop=True)
    for i in range(1, NUMBERS_PER_DRAW + 1):
        col = f"n{i}"
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    df = df.drop_duplicates(
        subset=["date"] + [f"n{i}" for i in range(1, NUMBERS_PER_DRAW + 1)],
        keep="first").reset_index(drop=True)
    return df

# ───────────────────────────────────────────────
# Analytics Signals
# ───────────────────────────────────────────────
def recency_scores(df, half_life):
    scores = np.zeros(TOTAL_NUMBERS + 1)
    if df.empty: return scores
    weights = 0.5 ** (np.arange(len(df)) / max(1, half_life))
    for idx, w in enumerate(weights):
        row = df.iloc[idx]
        for i in range(1, NUMBERS_PER_DRAW + 1):
            val = row.get(f"n{i}")
            if pd.notna(val):
                v = int(val)
                if 1 <= v <= TOTAL_NUMBERS:
                    scores[v] += w
    return scores

def gap_scores(df):
    scores   = np.zeros(TOTAL_NUMBERS + 1)
    last_seen = {n: None for n in range(1, TOTAL_NUMBERS + 1)}
    for idx in range(len(df)):
        row = df.iloc[idx]
        for i in range(1, NUMBERS_PER_DRAW + 1):
            val = row.get(f"n{i}")
            if pd.notna(val):
                v = int(val)
                if 1 <= v <= TOTAL_NUMBERS and last_seen[v] is None:
                    last_seen[v] = idx
    n_draws = max(len(df), 1)
    for num in range(1, TOTAL_NUMBERS + 1):
        ls = last_seen[num]
        scores[num] = n_draws if ls is None else ls
    mx = scores[1:].max()
    if mx > 0: scores[1:] /= mx
    return scores

def cooccurrence_matrix(df):
    co = np.zeros((TOTAL_NUMBERS + 1, TOTAL_NUMBERS + 1))
    for idx in range(len(df)):
        row  = df.iloc[idx]
        nums = []
        for i in range(1, NUMBERS_PER_DRAW + 1):
            val = row.get(f"n{i}")
            if pd.notna(val):
                v = int(val)
                if 1 <= v <= TOTAL_NUMBERS:
                    nums.append(v)
        for a in nums:
            for b in nums:
                if a != b:
                    co[a][b] += 1
    row_sums = co.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return co / row_sums

def sequence_trend_scores(df, window=30):
    scores = np.zeros(TOTAL_NUMBERS + 1)
    n = min(window, len(df))
    if n < 3: return scores
    recent = df.iloc[:n]
    x = np.arange(n, dtype=float)
    for num in range(1, TOTAL_NUMBERS + 1):
        series = np.zeros(n)
        for idx in range(n):
            row = recent.iloc[idx]
            for i in range(1, NUMBERS_PER_DRAW + 1):
                val = row.get(f"n{i}")
                if pd.notna(val) and int(val) == num:
                    series[idx] = 1; break
        series = series[::-1]
        scores[num] = np.polyfit(x, series, 1)[0]
    min_s = scores[1:].min()
    scores[1:] -= min_s
    mx = scores[1:].max()
    if mx > 0: scores[1:] /= mx
    return scores

def feedback_scores(fb):
    scores = np.zeros(TOTAL_NUMBERS + 1)
    for num in range(1, TOTAL_NUMBERS + 1):
        scores[num] = (fb.get(str(num), 0.0) + 1.0) / 2.0
    return scores

# ── Phase 2: Triplet hotspots ──────────────────
def triplet_hotspots(df, top_n=10) -> list:
    """Returns top_n most common number triplets across all draws."""
    counter = Counter()
    for idx in range(len(df)):
        row  = df.iloc[idx]
        nums = []
        for i in range(1, NUMBERS_PER_DRAW + 1):
            val = row.get(f"n{i}")
            if pd.notna(val):
                v = int(val)
                if 1 <= v <= TOTAL_NUMBERS:
                    nums.append(v)
        for triplet in itertools.combinations(sorted(nums), 3):
            counter[triplet] += 1
    return counter.most_common(top_n)

def triplet_score_array(df) -> np.ndarray:
    """Score each number by how often it appears in hot triplets."""
    scores = np.zeros(TOTAL_NUMBERS + 1)
    for triplet, count in triplet_hotspots(df, top_n=50):
        for n in triplet:
            scores[n] += count
    mx = scores[1:].max()
    if mx > 0: scores[1:] /= mx
    return scores

# ── Ensemble ───────────────────────────────────
def ensemble_scores(df, half_life, w_rec, w_gap, w_trend, w_coo, w_triplet, fb):
    s_rec  = recency_scores(df, half_life)
    s_gap  = gap_scores(df)
    s_tre  = sequence_trend_scores(df, window=30)
    s_fdb  = feedback_scores(fb)
    s_tri  = triplet_score_array(df)
    co     = cooccurrence_matrix(df)

    def norm(a):
        mx = a[1:].max()
        if mx > 0:
            a = a.copy(); a[1:] /= mx
        return a

    s_rec = norm(s_rec)
    s_coo = norm(np.array([0.0] + [co[n][1:].mean() for n in range(1, TOTAL_NUMBERS+1)]))

    total_w = w_rec + w_gap + w_trend + w_coo + w_triplet
    if total_w <= 0: total_w = 1.0

    combined = (
        w_rec     / total_w * s_rec  +
        w_gap     / total_w * s_gap  +
        w_trend   / total_w * s_tre  +
        w_coo     / total_w * s_coo  +
        w_triplet / total_w * s_tri
    )
    fb_mod = 0.5 + (s_fdb - 0.5) * 0.4
    combined[1:] *= fb_mod[1:]
    return combined, co

# ── Phase 2: Monte Carlo simulation ───────────────
def monte_carlo_predictions(scores, n_runs=MONTE_CARLO_RUNS) -> pd.DataFrame:
    """
    Simulate n_runs draws by sampling WITHOUT replacement using ensemble scores as probs.
    Returns frequency table of how often each combination of 8 appears.
    Also returns the top consensus 8 numbers (most frequently drawn across all sims).
    """
    numbers   = np.arange(1, TOTAL_NUMBERS + 1)
    probs     = scores[1:].copy()
    probs     = np.clip(probs, 0, None)
    if probs.sum() <= 0: probs = np.ones(TOTAL_NUMBERS)
    probs    /= probs.sum()

    tally = np.zeros(TOTAL_NUMBERS + 1, dtype=int)
    for _ in range(n_runs):
        draw = np.random.choice(numbers, size=NUMBERS_PER_DRAW, replace=False, p=probs)
        for n in draw:
            tally[n] += 1

    mc_df = pd.DataFrame({
        "Number":    numbers,
        "SimCount":  tally[1:],
        "SimPct":    (tally[1:] / n_runs * 100).round(2),
    }).sort_values("SimCount", ascending=False).reset_index(drop=True)

    consensus = sorted(mc_df.head(NUMBERS_PER_DRAW)["Number"].tolist())
    return mc_df, consensus

# ── Phase 2: Wheeling system ────────────────────
def generate_wheel(pool: list, guarantee: int = 3) -> list:
    """
    Abbreviated wheeling: generate a minimum set of tickets from `pool`
    such that any `guarantee` numbers from the pool appear together
    in at least one ticket.
    Uses a greedy cover approach — practical for pool sizes ≤ 12.
    Returns list of tickets (each = list of NUMBERS_PER_DRAW numbers).
    """
    pool    = sorted(pool)
    needed  = list(itertools.combinations(pool, guarantee))
    covered = set()
    tickets = []

    while len(covered) < len(needed):
        # pick the combination of NUMBERS_PER_DRAW numbers that covers most uncovered triplets
        candidates = list(itertools.combinations(pool, NUMBERS_PER_DRAW))
        random.shuffle(candidates)
        best_ticket, best_cover = None, set()
        for cand in candidates:
            cand_set = set(cand)
            new_cover = {t for t in needed if t not in covered and set(t).issubset(cand_set)}
            if len(new_cover) > len(best_cover):
                best_cover  = new_cover
                best_ticket = cand
            if len(best_cover) == len(needed) - len(covered):
                break
        if best_ticket is None:
            break
        tickets.append(list(best_ticket))
        covered |= best_cover

    return tickets

# ── Prediction generation ──────────────────────
def generate_predictions(scores, co, count):
    numbers    = np.arange(1, TOTAL_NUMBERS + 1)
    base_probs = np.clip(scores[1:], 0, None)
    if base_probs.sum() <= 0: base_probs = np.ones(TOTAL_NUMBERS)
    base_probs /= base_probs.sum()

    preds = [sorted(numbers[np.argsort(base_probs)[-NUMBERS_PER_DRAW:]].tolist())]

    for _ in range(1, count):
        probs     = base_probs.copy()
        chosen    = []
        available = set(range(1, TOTAL_NUMBERS + 1))
        for _ in range(NUMBERS_PER_DRAW):
            avail_idx  = np.array(sorted(available)) - 1
            avail_prob = np.clip(probs[avail_idx], 0, None)
            if avail_prob.sum() <= 0: avail_prob = np.ones(len(avail_idx))
            avail_prob /= avail_prob.sum()
            pick = int(np.random.choice(numbers[avail_idx], p=avail_prob))
            chosen.append(pick)
            available.remove(pick)
            for n in list(available):
                probs[n - 1] *= (1 + co[pick][n] * 0.5)
        preds.append(sorted(chosen))
    return preds

# ── Metrics ────────────────────────────────────
def overlap_pct(pred, actual):
    return 100.0 * len(set(pred) & set(actual)) / NUMBERS_PER_DRAW

def distance_score(pred, actual):
    if not pred or not actual: return 0.0
    dists = [min(abs(p - a) for a in actual) for p in pred]
    return max(0.0, 100.0 * (1.0 - np.mean(dists) / (TOTAL_NUMBERS / 2.0)))

def position_match(pred, actual):
    return 100.0 * sum(1 for p, a in zip(pred, actual) if p == a) / NUMBERS_PER_DRAW

def render_badge(n, hot):
    s = "display:inline-block;margin:3px;padding:6px 12px;border-radius:8px;font-size:1rem;font-weight:700;"
    if n in hot:
        return f"<span style='{s}background:#ffecec;color:#b30000;border:1px solid #ffb3b3'>{n}</span>"
    return f"<span style='{s}background:#eef6ff;color:#0a3f6b;border:1px solid #b3d4ff'>{n}</span>"

# ── Phase 2: Claude AI Analyst ─────────────────
def call_claude_analyst(df, scores, predictions, mc_consensus,
                        triplets, backtest_rows, signal_summary) -> str:
    """
    Sends a rich context prompt to Claude and returns its narrative analysis.
    """
    recent_draws = []
    for i in range(min(10, len(df))):
        row  = df.iloc[i]
        nums = [int(row[f"n{j}"]) for j in range(1, NUMBERS_PER_DRAW + 1) if pd.notna(row.get(f"n{j}"))]
        date = row["date"].strftime("%Y-%m-%d") if pd.notna(row.get("date")) else "?"
        recent_draws.append(f"  {date}: {nums}")

    top_triplets = [f"  {list(t)}: appeared {c} times" for t, c in triplets[:5]]
    bt_summary   = ""
    if backtest_rows:
        avg_ov = np.mean([r["Overlap %"] for r in backtest_rows])
        bt_summary = f"Average backtest overlap (last 20 draws): {avg_ov:.1f}%"

    prompt = f"""You are an expert lottery data analyst for Malaysia's Magnum Life lottery (pick 8 from 1–36).

You have access to the following statistical analysis of {len(df)} historical draws:

TOP ENSEMBLE SCORES (top 12 numbers):
{signal_summary}

RECENT 10 DRAWS:
{chr(10).join(recent_draws)}

TOP NUMBER TRIPLETS (most co-appearing):
{chr(10).join(top_triplets)}

STATISTICAL PREDICTION SET 1 (deterministic top-weighted):
{predictions[0]}

MONTE CARLO CONSENSUS (10,000 simulations):
{mc_consensus}

BACKTEST PERFORMANCE:
{bt_summary}

Based on all this data, provide:
1. A brief analysis of the current number landscape (2–3 sentences)
2. Your AI-reasoned recommended set of 8 numbers with justification for each
3. Two alternative sets of 8 numbers for coverage diversity
4. A confidence note (honest assessment, 2–3 sentences)
5. One key insight from the triplet or co-occurrence patterns

Be analytical, specific, and honest about the limitations of lottery prediction.
Format your response clearly with headers."""

    try:
        resp = requests.post(
            ANTHROPIC_API_URL,
            headers={"Content-Type": "application/json"},
            json={
                "model":      CLAUDE_MODEL,
                "max_tokens": 1200,
                "messages":   [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        if resp.status_code == 200:
            data = resp.json()
            return "".join(
                block.get("text", "")
                for block in data.get("content", [])
                if block.get("type") == "text"
            )
        else:
            return f"⚠️ Claude API error {resp.status_code}: {resp.text[:300]}"
    except Exception as e:
        return f"⚠️ Could not reach Claude API: {e}"

# ───────────────────────────────────────────────
# Sidebar
# ───────────────────────────────────────────────
with st.sidebar:
    st.subheader("⚙️ Data Settings")
    force_rescrape  = st.button("🔄 Force re-scrape now")
    auto_refresh_on = st.checkbox(f"⏱ Auto-refresh every {AUTOREFRESH_MINS} min", value=False)
    decay           = st.slider("Recency half-life (draws)", 10, 200, DEFAULT_HALF_LIFE, 5)
    pred_count      = st.slider("Prediction sets", 1, 10, DEFAULT_PRED_COUNT, 1)

    st.markdown("---")
    st.subheader("🔀 Signal Weights")
    w_recency  = st.slider("Recency frequency",       0, 100, DEFAULT_W_RECEN
