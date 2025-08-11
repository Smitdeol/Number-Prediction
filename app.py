import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from collections import Counter
import random
import time

# --- Function to scrape past results ---
def scrape_results():
    base_url = "https://en.lottolyzer.com/history/malaysia/magnum-life/page/{}/per-page/50/number-view"
    all_results = []

    page = 1
    while True:
        url = base_url.format(page)
        res = requests.get(url)
        if res.status_code != 200:
            break

        soup = BeautifulSoup(res.text, "html.parser")
        rows = soup.select("table tbody tr")
        if not rows:
            break

        for row in rows:
            cols = [c.get_text(strip=True) for c in row.find_all("td")]
            if len(cols) >= 9:
                date = cols[0]
                numbers = cols[1:9]
                all_results.append([date] + numbers)

        page += 1
        time.sleep(0.2)  # be nice to server

    df = pd.DataFrame(all_results, columns=["date"] + [f"num{i}" for i in range(1, 9)])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date", ascending=False).reset_index(drop=True)
    return df

# --- Prediction function ---
def predict_numbers(df, top_n=3):
    nums = df[[f"num{i}" for i in range(1, 8)]].values.flatten()
    nums = [int(n) for n in nums if n.isdigit()]
    freq = Counter(nums)
    top_nums = [n for n, _ in freq.most_common(8)]
    predictions = [sorted(random.sample(top_nums, 8)) for _ in range(top_n)]
    return predictions

# --- Quick Picks ---
def quick_picks(top_n=3):
    picks = []
    for _ in range(top_n):
        pick = sorted(random.sample(range(1, 38), 8))
        picks.append(pick)
    return picks

# --- Streamlit UI ---
st.set_page_config(page_title="Magnum Life Predictor", layout="wide")

st.title("🎯 Magnum Life Prediction (Malaysia)")
st.write("This app predicts Magnum Life numbers based on historical data.")

with st.spinner("Scraping past results..."):
    df_results = scrape_results()

# --- Predictions ---
st.subheader("🔮 Top 3 Predictions")
preds = predict_numbers(df_results, top_n=3)
for i, p in enumerate(preds, start=1):
    st.write(f"**Prediction {i}:**", ", ".join(str(x) for x in p))

# --- Quick Picks ---
st.subheader("🎲 Quick Picks (Random)")
quick = quick_picks(top_n=3)
for i, q in enumerate(quick, start=1):
    st.write(f"**Quick Pick {i}:**", ", ".join(str(x) for x in q))

# --- Past Results ---
with st.expander("📜 View Full Past Results"):
    st.dataframe(df_results)

# --- Graph ---
st.subheader("📊 Number Frequency")
all_nums = df_results[[f"num{i}" for i in range(1, 9)]].values.flatten()
all_nums = [int(n) for n in all_nums if str(n).isdigit()]
freq_df = pd.DataFrame(Counter(all_nums).items(), columns=["Number", "Frequency"]).sort_values("Frequency", ascending=False)
st.bar_chart(freq_df.set_index("Number"))

st.success(f"✅ Data loaded: {len(df_results)} draws")
