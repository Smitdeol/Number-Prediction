import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
from collections import Counter
import random

st.set_page_config(page_title="Magnum Life Predictor", layout="wide")
st.title("🎯 Magnum Life Prediction (Malaysia)")

st.subheader("Fetching latest results from Magnum Life official page...")

url = "https://www.magnum4d.my/en/Magnum-Life"
headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.1 Safari/605.1.15"
}

try:
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # Extract past draws: each draw block contains numbers
    draw_blocks = soup.select("div.draw-result") or soup.select("div.results-row")
    draws = []
    for block in draw_blocks:
        nums = [int(n.get_text()) for n in block.find_all(class_="number") if n.get_text().isdigit()]
        if len(nums) >= 8:
            draws.append(nums[:8])

    if not draws:
        st.error("No past numbers found. Page structure may have changed.")
        st.stop()

    df = pd.DataFrame(draws, columns=[f"Num {i+1}" for i in range(8)])
    st.subheader("📅 Past Draws")
    st.dataframe(df)

except Exception as e:
    st.error(f"Error fetching results: {e}")
    st.stop()

# Frequency analysis
st.subheader("🔍 Frequency Analysis")
all_nums = [n for draw in draws for n in draw]
counter = Counter(all_nums)
freq_df = pd.DataFrame(counter.items(), columns=["Number", "Frequency"]).sort_values("Frequency", ascending=False)
st.dataframe(freq_df)

# Prediction — most frequent
st.subheader("🎯 Predicted Numbers (Most Frequent)")
most_common = [n for n,_ in counter.most_common(8)]
st.write(most_common)

# Add random quick pick
st.subheader("🎲 Random Quick Pick")
st.write(random.sample(range(1,37), 8))
