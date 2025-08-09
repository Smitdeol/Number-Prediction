import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
from collections import Counter
import random

st.set_page_config(page_title="Magnum Life Predictor", layout="wide")
st.title("🎯 Magnum Life Prediction (Malaysia)")

st.subheader("Fetching latest results from Lottolyzer...")

# Pages to scrape (1 and 2)
base_url = "https://en.lottolyzer.com/history/malaysia/magnum-life/page/{}/per-page/50/number-view"
headers = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/115.0.0.0 Safari/537.36")
}

all_draws = []

for page in [1, 2]:
    url = base_url.format(page)
    resp = requests.get(url, headers=headers)
    if resp.status_code != 200:
        st.error(f"Failed to fetch page {page}: HTTP {resp.status_code}")
        st.stop()
    soup = BeautifulSoup(resp.text, "html.parser")
    # Each draw is a div (or li). We'll collect image alt attributes for numbers
    imgs = soup.find_all("img", alt=lambda x: x and x.isdigit())
    nums = [int(img['alt']) for img in imgs]
    # Group into draws of 8 numbers
    for i in range(0, len(nums), 8):
        draw = nums[i:i+8]
        if len(draw) == 8:
            all_draws.append(draw)

if not all_draws:
    st.error("No draws found — site structure may have changed.")
    st.stop()

df = pd.DataFrame(all_draws, columns=[f"Num {i+1}" for i in range(8)])
st.subheader("📅 Past Draws (Most Recent First)")
st.dataframe(df)

# Frequency analysis
st.subheader("🔍 Frequency Analysis")
all_numbers = [num for draw in all_draws for num in draw]
counter = Counter(all_numbers)
freq_df = pd.DataFrame(counter.items(), columns=["Number", "Frequency"]).sort_values("Frequency", ascending=False)
st.dataframe(freq_df)

# Predictions
st.subheader("🎯 Predicted Numbers (Most Frequent)")
predicted = [num for num, _ in counter.most_common(8)]
st.write(predicted)

# Random quick pick
st.subheader("🎲 Random Quick Pick")
st.write(random.sample(range(1, 37), 8))
