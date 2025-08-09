import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from collections import Counter
import random

st.set_page_config(page_title="Magnum Life Predictor", layout="wide")
st.title("🎯 Magnum Life Prediction (Malaysia)")

# Step 1: Scrape past results from Lottolyzer
st.subheader("Fetching latest results...")

url = "https://en.lottolyzer.com/history/malaysia/magnum-life/page/1/per-page/50/number-view"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/115.0.0.0 Safari/537.36"
}

try:
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        st.error(f"Failed to fetch data. HTTP Status: {response.status_code}")
        st.stop()

    soup = BeautifulSoup(response.text, "html.parser")

    # Extract numbers from table
    numbers_list = []
    for cell in soup.select("td.number"):
        num_text = cell.get_text(strip=True)
        if num_text.isdigit():
            numbers_list.append(int(num_text))

    if not numbers_list:
        st.error("No numbers found. The website might have changed its structure.")
        st.stop()

    # Group into draws of 8
    draws = [numbers_list[i:i+8] for i in range(0, len(numbers_list), 8)]

    # Put into DataFrame
    df = pd.DataFrame(draws, columns=[f"Num {i+1}" for i in range(8)])
    st.subheader("📅 Past Draws")
    st.dataframe(df)

except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

# Step 2: Frequency Analysis
st.subheader("🔍 Frequency Analysis")
all_numbers = [num for draw in draws for num in draw]
frequency = Counter(all_numbers)
freq_df = pd.DataFrame(frequency.items(), columns=["Number", "Frequency"])
freq_df = freq_df.sort_values(by="Frequency", ascending=False)
st.dataframe(freq_df)

# Step 3: Prediction (most frequent numbers)
st.subheader("🎯 Predicted Numbers (Most Frequent)")
top_numbers = [num for num, count in frequency.most_common(8)]
st.write("Based on frequency analysis, the predicted numbers are:")
st.write(top_numbers)

# Step 4: Random Prediction (optional)
st.subheader("🎲 Random Quick Pick")
random_numbers = random.sample(range(1, 38), 8)
st.write(random_numbers)
