import streamlit as st
import requests
import pandas as pd
from bs4 import BeautifulSoup
from collections import Counter

# Set page title
st.set_page_config(page_title="Magnum Life Prediction", layout="centered")

st.title("🎯 Magnum Life Number Prediction")

# Step 1: Scrape past results from Lottolyzer
st.subheader("Fetching latest results...")
url = "https://en.lottolyzer.com/history/malaysia/magnum-life/page/1/per-page/50/number-view"
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

# Extract numbers from table
numbers_list = []
for cell in soup.select("td.number"):  # matches each number cell
    num_text = cell.get_text(strip=True)
    if num_text.isdigit():
        numbers_list.append(int(num_text))

# Group into draws of 8
draws = [numbers_list[i:i+8] for i in range(0, len(numbers_list), 8)]

# Put into DataFrame
df = pd.DataFrame(draws, columns=[f"Num {i+1}" for i in range(8)])
st.subheader("📅 Past Draws")
st.dataframe(df)

# Step 2: Frequency analysis
all_numbers = [num for draw in draws for num in draw]
counter = Counter(all_numbers)
most_common = counter.most_common(8)
predicted_numbers = sorted([num for num, count in most_common])

# Step 3: Show prediction
st.subheader("🔮 Predicted Next Numbers")
st.write(predicted_numbers)

# Step 4: Show frequency table
freq_df = pd.DataFrame(counter.items(), columns=["Number", "Frequency"])
freq_df = freq_df.sort_values(by="Frequency", ascending=False).reset_index(drop=True)
st.subheader("📊 Number Frequency")
st.dataframe(freq_df)

st.success("Prediction generated successfully!")
