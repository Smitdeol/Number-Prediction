import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from itertools import combinations
import random
import plotly.express as px

# -----------------------------
# CONFIG
# -----------------------------
MAGNUM_LIFE_URL = "https://www.magnum4d.my/en/results"  # official results page
TOTAL_NUMBERS = 36  # Magnum Life has numbers from 1 to 36
NUMBERS_PER_TICKET = 8  # 8 numbers per ticket

# -----------------------------
# SCRAPER
# -----------------------------
def scrape_past_results():
    st.info("Scraping latest Magnum Life results... please wait.")
    try:
        resp = requests.get(MAGNUM_LIFE_URL, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200:
            st.error("Failed to fetch results page.")
            return []

        soup = BeautifulSoup(resp.text, "lxml")

        # Find all result containers for Magnum Life
        result_blocks = soup.find_all("div", class_="game-box")
        past_results = []

        for block in result_blocks:
            title = block.find("h4")
            if title and "Magnum Life" in title.text:
                numbers = block.find_all("span", class_="number")
                draw_numbers = [int(n.text) for n in numbers if n.text.isdigit()]
                if len(draw_numbers) >= NUMBERS_PER_TICKET:
                    past_results.append(draw_numbers[:NUMBERS_PER_TICKET])

        return past_results

    except Exception as e:
        st.error(f"Error scraping: {e}")
        return []


# -----------------------------
# FREQUENCY ANALYSIS
# -----------------------------
def analyze_frequency(results):
    all_nums = [num for draw in results for num in draw]
    freq_df = pd.DataFrame({"Number": range(1, TOTAL_NUMBERS+1)})
    freq_df["Frequency"] = freq_df["Number"].apply(lambda n: all_nums.count(n))
    return freq_df.sort_values(by="Frequency", ascending=False)


# -----------------------------
# GENERATE SUGGESTED NUMBERS
# -----------------------------
def generate_suggestions(freq_df, num_sets=5):
    # Pick top hot numbers for better odds
    hot_numbers = freq_df.sort_values("Frequency", ascending=False)["Number"].tolist()
    suggestions = []
    for _ in range(num_sets):
        picks = sorted(random.sample(hot_numbers[:20], NUMBERS_PER_TICKET))
        suggestions.append(picks)
    return suggestions


# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="Magnum Life Predictor", layout="wide")
st.title("🎯 Magnum Life Predictor (Malaysia)")

if st.button("Fetch & Analyze Latest Results"):
    past_results = scrape_past_results()

    if not past_results:
        st.warning("No results found. Try again later.")
    else:
        st.success(f"Fetched {len(past_results)} past draws!")

        # Show past results table
        st.subheader("Past Results")
        df_results = pd.DataFrame(past_results, columns=[f"Num{i+1}" for i in range(NUMBERS_PER_TICKET)])
        st.dataframe(df_results)

        # Analyze frequency
        st.subheader("Number Frequency Analysis")
        freq_df = analyze_frequency(past_results)
        fig = px.bar(freq_df, x="Number", y="Frequency", title="Number Frequencies")
        st.plotly_chart(fig)

        # Generate suggestions
        st.subheader("Suggested Numbers to Play")
        suggestions = generate_suggestions(freq_df, num_sets=7)
        for i, s in enumerate(suggestions, 1):
            st.write(f"Set {i}: {', '.join(map(str, s))}")

st.caption("Disclaimer: This tool is for educational purposes only and does not guarantee winnings.")
