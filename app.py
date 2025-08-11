import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import random
from collections import Counter

# ===================
# SCRAPER FUNCTION
# ===================
def scrape_results(pages=999):
    results = []
    for page in range(1, pages + 1):
        url = f"https://en.lottolyzer.com/history/malaysia/magnum-life/page/{page}/per-page/50/number-view"
        res = requests.get(url)
        if res.status_code != 200:
            break
        soup = BeautifulSoup(res.text, "html.parser")

        rows = soup.select("table tbody tr")
        if not rows:
            break

        for row in rows:
            cols = row.find_all("td")
            if len(cols) >= 9:
                date = cols[0].text.strip()
                numbers = [c.text.strip() for c in cols[1:9]]
                results.append({
                    "date": date,
                    "numbers": numbers
                })
    return pd.DataFrame(results)

# ===================
# PREDICTION FUNCTION
# ===================
def predict_numbers(df, top_n=3):
    all_numbers = [num for sublist in df['numbers'] for num in sublist]
    counter = Counter(all_numbers)
    top_nums = [num for num, _ in counter.most_common(15)]  # take top 15 most frequent

    predictions = []
    for _ in range(top_n):
        if len(top_nums) >= 8:
            predictions.append(sorted(random.sample(top_nums, 8)))
        else:
            predictions.append(sorted(top_nums))
    return predictions

# ===================
# STREAMLIT APP
# ===================
st.set_page_config(page_title="Number Prediction", layout="wide")
st.title("🎯 Magnum Life Number Prediction")

st.write("Scraping past results... please wait.")
df_results = scrape_results()

if not df_results.empty:
    df_results['date'] = pd.to_datetime(df_results['date'], errors='coerce')
    df_results = df_results.dropna(subset=['date'])
    df_results['date'] = df_results['date'].dt.strftime('%Y-%m-%d')

    # Top 3 predictions
    preds = predict_numbers(df_results, top_n=3)
    st.subheader("🔮 Top 3 Predictions")
    for i, p in enumerate(preds, 1):
        st.write(f"Prediction {i}: {', '.join(p)}")

    # Quick Picks (Top 3 style)
    quick_picks = predict_numbers(df_results, top_n=3)
    st.subheader("🎲 Quick Picks (random)")
    for i, qp in enumerate(quick_picks, 1):
        st.write(f"Quick Pick {i}: {', '.join(qp)}")

    # Past results table
    with st.expander("📜 View Full Past Results"):
        st.dataframe(df_results, use_container_width=True)

else:
    st.error("No past results found. Please check the source website.")
