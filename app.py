import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import random
from datetime import datetime
from sklearn.metrics import pairwise_distances

st.set_page_config(page_title="Magnum Life Predictor", layout="wide")
st.title("🎯 Magnum Life Prediction Analyzer")

# --- Utility: Scrape data from lottolyzer ---
@st.cache_data(ttl=3600)
def scrape_past_results(pages=24):
    all_results = []
    for page in range(1, pages + 1):
        url = f"https://en.lottolyzer.com/history/malaysia/magnum-life/page/{page}/per-page/50/number-view"
        res = requests.get(url)
        if res.status_code != 200:
            continue
        soup = BeautifulSoup(res.text, "html.parser")
        rows = soup.select("tr")
        for row in rows:
            cols = [c.get_text(strip=True) for c in row.select("td")]
            if len(cols) >= 9:
                try:
                    date_text = cols[0].strip()
                    draw_date = datetime.strptime(date_text, "%d/%m/%Y").date()
                    numbers = [int(n) for n in cols[1:9] if n.isdigit()]
                    if len(numbers) == 8:
                        all_results.append({
                            "date": draw_date,
                            "numbers": numbers
                        })
                except Exception:
                    pass
    df = pd.DataFrame(all_results)
    return df

# --- Predict numbers using weighted frequency ---
def predict_numbers(df_results, top_n=3):
    all_nums = [n for row in df_results["numbers"] for n in row]
    freq = pd.Series(all_nums).value_counts()
    top_nums = list(freq.index[:25])  # use top 25 frequent numbers as pool
    predictions = [sorted(random.sample(top_nums, 8)) for _ in range(top_n)]
    return predictions, freq

# --- Accuracy calculation between actual and predicted ---
def compare_predictions(df_results, predictions):
    latest_actual = df_results.iloc[0]
    actual_numbers = latest_actual["numbers"]

    comparison = []
    for idx, pred in enumerate(predictions, 1):
        overlap = len(set(pred) & set(actual_numbers))
        distance = np.mean(pairwise_distances(
            np.array(pred).reshape(-1, 1),
            np.array(actual_numbers).reshape(-1, 1),
            metric="euclidean"
        ))
        closeness = (overlap / len(actual_numbers)) * 100
        comparison.append({
            "Prediction #": idx,
            "Predicted Numbers": pred,
            "Actual Numbers": actual_numbers,
            "Overlap (Matched)": overlap,
            "Distance": round(distance, 2),
            "Closeness %": f"{closeness:.2f}%"
        })
    return pd.DataFrame(comparison)

# --- Main Tabs ---
tab_scraper, tab_prediction, tab_compare = st.tabs([
    "📊 Past Results", 
    "🎯 Predictions", 
    "⚖️ Compare Actual vs Predicted"
])

# --- Tab 1: Past Results ---
with tab_scraper:
    st.header("📅 Past Magnum Life Results")
    with st.spinner("Scraping full history from Lottolyzer..."):
        df_results = scrape_past_results(pages=24)
    if not df_results.empty:
        st.success(f"Loaded {len(df_results)} past draws.")
        df_results_display = df_results.copy()
        df_results_display["numbers"] = df_results_display["numbers"].apply(lambda x: " ".join(map(str, x)))
        st.dataframe(df_results_display, use_container_width=True)
    else:
        st.error("Unable to fetch results from Lottolyzer.")

# --- Tab 2: Prediction ---
with tab_prediction:
    st.header("🎯 AI-Based Predictions")
    if 'df_results' not in locals() or df_results.empty:
        st.warning("No data loaded yet. Please open 'Past Results' tab first.")
    else:
        predictions, freq = predict_numbers(df_results, top_n=3)
        for i, pred in enumerate(predictions, 1):
            st.markdown(f"#### 🔹 Prediction #{i}")
            st.write(", ".join(map(str, pred)))
        st.divider()
        st.subheader("Number Frequency (Top 25)")
        st.bar_chart(freq.head(25))

# --- Tab 3: Actual vs Predicted ---
with tab_compare:
    st.header("⚖️ Actual vs Predicted (Latest Draw)")
    if 'df_results' not in locals() or df_results.empty:
        st.warning("No data loaded yet.")
    else:
        predictions, _ = predict_numbers(df_results, top_n=3)
        compare_df = compare_predictions(df_results, predictions)
        st.dataframe(compare_df, use_container_width=True)
        st.markdown("""
        - **Overlap (Matched)** → How many numbers appeared in both predicted & actual draw.  
        - **Distance** → Numerical spacing difference between predictions & actual.  
        - **Closeness %** → Percentage of accuracy relative to 8 total numbers.
        """)

st.caption("© 2025 Magnum Life Analyzer | Powered by Streamlit & Python AI")
