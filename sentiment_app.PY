import streamlit as st
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

# Function to analyze sentiment
def get_sentiment(text):
    analysis = TextBlob(str(text))
    polarity = analysis.sentiment.polarity
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

# Streamlit UI
st.title("📊 AI-Driven Sentiment Analysis Tool")
st.write("Upload a CSV file with reviews, and this app will predict sentiment.")

# Upload CSV
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file,encoding='latin1')

    # Display original data
    st.subheader("📄 Original Data")
    st.write(df)

    # Check for 'Review' or 'Summary' column
    review_column = None
    if 'Review' in df.columns:
        review_column = 'Review'
    elif 'Summary' in df.columns:
        review_column = 'Summary'
    else:
        st.error("⚠️ No 'Review' or 'Summary' column found in the uploaded file.")
    
    if review_column:
        # Apply sentiment analysis
        df["Sentiment"] = df[review_column].apply(get_sentiment)

        # Show results
        st.subheader("✅ Sentiment Predictions")
        st.write(df[[review_column, "Sentiment"]])

        # Show sentiment counts
        sentiment_counts = df["Sentiment"].value_counts()
        st.subheader("📈 Sentiment Distribution")
        st.bar_chart(sentiment_counts)

        # Overall sentiment decision
        overall = sentiment_counts.idxmax()
        st.success(f"🔎 Overall Sentiment: **{overall}**")

        # Downloadable results
        csv = df.to_csv(index=False)
        st.download_button("📥 Download Results CSV", csv, "sentiment_results.csv", "text/csv")
