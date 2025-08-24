import streamlit as st
import yfinance as yf
import pandas as pd


st.title("Stock Price Analyzer")

# Input 

col1, col2, col3= st.columns(3)

with col1:
    # Taking input from user
    ticker = st.text_input("Enter the stock ticker here", "MSFT")


with col2:
    start_date = st.date_input("Start date", value=pd.to_datetime("2024-01-01"))


with col3:
    end_date = st.date_input("End date", value=pd.to_datetime("today"))

# Fetch ticker data
ticker_data = yf.Ticker(ticker)
ticker_df = ticker_data.history(period="1y")

# Company info
info = ticker_data.info
st.subheader(f"Name: {info.get('shortName', 'N/A')}")
st.subheader(f"Sector: {info.get('sector', 'N/A')}")


# Display dataframe
st.dataframe(ticker_df)

# Price chart
st.subheader("Price over time")



# Fetch data for the given date range
data = ticker_data.history(start=start_date, end=end_date)

# Calculate 20-day moving average
data["20_Day_MA"] = data["Close"].rolling(window=20).mean()
data["50_Day_MA"] = data["Close"].rolling(window=50).mean()
data["100_Day_MA"] = data["Close"].rolling(window=100).mean()
data["200_Day_MA"] = data["Close"].rolling(window=200).mean()



# Show chart with Close and 20-day MA
st.line_chart(data[["Close", "20_Day_MA","50_Day_MA","100_Day_MA"]], color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])


# Volume chart
st.subheader("Volume over time")
st.bar_chart(ticker_df["Volume"])


