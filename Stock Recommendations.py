import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pytz
import warnings

warnings.filterwarnings("ignore")

# Function to calculate technical indicators
def calculate_indicators(df):
    df['SMA20'] = df['Price'].rolling(window=20).mean()
    df['EMA20'] = df['Price'].ewm(span=20, adjust=False).mean()
    
    delta = df['Price'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df['EMA12'] = df['Price'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Price'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    df['Recommendation'] = df.apply(get_recommendation, axis=1)

    return df

# Function to generate buy/sell/hold signals
def get_recommendation(row):
    signals = ["Buy" if row['RSI'] < 30 or row['MACD'] > row['Signal_Line'] else 
               "Sell" if row['RSI'] > 70 or row['MACD'] < row['Signal_Line'] else "Hold"]
    return signals[0]

# Function to plot charts
def plot_charts(df, ticker):
    fig, axs = plt.subplots(3, 1, figsize=(14, 12))

    # Price and Moving Averages
    axs[0].plot(df.index, df['Price'], label='Price', color='black')
    axs[0].plot(df.index, df['SMA20'], label='SMA20', color='blue')
    axs[0].plot(df.index, df['EMA20'], label='EMA20', color='green')
    axs[0].set_title(f'{ticker} Price & Moving Averages')
    axs[0].legend()

    # RSI Indicator
    axs[1].plot(df.index, df['RSI'], label='RSI', color='purple')
    axs[1].axhline(30, linestyle='--', color='red')
    axs[1].axhline(70, linestyle='--', color='red')
    axs[1].set_title('RSI Indicator')
    axs[1].legend()

    # MACD Indicator
    axs[2].plot(df.index, df['MACD'], label='MACD', color='orange')
    axs[2].plot(df.index, df['Signal_Line'], label='Signal Line', color='blue')
    axs[2].axhline(0, linestyle='--', color='gray')
    axs[2].set_title('MACD Indicator')
    axs[2].legend()

    plt.tight_layout()
    st.pyplot(fig)  # ✅ Ensure plots render in Streamlit

# Streamlit UI setup
st.title("📊NSE Stock Analysis Tool")

symbol = st.text_input("Enter NSE stock symbol (Nifty Stocks) (e.g., RELIANCE.NS, Infy.ns)", "").upper()
interval = st.selectbox("Select Computational Interval (In Minuts/Hours/Day)", ["1m", "5m", "15m", "1h", "1d"])
period = st.selectbox("Select Period - To copute with Historical Data (In Days/Months)", ["1d", "5d", "1mo", "3mo"])

if symbol:
    ticker = symbol + ".NS" if not symbol.endswith(".NS") else symbol

    st.write(f"Fetching data for **{ticker}**...")
    
    try:
        df = yf.download(tickers=ticker, interval=interval, period=period, progress=False)
        if df.empty:
            st.warning("⚠️ No data fetched. Please check the stock symbol, interval, or period.")
        else:
            # Convert timestamps to Indian Time
            if interval == "1d":
                df.index = pd.to_datetime(df.index).tz_localize('UTC').tz_convert('Asia/Kolkata')
            else:
                df.index = df.index.tz_convert('Asia/Kolkata')

            # Select relevant columns
            df = df[['Close', 'Volume']]
            df.columns = ['Price', 'Volume']
            df = calculate_indicators(df)

            st.write("### 📌 Latest Data Preview")
            st.dataframe(df.tail())

            # Plot the charts
            plot_charts(df, ticker)

            # Download button for CSV
            st.download_button("Download Data as CSV", df.to_csv(index=True), file_name="stock_data.csv", mime="text/csv")

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
