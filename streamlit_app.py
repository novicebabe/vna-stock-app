import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import requests
import zipfile
import os
from datetime import datetime, timedelta
import ta
import matplotlib.pyplot as plt

DB_PATH = "stockvna.db"
TABLE_NAME = "vnieod"

# Function to check if database has data
def database_has_data():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='{TABLE_NAME}'")
    exists = cursor.fetchone()[0] > 0
    conn.close()
    return exists

# Function to download and process ZIP file
def download_and_extract_zip(date_str, is_initializing=False):
    url = f"https://cafef1.mediacdn.vn/data/ami_data/{date_str}/CafeF.SolieuGD.{'Upto' if is_initializing else ''}{date_str[-2:]}{date_str[4:6]}{date_str[:4]}.zip"
    zip_path = "data.zip"
    extract_folder = "extracted_data"
    
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(zip_path, "wb") as f:
            f.write(response.content)
        
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_folder)
        
        os.remove(zip_path)
        process_extracted_csvs(extract_folder)
        return True
    return False

# Function to process extracted CSVs
def process_extracted_csvs(folder_path):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            Ticker TEXT,
            TransactionDate TEXT,
            Open REAL,
            High REAL,
            Low REAL,
            Close REAL,
            Volume INTEGER,
            Exchange TEXT
        )
    """)
    
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            exchange = "UPCOM" if "UPCOM" in file else "HNX" if "HNX" in file else "HSX"
            df = pd.read_csv(os.path.join(folder_path, file))
            df.columns = [col.replace("<", "").replace(">", "") for col in df.columns]
            df.rename(columns={"DTYYYYMMDD": "TransactionDate"}, inplace=True)
            df["Exchange"] = exchange
            df.to_sql(TABLE_NAME, conn, if_exists="append", index=False)
    
    conn.commit()
    conn.close()

# RSI Calculation
def get_stocks_with_low_rsi(rsi_threshold=30, recent_days=7):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn)
    conn.close()
    
    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"].astype(str))
    recent_df = df[df["TransactionDate"] >= (datetime.today() - timedelta(days=recent_days))]
    
    results = []
    for symbol in recent_df["Ticker"].unique():
        stock_df = recent_df[recent_df["Ticker"] == symbol].sort_values("TransactionDate")
        stock_df["RSI"] = ta.momentum.RSIIndicator(stock_df["Close"], window=14).rsi()
        if stock_df["RSI"].iloc[-1] < rsi_threshold:
            results.append((symbol, stock_df["RSI"].iloc[-1]))
    
    return results

# Function to plot MACD chart
def plot_macd_chart(ticker):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME} WHERE Ticker='{ticker}' ORDER BY TransactionDate", conn)
    conn.close()
    
    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"].astype(str))
    df["MACD"] = ta.trend.MACD(df["Close"]).macd()
    df["Signal"] = ta.trend.MACD(df["Close"]).macd_signal()
    
    plt.figure(figsize=(10, 5))
    plt.plot(df["TransactionDate"], df["MACD"], label="MACD", color='blue')
    plt.plot(df["TransactionDate"], df["Signal"], label="Signal", color='red')
    plt.title(f"MACD Chart for {ticker}")
    plt.legend()
    st.pyplot(plt)

# Streamlit UI
st.title("Stock Analysis Tool")
selected_date = st.date_input("Select Date", datetime.today())
date_str = selected_date.strftime("%Y%m%d")

data_exists = database_has_data()
if data_exists:
    if st.button("Refresh Database"):
        if download_and_extract_zip(date_str, is_initializing=False):
            st.success("Database refreshed successfully!")
        else:
            st.error("Failed to refresh database.")
else:
    if st.button("Initialize Database"):
        if download_and_extract_zip(date_str, is_initializing=True):
            st.success("Database initialized successfully!")
        else:
            st.error("Failed to initialize database.")

rsi_threshold = st.slider("RSI Threshold", min_value=10, max_value=50, value=30)
recent_days = st.number_input("Most recent dates", min_value=1, max_value=30, value=7)

if st.button("Analyze RSI"):
    stocks = get_stocks_with_low_rsi(rsi_threshold=rsi_threshold, recent_days=recent_days)
    if stocks:
        st.write("Stocks with RSI below threshold:")
        stock_df = pd.DataFrame(stocks, columns=["Symbol", "RSI"])
        st.dataframe(stock_df)
        
        for ticker in stock_df["Symbol"]:
            plot_macd_chart(ticker)
    else:
        st.write("No stocks found.")
