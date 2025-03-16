import streamlit as st
import os
import pymysql
import pandas as pd
from dotenv import load_dotenv
import zipfile
import requests
from datetime import datetime, timedelta
import ta
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator

# from streamlit_dynamic_filters import DynamicFilters
# from st_aggrid import AgGrid, GridOptionsBuilder

from sqlalchemy import create_engine

# Load environment variables
# load_dotenv()

db_config = st.secrets["mySQL"]

# Retrieve MySQL credentials from .env file
# MYSQL_HOST = os.getenv("MYSQL_HOST")
# MYSQL_PORT = int(os.getenv("MYSQL_PORT", 3306))
# MYSQL_USER = os.getenv("MYSQL_USER")
# MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
# MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")

MYSQL_HOST = db_config["MYSQL_HOST"]
MYSQL_PORT = db_config["MYSQL_PORT"]
MYSQL_USER = db_config["MYSQL_USER"]
MYSQL_PASSWORD = db_config["MYSQL_PASSWORD"]
MYSQL_DATABASE = db_config["MYSQL_DATABASE"]


TABLE_NAME='vnieod'

# Connect to MySQL
def get_connection():
    return pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE,
        ssl={"ssl": True},
    )

database_con = f'mysql+pymysql://avnadmin:{MYSQL_PASSWORD}@mysql-337fec1-stock-vna.i.aivencloud.com:10900/defaultdb'

engine = create_engine(database_con)


# Check if database has data
def check_database():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT max(TransactionDate) as transdate FROM vnieod;")
    result = cursor.fetchone()[0]
    conn.close()
    return result

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
    conn = get_connection() #sqlite3.connect(DB_PATH)
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
            df.to_sql(TABLE_NAME, engine, if_exists="append", index=False)
    
    conn.commit()
    conn.close()

# RSI Calculation
def get_stocks_with_low_rsi(rsi_threshold=30, recent_days=7, skip_future_cont=True):
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM vnieod WHERE TransactionDate >= DATE_SUB(CURDATE(), INTERVAL 365 DAY)", conn)
    conn.close()
    
    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"].astype(str))
    
    # Calculate RSI for all transaction dates first
    df["RSI"] = df.groupby("Ticker")["Close"].transform(lambda x: RSIIndicator(x, window=14).rsi())
    
    # Apply RSI filter
    filtered_df = df[df["RSI"] < rsi_threshold]
    
    # Apply recent date filter
    filtered_df = filtered_df[filtered_df["TransactionDate"] >= (datetime.today() - timedelta(days=recent_days))]
    
    filtered_df = filtered_df.sort_values(by='TransactionDate', ascending=False)
    filtered_df = filtered_df.drop_duplicates(subset=['Ticker'], keep='first')

    filtered_df= filtered_df[filtered_df['RSI'] > 0]

    if skip_future_cont == True:
        filtered_df = filtered_df[filtered_df['Ticker'].str.len() <=4]

    results = filtered_df[["Ticker", "Close", "TransactionDate", "RSI", "Exchange"]].drop_duplicates()
    results = results.sort_values(by=['TransactionDate', 'RSI'], ascending=[False, True])
    return df, results

# Function to plot MACD chart
def plot_macd_chart(ticker, exchange, df):
    # conn = get_connection() #sqlite3.connect(DB_PATH)
    # df = pd.read_sql(f"SELECT * FROM {TABLE_NAME} WHERE Ticker='{ticker}' ORDER BY TransactionDate", conn)
    # conn.close()
    df_macd = df[(df['Ticker'] == ticker ) & (df['Exchange'] == exchange )].copy()
    df_macd["TransactionDate"] = pd.to_datetime(df_macd["TransactionDate"].astype(str))
    df_macd["MACD"] = ta.trend.MACD(df_macd["Close"]).macd()
    df_macd["Signal"] = ta.trend.MACD(df_macd["Close"]).macd_signal()
    
    #drop nan
    df_macd = df_macd.dropna(subset=['MACD', 'Signal'])
    if len(df_macd) < 20:
        return 

    plt.figure(figsize=(10, 5))
    plt.plot(df_macd["TransactionDate"], df_macd["MACD"], label="MACD", color='blue')
    plt.plot(df_macd["TransactionDate"], df_macd["Signal"], label="Signal", color='red')
    plt.axhline(y=0, color='black', linestyle='--')

    plt.title(f"MACD Chart for {ticker} - {exchange}")
    plt.legend()
    st.pyplot(plt)

# Streamlit UI
st.title("Stock Analysis Tool")
selected_date = st.date_input("Select Date", datetime.today(), format="YYYY-MM-DD")
date_str = selected_date.strftime("%Y%m%d")

# st.write("DB MYSQL_URL:", st.secrets["MYSQL_URL"])

max_date = check_database()
if max_date:
    date_diff = max_date - selected_date
    
    st.write('date diff:', date_diff.days)

    if st.button("Refresh Database"):
        while max_date <= selected_date:

            if download_and_extract_zip(date_str, is_initializing=False):
                st.success("Database refreshed successfully!")
            else:
                st.error("Failed to refresh database.")

            max_date += pd.Timedelta(days=1)

            date_str = max_date.strftime("%Y%m%d")
else:
    if st.button("Initialize Database"):
        if download_and_extract_zip(date_str, is_initializing=True):
            st.success("Database initialized successfully!")
        else:
            st.error("Failed to initialize database.")

rsi_threshold = st.slider("RSI Threshold", min_value=10, max_value=50, value=30)
recent_days = st.number_input("Most recent dates", min_value=1, max_value=30, value=7)

future = st.checkbox('Skip Future Contracs', value=True)

# is_fetched_data = False

if st.button("Analyze RSI"):
    df_365, stocks = get_stocks_with_low_rsi(rsi_threshold=rsi_threshold, recent_days=recent_days, skip_future_cont=future)
    if stocks is not None:
        is_fetched_data = True 
        
        st.write("Stocks with RSI below threshold:")

        # dynamic_filters = DynamicFilters(stocks, filters=['Ticker', 'Exchange'])
        
        # stock_df = pd.DataFrame(stocks, columns=["Symbol", "RSI", "Date"])
        # st.dataframe(stock_df)
        
        stocks.reset_index()
        st.table(stocks)

        for row in stocks.itertuples():
            ticker = row[1]
            exchange = row[5]
            df_macd = stocks[(stocks['Ticker'] == ticker) & (stocks['Exchange'] == exchange)].copy()
            
            plot_macd_chart(ticker, exchange, df_365)
        
        # gb = GridOptionsBuilder.from_dataframe(stocks)

        # gb.configure_columns(["Ticker", "Exchange"], editable=True)
        # go = gb.build()

        # ag = AgGrid(
        #     stocks, 
        #     gridOptions=go, 
        #     height=800, 
        #     fit_columns_on_grid_load=True
        # )

        # gridOptions = {
        # 'enableCellEdit': True,
        # 'enableFilter': True,
        # }        
  
    else:
        
        st.write("No stocks found.")
# else:
#     if is_fetched_data is False:
#         df_dummy = pd.DataFrame(columns=['Ticker', 'Exchange'])
#         dynamic_filters = DynamicFilters(df_dummy, filters=['Ticker', 'Exchange'])

# with st.sidebar:
#     st.write("Apply filters in any order 👇")

# dynamic_filters.display_filters(location='sidebar')
# dynamic_filters.display_df()