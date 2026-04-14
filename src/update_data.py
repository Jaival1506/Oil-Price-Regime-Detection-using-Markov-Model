print("🚀 SCRIPT STARTED")
import yfinance as yf
import pandas as pd
import os

def update_brent_data():

    
    ticker = "BZ=F"

    
    df = yf.download(ticker, start="2015-01-01")

    
    df.reset_index(inplace=True)
    df = df[["Date", "Close"]]
    df.rename(columns={"Close": "Price"}, inplace=True)

    
    save_path = os.path.join("data", "brent_data.csv")
    df.to_csv(save_path, index=False)

    print("Brent data updated successfully!")


if __name__ == "__main__":
    update_brent_data()