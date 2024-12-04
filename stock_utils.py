# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 14:54:12 2024

@author: gowtham.balachan
"""

import os
import streamlit as st
import yfinance as yf
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

class StockUtils:
    @staticmethod
    def load_environment():
        """Load environment variables from .env file"""
        env_path = Path('./config') / '.env'
        load_dotenv(dotenv_path=env_path)

    @staticmethod
    def get_sp500_tickers():
        """Fetch S&P 500 stock tickers from Wikipedia"""
        try:
            sp500_table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
            return sorted(sp500_table['Symbol'].tolist())
        except Exception as e:
            st.error(f"Error fetching S&P 500 tickers: {e}")
            return []

    @staticmethod
    def summarize_stock_data(ticker, period='1mo'):
        """Fetch and summarize stock data for a given ticker"""
        try:
            stock = yf.Ticker(ticker)
            
            # Fetch historical data and minimal summary
            hist = stock.history(period=period)
            hist_summary = {
                'ticker': ticker,
                'price_change_pct': round((hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1) * 100, 2),
            }
            
            # Get key statistics
            info = stock.info
            hist_summary.update({
                'sector': info.get('sector', 'N/A'),
                'current_price': round(info.get('currentPrice', 0), 2)
            })
            
            return hist_summary
        except Exception as e:
            st.warning(f"Could not fetch data for {ticker}: {e}")
            return None