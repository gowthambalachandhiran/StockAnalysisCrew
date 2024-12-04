import os
import streamlit as st
import yfinance as yf
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq
from pathlib import Path
import pandas as pd
import litellm

# Load environment variables
env_path = Path('./config') / '.env'
load_dotenv(dotenv_path=env_path)

# Configure LiteLLM
litellm.set_verbose = False
litellm.api_key = os.getenv("GROQ_API_KEY")

# Function to fetch S&P 500 tickers
def get_sp500_tickers():
    try:
        sp500_table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        return sorted(sp500_table['Symbol'].tolist())
    except Exception as e:
        st.error(f"Error fetching S&P 500 tickers: {e}")
        return []

# Function to initialize Groq LLM
def initialize_groq_llm():
    return ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model="groq/llama-3.1-70b-versatile",
        temperature=0.7
    )

# Simplified stock data summary
def summarize_stock_data(ticker, period='1mo'):
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

# Create Agents with extremely concise roles
def create_agents(llm):
    return [
        Agent(
            role="Technical indicator analyst",
            goal="Provide ultra-concise market insights",
            backstory="Expert in distilling complex market data using technical indicators and then decide trading stratergy",
            llm=llm,
            verbose=False
        ),
        Agent(
            role="Trading Strategist",
            goal="Craft minimal, high-impact trading recommendations",
            backstory="Invest based on company profile and news about the company",
            llm=llm,
            verbose=False
        )
    ]

# Create Tasks with extremely minimal descriptions
def create_tasks(agents, stock_data):
    # Minimal stock summary string
    stock_summary = " | ".join([
        f"{stock['ticker']}(Price:${stock['current_price']},Change:{stock['price_change_pct']}%,Sector:{stock['sector']})" 
        for stock in stock_data
    ])

    return [
        Task(
            description=f"""
            Analyze these stocks: {stock_summary}
            
            Provide a hyper-concise report:
            - Key market trends
            - Cross-stock performance comparison
            - Critical insights
            """,
            expected_output="Ultra-Concise Market Analysis (max 150 words)",
            agent=agents[0]
        ),
        Task(
            description=f"""
            Develop trading strategy for: {stock_summary}
            
            Deliver:
            - Immediate trading opportunities
            - Key risk/reward points
            - Minimal actionable advice
            """,
            expected_output="Concise Trading Strategy (max 100 words)",
            agent=agents[1]
        )
    ]

# Main Streamlit App
def main():
    st.set_page_config(page_title="Lean Stock Research AI", page_icon=":chart_with_upwards_trend:", layout="wide")
    
    st.title("🚀 Lean Stock Research Assistant")
    
    # Fetch S&P 500 tickers
    sp500_tickers = get_sp500_tickers()
    
    # Sidebar for ticker selection
    st.sidebar.header("Stock Selection")
    selected_tickers = st.sidebar.multiselect(
        "Select up to 3 S&P 500 Stocks",
        sp500_tickers,
        max_selections=3
    )
    
    # Period selection
    period = st.sidebar.selectbox(
        "Select Analysis Period",
        ["1mo", "3mo", "6mo"],
        index=0
    )
    
    # Research button
    if st.sidebar.button("Perform AI Stock Research"):
        if not selected_tickers:
            st.error("Please select at least one stock!")
            return
        
        # Show loading state
        with st.spinner('Conducting AI stock research...'):
            try:
                # Initialize LLM
                llm = initialize_groq_llm()
                
                # Fetch and summarize stock data
                stock_data = [summarize_stock_data(ticker, period) for ticker in selected_tickers]
                stock_data = [data for data in stock_data if data is not None]
                
                # Create agents and tasks
                agents = create_agents(llm)
                tasks = create_tasks(agents, stock_data)
                
                # Create and run the crew
                crew = Crew(
                    agents=agents,
                    tasks=tasks,
                    verbose=False
                )
                
                # Kickoff the research
                results = crew.kickoff()
                
                # Display results
                st.header("🔍 Research Insights")
                
                # Create tabs for analyses
                tab1, tab2 = st.tabs([
                    "Market Analysis", 
                    "Trading Strategy"
                ])
                
                with tab1:
                    st.subheader("Market Trend Analysis")
                    st.write(results.tasks_output[0].raw)
                
                with tab2:
                    st.subheader("Trading Strategy")
                    st.write(results.raw)
                    st.write(results.tasks_output[0].description)
                    #st.write(results)
                    
                # Display basic stock information
                st.header("📊 Stock Details")
                for stock in stock_data:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(f"{stock['ticker']} Price", f"${stock['current_price']}")
                    with col2:
                        st.metric("Sector", stock['sector'])
                    with col3:
                        st.metric("Price Change", f"{stock['price_change_pct']}%")
            
            except Exception as e:
                st.error(f"Research error: {e}")

# Run the Streamlit app
if __name__ == "__main__":
    main()