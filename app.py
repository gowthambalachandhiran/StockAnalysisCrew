# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 16:45:25 2024

@author: gowtham.balachan
"""

import os
import yfinance as yf
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq
from pathlib import Path
import litellm

# Provide the full path to the .env file
env_path = Path('configs') / '.env'
load_dotenv(dotenv_path=env_path)

# Configure LiteLLM to use Groq
litellm.set_verbose = False
litellm.api_key = ""

# Function to initialize Groq LLM
def initialize_groq_llm():
    return ChatGroq(
        groq_api_key="",
        model="groq/llama-3.1-70b-versatile",
        temperature=0.7
    )

# Function to fetch stock data
def fetch_stock_data(tickers, period='1mo'):
    """
    Fetch stock data for given tickers
    :param tickers: List of stock tickers
    :param period: Data retrieval period
    :return: Dictionary of stock data
    """
    stock_data = {}
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        # Fetch historical data
        hist = stock.history(period=period)
        
        # Get key statistics
        info = stock.info
        stock_data[ticker] = {
            'history': hist,
            'current_price': info.get('currentPrice', 0),
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', 0),
            'dividend_yield': info.get('dividendYield', 0)
        }
    return stock_data

# Create Agents
def create_agents(llm):
    return [
        Agent(
            role="Market Analysis Manager",
            goal="Provide comprehensive market analysis and insights",
            backstory="Experienced market analyst with expertise in identifying trends and patterns",
            llm=llm,
            verbose=True
        ),
        Agent(
            role="Day Trader",
            goal="Develop effective intraday trading strategies",
            backstory="Skilled day trader with extensive experience in technical analysis",
            llm=llm,
            verbose=True
        ),
        Agent(
            role="Portfolio Manager",
            goal="Create optimal long-term portfolio strategies",
            backstory="Seasoned portfolio manager with focus on sustainable growth",
            llm=llm,
            verbose=True
        )
    ]

# Create Tasks
def create_tasks(agents, stock_data):
    return [
        Task(
            description=f"""
            Conduct a comprehensive market analysis using the following stock data:
            {stock_data}
            
            Provide a detailed report including:
            - Overall market trends
            - Performance analysis of individual stocks
            - Macroeconomic factors influencing the market
            - Risk assessment
            """,
            expected_output="""
            Comprehensive Market Analysis Report:
            1. Market Trend Summary
            2. Stock Performance Breakdown
            3. Economic Indicators Impact
            4. Risk and Opportunity Assessment
            """,
            agent=agents[0]  # Market Analysis Manager
        ),
        Task(
            description=f"""
            Develop intraday trading strategies based on the following market data:
            {stock_data}
            
            Create a strategic trading plan that includes:
            - Short-term trading opportunities
            - Entry and exit points
            - Risk management strategies
            - Technical analysis insights
            """,
            expected_output="""
            Intraday Trading Strategy Report:
            1. Specific Trade Recommendations
            2. Technical Analysis Indicators
            3. Entry and Exit Point Guidance
            4. Risk Mitigation Strategies
            """,
            agent=agents[1]  # Day Trader
        ),
        Task(
            description=f"""
            Design a long-term portfolio strategy considering:
            {stock_data}
            
            Develop a comprehensive investment approach that includes:
            - Asset allocation recommendations
            - Diversification strategy
            - Long-term growth potential
            - Investment horizon considerations
            """,
            expected_output="""
            Long-Term Portfolio Strategy Document:
            1. Strategic Asset Allocation
            2. Diversification Approach
            3. Investment Growth Projections
            4. Risk-Adjusted Return Analysis
            """,
            agent=agents[2]  # Portfolio Manager
        )
    ]

def main():
    # Define stock tickers
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
    
    # Initialize LLM
    llm = initialize_groq_llm()
    
    # Fetch stock data
    stock_data = fetch_stock_data(tickers)
    
    # Create agents and tasks
    agents = create_agents(llm)
    tasks = create_tasks(agents, stock_data)
    
    # Create and run the crew
    crew = Crew(
        agents=agents,
        tasks=tasks,
        verbose=True
    )
    
    # Kickoff the research
    results = crew.kickoff()
    
    # Print results
    print("\n === Research Results ===")
    for i, result in enumerate(results, 1):
        print(f"\nTask {i} Result:")
        print(result)

if __name__ == "__main__":
    main()