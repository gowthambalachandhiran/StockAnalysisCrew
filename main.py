import os
import streamlit as st
import litellm
from crewai import Crew
from langchain_groq import ChatGroq

# Import custom classes
from stock_utils import StockUtils
from stock_agents import StockAgents
from stock_tasks import StockTasks

class StockResearchApp:
    def __init__(self):
        # Configure environment and LiteLLM
        StockUtils.load_environment()
        litellm.set_verbose = False
        litellm.api_key = os.getenv("GROQ_API_KEY")

    def initialize_groq_llm(self):
        """Initialize Groq Language Model"""
        return ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model="groq/llama-3.1-70b-versatile",
            temperature=0.7
        )

    def run(self):
        """Main Streamlit application"""
        st.set_page_config(page_title="Lean Stock Research AI", page_icon=":chart_with_upwards_trend:", layout="wide")
        
        st.title("🚀 Lean Stock Research Assistant")
        
        # Fetch S&P 500 tickers
        sp500_tickers = StockUtils.get_sp500_tickers()
        
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
                    llm = self.initialize_groq_llm()
                    
                    # Fetch and summarize stock data
                    stock_data = [StockUtils.summarize_stock_data(ticker, period) for ticker in selected_tickers]
                    stock_data = [data for data in stock_data if data is not None]
                    
                    # Create agents and tasks
                    agents = StockAgents.create_agents(llm)
                    tasks = StockTasks.create_tasks(agents, stock_data)
                    
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

def main():
    app = StockResearchApp()
    app.run()

if __name__ == "__main__":
    main()