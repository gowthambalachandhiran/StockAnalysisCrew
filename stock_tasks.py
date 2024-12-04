# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 15:29:26 2024

@author: gowtham.balachan
"""

from crewai import Task

class StockTasks:
    @staticmethod
    def create_tasks(agents, stock_data):
        """Create tasks for stock research"""
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
                - Display graphs like RSI,Bollinger bands etc
                """,
                expected_output="Ultra-Concise Market Analysis (max 100 words)",
                agent=agents[0]
            ),
            Task(
                description=f"""
                Develop trading strategy for: {stock_summary}
                
                Deliver:
                - Immediate trading opportunities
                - Key risk/reward points
                - Minimal actionable advice
                - How the domain the oraganization operates on would change?
                - Investment options
                - Any gossip or rumour about the company or its future
                """,
                expected_output="Concise Trading Strategy (max 150 words)",
                agent=agents[1]
            )
        ]