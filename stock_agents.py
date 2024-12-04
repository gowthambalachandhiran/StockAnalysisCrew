# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 15:06:21 2024

@author: gowtham.balachan
"""

from crewai import Agent
from langchain_groq import ChatGroq
import os

class StockAgents:
    @staticmethod
    def create_agents(llm):
        """Create AI agents for stock research"""
        return [
            Agent(
                role="Technical indicator analyst",
                goal="Provide ultra-concise market insights",
                backstory="Expert in distilling complex market data using technical indicators and then decide trading strategy",
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