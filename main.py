#!/usr/bin/env python3
"""
Human Digital Twin Engine - Main Entry Point
The World's First Open Source Human Digital Twin Engine

This is the main entry point for the Human Digital Twin Engine.
Run with: streamlit run main.py
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from ui.dashboard import HumanDigitalTwinDashboard
from engine.body_simulator import HumanBodySimulator
from data.preprocess import DataPreprocessor
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point for the Human Digital Twin Engine"""
    
    # Configure Streamlit page
    st.set_page_config(
        page_title="Human Digital Twin Engine",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for dark theme
    st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stApp {
        background-color: #0e1117;
    }
    .stSidebar {
        background-color: #1e1e1e;
    }
    .stSelectbox > div > div {
        background-color: #2d2d2d;
        color: #fafafa;
    }
    .stButton > button {
        background-color: #00d4aa;
        color: #000000;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #00b894;
        color: #000000;
    }
    .metric-card {
        background-color: #1e1e1e;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #00d4aa;
        margin: 0.5rem 0;
    }
    .neon-text {
        color: #00d4aa;
        text-shadow: 0 0 10px #00d4aa;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize the dashboard
    try:
        dashboard = HumanDigitalTwinDashboard()
        dashboard.render()
    except Exception as e:
        logger.error(f"Error initializing dashboard: {e}")
        st.error(f"Failed to initialize Human Digital Twin Engine: {e}")
        st.info("Please check the logs and ensure all dependencies are installed.")

if __name__ == "__main__":
    main()