"""
Human Digital Twin Dashboard - Main user interface

This module provides the main dashboard interface for the Human Digital Twin Engine,
featuring a modern dark theme with real-time visualization and control panels.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import time
import logging
from typing import Dict, List, Optional, Any
import json
from pathlib import Path

from ..engine.body_simulator import HumanBodySimulator
from ..ai_models.genomics_ai import GenomicsAI
from ..ai_models.proteomics_ai import ProteomicsAI
from ..ai_models.pathway_gnn import PathwayGNN
from ..ai_models.multiomics_fusion import MultiomicsFusion
from .visualizer_3d import HumanBody3DVisualizer
from .controls import SimulationControls

logger = logging.getLogger(__name__)

class HumanDigitalTwinDashboard:
    """
    Main dashboard for the Human Digital Twin Engine.
    
    This class provides a comprehensive interface for interacting with the
    human digital twin, including real-time visualization, simulation controls,
    and AI-powered analysis tools.
    """
    
    def __init__(self):
        """Initialize the dashboard"""
        self.body_simulator = None
        self.ai_models = {}
        self.visualizer = HumanBody3DVisualizer()
        self.controls = SimulationControls()
        
        # Initialize AI models
        self._initialize_ai_models()
        
        # Dashboard state
        self.simulation_running = False
        self.simulation_data = []
        self.current_experiment = None
        
        logger.info("Initialized Human Digital Twin Dashboard")
    
    def _initialize_ai_models(self):
        """Initialize AI models"""
        try:
            self.ai_models['genomics'] = GenomicsAI()
            self.ai_models['proteomics'] = ProteomicsAI()
            self.ai_models['pathway_gnn'] = PathwayGNN()
            self.ai_models['multiomics'] = MultiomicsFusion()
            logger.info("AI models initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing AI models: {e}")
    
    def render(self):
        """Render the main dashboard"""
        # Custom CSS for dark theme
        self._render_custom_css()
        
        # Header
        self._render_header()
        
        # Sidebar
        self._render_sidebar()
        
        # Main content area
        self._render_main_content()
        
        # Footer
        self._render_footer()
    
    def _render_custom_css(self):
        """Render custom CSS for dark theme"""
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
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 5px;
        }
        .status-healthy { background-color: #00d4aa; }
        .status-warning { background-color: #ffa500; }
        .status-critical { background-color: #ff4444; }
        </style>
        """, unsafe_allow_html=True)
    
    def _render_header(self):
        """Render the dashboard header"""
        st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <h1 class="neon-text">🧬 Human Digital Twin Engine</h1>
            <p style="color: #888; font-size: 1.2rem;">The World's First Open Source Human Digital Twin</p>
        </div>
        """, unsafe_allow_html=True)
    
    def _render_sidebar(self):
        """Render the sidebar with controls"""
        with st.sidebar:
            st.markdown("## 🎛️ Control Panel")
            
            # Simulation controls
            self._render_simulation_controls()
            
            # AI model controls
            self._render_ai_controls()
            
            # Data management
            self._render_data_management()
            
            # Export/Import
            self._render_export_import()
    
    def _render_simulation_controls(self):
        """Render simulation controls"""
        st.markdown("### 🏃‍♂️ Simulation Controls")
        
        # Initialize body simulator
        if self.body_simulator is None:
            if st.button("🚀 Initialize Human Body", key="init_body"):
                with st.spinner("Initializing human body simulator..."):
                    self.body_simulator = HumanBodySimulator(
                        age=st.session_state.get('age', 25),
                        sex=st.session_state.get('sex', 'male'),
                        weight=st.session_state.get('weight', 70),
                        height=st.session_state.get('height', 175)
                    )
                    st.success("Human body simulator initialized!")
        
        if self.body_simulator:
            # Body parameters
            st.markdown("#### Body Parameters")
            age = st.slider("Age (years)", 18, 100, 25, key="age")
            sex = st.selectbox("Sex", ["male", "female"], key="sex")
            weight = st.slider("Weight (kg)", 40, 150, 70, key="weight")
            height = st.slider("Height (cm)", 140, 220, 175, key="height")
            
            # Time controls
            st.markdown("#### Time Controls")
            time_acceleration = st.slider("Time Acceleration", 0.1, 100.0, 1.0, key="time_accel")
            if st.button("⏯️ Start Simulation", key="start_sim"):
                self.simulation_running = True
                self.body_simulator.set_time_acceleration(time_acceleration)
            
            if st.button("⏸️ Pause Simulation", key="pause_sim"):
                self.simulation_running = False
            
            if st.button("⏹️ Stop Simulation", key="stop_sim"):
                self.simulation_running = False
                self.simulation_data = []
    
    def _render_ai_controls(self):
        """Render AI model controls"""
        st.markdown("### 🤖 AI Models")
        
        # Genomics AI
        if st.button("🧬 Run Genomics Analysis", key="genomics_analysis"):
            if self.body_simulator:
                with st.spinner("Running genomics analysis..."):
                    # Mock genomics analysis
                    genomics_results = self.ai_models['genomics'].generate_genomic_report(
                        "ATGCGATCGATCGATCG", "liver"
                    )
                    st.session_state['genomics_results'] = genomics_results
                    st.success("Genomics analysis completed!")
        
        # Proteomics AI
        if st.button("🔬 Run Proteomics Analysis", key="proteomics_analysis"):
            if self.body_simulator:
                with st.spinner("Running proteomics analysis..."):
                    # Mock proteomics analysis
                    proteomics_results = self.ai_models['proteomics'].generate_protein_report(
                        "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD"
                    )
                    st.session_state['proteomics_results'] = proteomics_results
                    st.success("Proteomics analysis completed!")
        
        # Pathway GNN
        if st.button("🕸️ Run Pathway Analysis", key="pathway_analysis"):
            if self.body_simulator:
                with st.spinner("Running pathway analysis..."):
                    # Mock pathway analysis
                    pathway_results = self.ai_models['pathway_gnn'].generate_pathway_report("KEGG_00010")
                    st.session_state['pathway_results'] = pathway_results
                    st.success("Pathway analysis completed!")
    
    def _render_data_management(self):
        """Render data management controls"""
        st.markdown("### 📊 Data Management")
        
        # Load sample data
        if st.button("📥 Load Sample Data", key="load_sample"):
            with st.spinner("Loading sample data..."):
                # Mock data loading
                st.session_state['sample_data_loaded'] = True
                st.success("Sample data loaded!")
        
        # Clear data
        if st.button("🗑️ Clear All Data", key="clear_data"):
            st.session_state.clear()
            self.body_simulator = None
            st.success("All data cleared!")
    
    def _render_export_import(self):
        """Render export/import controls"""
        st.markdown("### 💾 Export/Import")
        
        # Export simulation
        if st.button("📤 Export Simulation", key="export_sim"):
            if self.body_simulator:
                simulation_state = self.body_simulator.export_state()
                st.download_button(
                    label="Download Simulation State",
                    data=json.dumps(simulation_state, indent=2),
                    file_name="simulation_state.json",
                    mime="application/json"
                )
        
        # Import simulation
        uploaded_file = st.file_uploader("Import Simulation", type=['json'], key="import_sim")
        if uploaded_file is not None:
            try:
                simulation_state = json.load(uploaded_file)
                if self.body_simulator:
                    self.body_simulator.import_state(simulation_state)
                    st.success("Simulation state imported successfully!")
            except Exception as e:
                st.error(f"Error importing simulation: {e}")
    
    def _render_main_content(self):
        """Render the main content area"""
        # Create tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🏠 Dashboard", "🧬 3D Body", "📊 Analytics", "🔬 AI Analysis", "⚗️ Experiments"
        ])
        
        with tab1:
            self._render_dashboard_tab()
        
        with tab2:
            self._render_3d_body_tab()
        
        with tab3:
            self._render_analytics_tab()
        
        with tab4:
            self._render_ai_analysis_tab()
        
        with tab5:
            self._render_experiments_tab()
    
    def _render_dashboard_tab(self):
        """Render the main dashboard tab"""
        st.markdown("## 📊 Real-time Health Dashboard")
        
        if self.body_simulator:
            # Get current health metrics
            health_summary = self.body_simulator.get_health_summary()
            
            # Health metrics cards
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="Overall Health",
                    value=f"{health_summary['overall_health_score']:.2f}",
                    delta="0.02"
                )
            
            with col2:
                st.metric(
                    label="Age",
                    value=f"{health_summary['age']:.1f} years",
                    delta="0.1"
                )
            
            with col3:
                st.metric(
                    label="Heart Rate",
                    value=f"{health_summary['vital_signs']['heart_rate']:.0f} bpm",
                    delta="2"
                )
            
            with col4:
                st.metric(
                    label="Temperature",
                    value=f"{health_summary['vital_signs']['temperature']:.1f}°C",
                    delta="0.1"
                )
            
            # Real-time charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📈 Vital Signs")
                self._render_vital_signs_chart()
            
            with col2:
                st.markdown("### 🧬 Organ Health")
                self._render_organ_health_chart()
            
            # Run simulation step if running
            if self.simulation_running:
                with st.spinner("Running simulation..."):
                    metrics = self.body_simulator.simulate_step(1.0)
                    self.simulation_data.append(metrics)
                    
                    # Limit data size
                    if len(self.simulation_data) > 100:
                        self.simulation_data = self.simulation_data[-100:]
                    
                    time.sleep(0.1)  # Small delay for visualization
        else:
            st.info("👆 Please initialize the human body simulator from the sidebar to begin.")
    
    def _render_3d_body_tab(self):
        """Render the 3D body visualization tab"""
        st.markdown("## 🧬 3D Human Body Visualization")
        
        if self.body_simulator:
            # 3D visualization
            self.visualizer.render_3d_body(self.body_simulator)
        else:
            st.info("👆 Please initialize the human body simulator to view 3D visualization.")
    
    def _render_analytics_tab(self):
        """Render the analytics tab"""
        st.markdown("## 📊 Advanced Analytics")
        
        if self.body_simulator and self.simulation_data:
            # Health trends
            st.markdown("### 📈 Health Trends")
            self._render_health_trends_chart()
            
            # Biomarker analysis
            st.markdown("### 🧪 Biomarker Analysis")
            self._render_biomarker_analysis()
            
            # Correlation analysis
            st.markdown("### 🔗 Correlation Analysis")
            self._render_correlation_analysis()
        else:
            st.info("👆 Please run a simulation to view analytics.")
    
    def _render_ai_analysis_tab(self):
        """Render the AI analysis tab"""
        st.markdown("## 🤖 AI-Powered Analysis")
        
        # Genomics results
        if 'genomics_results' in st.session_state:
            st.markdown("### 🧬 Genomics Analysis")
            genomics_results = st.session_state['genomics_results']
            
            col1, col2 = st.columns(2)
            with col1:
                st.json(genomics_results['gene_expression_prediction'])
            with col2:
                st.json(genomics_results['genomic_features'])
        
        # Proteomics results
        if 'proteomics_results' in st.session_state:
            st.markdown("### 🔬 Proteomics Analysis")
            proteomics_results = st.session_state['proteomics_results']
            
            col1, col2 = st.columns(2)
            with col1:
                st.json(proteomics_results['properties'])
            with col2:
                st.json(proteomics_results['function_prediction'])
        
        # Pathway results
        if 'pathway_results' in st.session_state:
            st.markdown("### 🕸️ Pathway Analysis")
            pathway_results = st.session_state['pathway_results']
            
            st.json(pathway_results['structure_analysis'])
    
    def _render_experiments_tab(self):
        """Render the experiments tab"""
        st.markdown("## ⚗️ Virtual Experiments")
        
        if self.body_simulator:
            # Drug intervention
            st.markdown("### 💊 Drug Interventions")
            drug_name = st.selectbox("Select Drug", ["aspirin", "metformin", "statin", "ace_inhibitor"])
            dose = st.slider("Dose (mg)", 10, 1000, 100)
            
            if st.button("Apply Drug", key="apply_drug"):
                self.body_simulator.apply_drug(drug_name, dose)
                st.success(f"Applied {dose}mg of {drug_name}")
            
            # Gene editing
            st.markdown("### ✂️ Gene Editing")
            gene_name = st.selectbox("Select Gene", ["TP53", "MYC", "BRCA1", "EGFR"])
            expression_change = st.slider("Expression Change", -1.0, 1.0, 0.0, 0.1)
            
            if st.button("Edit Gene", key="edit_gene"):
                self.body_simulator.edit_gene(gene_name, expression_change)
                st.success(f"Edited {gene_name} expression by {expression_change}")
            
            # Time acceleration experiment
            st.markdown("### ⏰ Time Acceleration")
            acceleration = st.slider("Acceleration Factor", 1.0, 100.0, 1.0)
            duration = st.slider("Duration (hours)", 1, 24, 1)
            
            if st.button("Run Time Experiment", key="time_experiment"):
                with st.spinner(f"Running {duration}h experiment at {acceleration}x speed..."):
                    self.body_simulator.set_time_acceleration(acceleration)
                    for _ in range(duration * 3600):  # Simulate duration
                        self.body_simulator.simulate_step(1.0)
                    st.success(f"Completed {duration}h experiment!")
        else:
            st.info("👆 Please initialize the human body simulator to run experiments.")
    
    def _render_vital_signs_chart(self):
        """Render vital signs chart"""
        if self.simulation_data:
            # Create DataFrame from simulation data
            df = pd.DataFrame([
                {
                    'time': i,
                    'heart_rate': data.heart_rate,
                    'temperature': data.temperature,
                    'blood_pressure': data.blood_pressure_systolic
                }
                for i, data in enumerate(self.simulation_data)
            ])
            
            # Create plot
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df['time'], y=df['heart_rate'],
                mode='lines', name='Heart Rate',
                line=dict(color='#00d4aa')
            ))
            
            fig.add_trace(go.Scatter(
                x=df['time'], y=df['temperature'],
                mode='lines', name='Temperature',
                line=dict(color='#ff6b6b'),
                yaxis='y2'
            ))
            
            fig.update_layout(
                title="Vital Signs Over Time",
                xaxis_title="Time (seconds)",
                yaxis_title="Heart Rate (bpm)",
                yaxis2=dict(title="Temperature (°C)", overlaying="y", side="right"),
                template="plotly_dark",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_organ_health_chart(self):
        """Render organ health chart"""
        if self.body_simulator:
            health_summary = self.body_simulator.get_health_summary()
            organ_health = health_summary['organ_health']
            
            # Create DataFrame
            df = pd.DataFrame([
                {'organ': organ, 'health': health}
                for organ, health in organ_health.items()
            ])
            
            # Create bar chart
            fig = px.bar(
                df, x='organ', y='health',
                title="Organ Health Scores",
                color='health',
                color_continuous_scale='RdYlGn'
            )
            
            fig.update_layout(
                template="plotly_dark",
                height=400,
                xaxis_title="Organ",
                yaxis_title="Health Score"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_health_trends_chart(self):
        """Render health trends chart"""
        if self.simulation_data:
            df = pd.DataFrame([
                {
                    'time': i,
                    'overall_health': data.cellular_health_score,
                    'epigenetic_age': data.epigenetic_age
                }
                for i, data in enumerate(self.simulation_data)
            ])
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df['time'], y=df['overall_health'],
                mode='lines', name='Overall Health',
                line=dict(color='#00d4aa')
            ))
            
            fig.add_trace(go.Scatter(
                x=df['time'], y=df['epigenetic_age'],
                mode='lines', name='Epigenetic Age',
                line=dict(color='#ff6b6b'),
                yaxis='y2'
            ))
            
            fig.update_layout(
                title="Health Trends",
                xaxis_title="Time (seconds)",
                yaxis_title="Overall Health Score",
                yaxis2=dict(title="Epigenetic Age", overlaying="y", side="right"),
                template="plotly_dark",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_biomarker_analysis(self):
        """Render biomarker analysis"""
        if self.simulation_data:
            latest_data = self.simulation_data[-1]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Glucose Level", f"{latest_data.glucose_level:.1f} mg/dL")
                st.metric("Cholesterol", f"{latest_data.cholesterol_total:.1f} mg/dL")
            
            with col2:
                st.metric("Inflammation (CRP)", f"{latest_data.inflammation_markers['CRP']:.2f}")
                st.metric("Inflammation (IL6)", f"{latest_data.inflammation_markers['IL6']:.2f}")
    
    def _render_correlation_analysis(self):
        """Render correlation analysis"""
        if self.simulation_data:
            # Create correlation matrix
            df = pd.DataFrame([
                {
                    'heart_rate': data.heart_rate,
                    'temperature': data.temperature,
                    'glucose': data.glucose_level,
                    'cholesterol': data.cholesterol_total,
                    'health_score': data.cellular_health_score
                }
                for data in self.simulation_data
            ])
            
            corr_matrix = df.corr()
            
            fig = px.imshow(
                corr_matrix,
                title="Biomarker Correlations",
                color_continuous_scale='RdBu',
                aspect='auto'
            )
            
            fig.update_layout(
                template="plotly_dark",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_footer(self):
        """Render the dashboard footer"""
        st.markdown("""
        <div style="text-align: center; padding: 2rem 0; color: #888;">
            <p>🧬 Human Digital Twin Engine - Open Source Human Biology Simulation</p>
            <p>Built with ❤️ for advancing human health and understanding</p>
        </div>
        """, unsafe_allow_html=True)