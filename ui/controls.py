"""
Simulation Controls - Control panel for the Human Digital Twin

This module provides control interfaces for managing simulations,
experiments, and interventions in the Human Digital Twin Engine.
"""

import streamlit as st
import logging
from typing import Dict, List, Optional, Any
import json
import time

logger = logging.getLogger(__name__)

class SimulationControls:
    """
    Control panel for simulation management.
    
    This class provides interfaces for controlling simulations,
    managing experiments, and applying interventions.
    """
    
    def __init__(self):
        """Initialize simulation controls"""
        self.active_experiments = []
        self.intervention_history = []
        
        logger.info("Initialized Simulation Controls")
    
    def render_drug_intervention_panel(self, body_simulator):
        """Render drug intervention control panel"""
        st.markdown("### 💊 Drug Interventions")
        
        # Drug selection
        drug_options = {
            "aspirin": {"dose_range": (50, 1000), "unit": "mg", "description": "Anti-inflammatory, pain relief"},
            "metformin": {"dose_range": (500, 2000), "unit": "mg", "description": "Diabetes medication"},
            "statin": {"dose_range": (10, 80), "unit": "mg", "description": "Cholesterol lowering"},
            "ace_inhibitor": {"dose_range": (2.5, 40), "unit": "mg", "description": "Blood pressure control"},
            "beta_blocker": {"dose_range": (25, 200), "unit": "mg", "description": "Heart rate control"},
            "insulin": {"dose_range": (1, 100), "unit": "units", "description": "Blood sugar control"}
        }
        
        selected_drug = st.selectbox("Select Drug", list(drug_options.keys()))
        
        if selected_drug:
            drug_info = drug_options[selected_drug]
            st.info(f"**{selected_drug.title()}**: {drug_info['description']}")
            
            # Dose selection
            min_dose, max_dose = drug_info['dose_range']
            dose = st.slider(
                f"Dose ({drug_info['unit']})",
                min_dose, max_dose, min_dose,
                key=f"dose_{selected_drug}"
            )
            
            # Duration
            duration = st.slider("Duration (hours)", 1, 24, 1, key=f"duration_{selected_drug}")
            
            # Administration method
            method = st.selectbox(
                "Administration Method",
                ["oral", "intravenous", "subcutaneous", "topical"],
                key=f"method_{selected_drug}"
            )
            
            # Apply drug
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button(f"Apply {selected_drug.title()}", key=f"apply_{selected_drug}"):
                    if body_simulator:
                        body_simulator.apply_drug(selected_drug, dose)
                        
                        # Record intervention
                        intervention = {
                            "type": "drug",
                            "name": selected_drug,
                            "dose": dose,
                            "unit": drug_info['unit'],
                            "duration": duration,
                            "method": method,
                            "timestamp": time.time()
                        }
                        self.intervention_history.append(intervention)
                        
                        st.success(f"Applied {dose} {drug_info['unit']} of {selected_drug}")
                    else:
                        st.error("Please initialize the body simulator first")
            
            with col2:
                if st.button(f"Remove {selected_drug.title()}", key=f"remove_{selected_drug}"):
                    if body_simulator and selected_drug in body_simulator.active_drugs:
                        del body_simulator.active_drugs[selected_drug]
                        st.success(f"Removed {selected_drug}")
    
    def render_gene_editing_panel(self, body_simulator):
        """Render gene editing control panel"""
        st.markdown("### ✂️ Gene Editing")
        
        # Gene selection
        gene_options = {
            "TP53": {"function": "Tumor suppressor", "effects": "DNA repair, apoptosis"},
            "MYC": {"function": "Transcription factor", "effects": "Cell growth, proliferation"},
            "BRCA1": {"function": "DNA repair", "effects": "Homologous recombination"},
            "EGFR": {"function": "Growth factor receptor", "effects": "Cell signaling, growth"},
            "KRAS": {"function": "GTPase", "effects": "Cell signaling, proliferation"},
            "APOE": {"function": "Lipoprotein", "effects": "Cholesterol transport"},
            "COMT": {"function": "Enzyme", "effects": "Neurotransmitter metabolism"},
            "CYP2D6": {"function": "Drug metabolism", "effects": "Drug processing"}
        }
        
        selected_gene = st.selectbox("Select Gene", list(gene_options.keys()))
        
        if selected_gene:
            gene_info = gene_options[selected_gene]
            st.info(f"**{selected_gene}**: {gene_info['function']} - {gene_info['effects']}")
            
            # Expression change
            expression_change = st.slider(
                "Expression Change",
                -1.0, 1.0, 0.0, 0.1,
                key=f"expression_{selected_gene}",
                help="Negative values decrease expression, positive values increase expression"
            )
            
            # Editing method
            method = st.selectbox(
                "Editing Method",
                ["CRISPR-Cas9", "TALEN", "Zinc Finger", "Base Editing", "Prime Editing"],
                key=f"method_{selected_gene}"
            )
            
            # Target cells
            target_cells = st.multiselect(
                "Target Cell Types",
                ["all", "cardiomyocytes", "neurons", "hepatocytes", "beta_cells", "t_cells"],
                default=["all"],
                key=f"targets_{selected_gene}"
            )
            
            # Apply gene editing
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button(f"Edit {selected_gene}", key=f"edit_{selected_gene}"):
                    if body_simulator:
                        body_simulator.edit_gene(selected_gene, expression_change)
                        
                        # Record intervention
                        intervention = {
                            "type": "gene_editing",
                            "gene": selected_gene,
                            "expression_change": expression_change,
                            "method": method,
                            "target_cells": target_cells,
                            "timestamp": time.time()
                        }
                        self.intervention_history.append(intervention)
                        
                        st.success(f"Edited {selected_gene} expression by {expression_change}")
                    else:
                        st.error("Please initialize the body simulator first")
            
            with col2:
                if st.button(f"Reset {selected_gene}", key=f"reset_{selected_gene}"):
                    if body_simulator:
                        body_simulator.edit_gene(selected_gene, -expression_change)
                        st.success(f"Reset {selected_gene} expression")
    
    def render_time_control_panel(self, body_simulator):
        """Render time control panel"""
        st.markdown("### ⏰ Time Controls")
        
        # Current time
        if body_simulator:
            current_time = body_simulator.simulation_time
            st.metric("Simulation Time", f"{current_time:.1f} seconds")
            st.metric("Time Acceleration", f"{body_simulator.time_acceleration:.1f}x")
        
        # Time acceleration controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("⏯️ Play", key="play_time"):
                if body_simulator:
                    body_simulator.set_time_acceleration(1.0)
                    st.success("Simulation running at normal speed")
        
        with col2:
            if st.button("⏸️ Pause", key="pause_time"):
                if body_simulator:
                    body_simulator.set_time_acceleration(0.0)
                    st.success("Simulation paused")
        
        with col3:
            if st.button("⏹️ Stop", key="stop_time"):
                if body_simulator:
                    body_simulator.set_time_acceleration(0.0)
                    body_simulator.simulation_time = 0.0
                    st.success("Simulation stopped and reset")
        
        # Time acceleration slider
        acceleration = st.slider(
            "Time Acceleration",
            0.0, 100.0, 1.0, 0.1,
            key="time_acceleration",
            help="1.0 = real time, 10.0 = 10x faster, 0.0 = paused"
        )
        
        if st.button("Set Acceleration", key="set_acceleration"):
            if body_simulator:
                body_simulator.set_time_acceleration(acceleration)
                st.success(f"Time acceleration set to {acceleration}x")
        
        # Time jump controls
        st.markdown("#### Time Jump")
        
        col1, col2 = st.columns(2)
        
        with col1:
            jump_duration = st.number_input(
                "Jump Duration (seconds)",
                min_value=1, max_value=86400, value=3600,
                key="jump_duration"
            )
        
        with col2:
            if st.button("Jump Forward", key="jump_forward"):
                if body_simulator:
                    for _ in range(int(jump_duration)):
                        body_simulator.simulate_step(1.0)
                    st.success(f"Jumped forward {jump_duration} seconds")
    
    def render_experiment_panel(self, body_simulator):
        """Render experiment control panel"""
        st.markdown("### ⚗️ Virtual Experiments")
        
        # Experiment templates
        experiment_templates = {
            "aging_simulation": {
                "name": "Aging Simulation",
                "description": "Simulate 10 years of aging in 1 hour",
                "duration": 3600,
                "acceleration": 10.0,
                "interventions": []
            },
            "drug_response": {
                "name": "Drug Response Study",
                "description": "Test response to multiple drugs",
                "duration": 7200,
                "acceleration": 5.0,
                "interventions": [
                    {"type": "drug", "name": "aspirin", "dose": 100, "time": 0},
                    {"type": "drug", "name": "metformin", "dose": 500, "time": 1800}
                ]
            },
            "gene_knockout": {
                "name": "Gene Knockout Study",
                "description": "Study effects of gene knockout",
                "duration": 5400,
                "acceleration": 3.0,
                "interventions": [
                    {"type": "gene_editing", "gene": "TP53", "change": -0.8, "time": 0}
                ]
            },
            "stress_response": {
                "name": "Stress Response",
                "description": "Simulate stress and recovery",
                "duration": 1800,
                "acceleration": 2.0,
                "interventions": [
                    {"type": "stress", "intensity": 0.8, "time": 0},
                    {"type": "stress", "intensity": 0.0, "time": 900}
                ]
            }
        }
        
        # Select experiment
        selected_experiment = st.selectbox(
            "Select Experiment Template",
            list(experiment_templates.keys()),
            format_func=lambda x: experiment_templates[x]["name"]
        )
        
        if selected_experiment:
            experiment = experiment_templates[selected_experiment]
            st.info(f"**{experiment['name']}**: {experiment['description']}")
            
            # Display experiment details
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Duration", f"{experiment['duration']/3600:.1f} hours")
                st.metric("Acceleration", f"{experiment['acceleration']:.1f}x")
            
            with col2:
                st.metric("Interventions", len(experiment['interventions']))
                st.metric("Estimated Time", f"{experiment['duration']/experiment['acceleration']/60:.1f} min")
            
            # Run experiment
            if st.button("Run Experiment", key=f"run_{selected_experiment}"):
                if body_simulator:
                    self._run_experiment(body_simulator, experiment)
                else:
                    st.error("Please initialize the body simulator first")
        
        # Custom experiment
        st.markdown("#### Custom Experiment")
        
        if st.button("Create Custom Experiment", key="create_custom"):
            st.session_state['show_custom_experiment'] = True
        
        if st.session_state.get('show_custom_experiment', False):
            self._render_custom_experiment_panel(body_simulator)
    
    def _run_experiment(self, body_simulator, experiment):
        """Run a virtual experiment"""
        st.markdown(f"### Running: {experiment['name']}")
        
        # Set up experiment
        body_simulator.set_time_acceleration(experiment['acceleration'])
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Run simulation
        start_time = time.time()
        total_steps = experiment['duration']
        
        for step in range(total_steps):
            # Apply interventions at specified times
            for intervention in experiment['interventions']:
                if step == intervention['time']:
                    if intervention['type'] == 'drug':
                        body_simulator.apply_drug(intervention['name'], intervention['dose'])
                    elif intervention['type'] == 'gene_editing':
                        body_simulator.edit_gene(intervention['gene'], intervention['change'])
                    elif intervention['type'] == 'stress':
                        # Apply stress (simplified)
                        for organ in body_simulator.organs.values():
                            organ.apply_damage(intervention['intensity'] * 0.1)
            
            # Run simulation step
            body_simulator.simulate_step(1.0)
            
            # Update progress
            progress = (step + 1) / total_steps
            progress_bar.progress(progress)
            status_text.text(f"Step {step + 1}/{total_steps} - {progress*100:.1f}% complete")
            
            # Small delay for visualization
            time.sleep(0.01)
        
        # Experiment complete
        elapsed_time = time.time() - start_time
        st.success(f"Experiment completed in {elapsed_time:.1f} seconds!")
        
        # Store experiment results
        experiment_result = {
            "name": experiment['name'],
            "duration": experiment['duration'],
            "acceleration": experiment['acceleration'],
            "interventions": experiment['interventions'],
            "final_health": body_simulator.get_health_summary(),
            "timestamp": time.time()
        }
        
        self.active_experiments.append(experiment_result)
    
    def _render_custom_experiment_panel(self, body_simulator):
        """Render custom experiment creation panel"""
        st.markdown("#### Create Custom Experiment")
        
        # Experiment name
        experiment_name = st.text_input("Experiment Name", key="custom_name")
        
        # Duration and acceleration
        col1, col2 = st.columns(2)
        
        with col1:
            duration = st.number_input("Duration (seconds)", 1, 86400, 3600, key="custom_duration")
        
        with col2:
            acceleration = st.number_input("Acceleration", 0.1, 100.0, 1.0, key="custom_acceleration")
        
        # Interventions
        st.markdown("#### Interventions")
        
        interventions = []
        
        if st.button("Add Drug Intervention", key="add_drug"):
            st.session_state['add_drug_intervention'] = True
        
        if st.session_state.get('add_drug_intervention', False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                drug_name = st.selectbox("Drug", ["aspirin", "metformin", "statin"], key="custom_drug")
            
            with col2:
                dose = st.number_input("Dose", 1, 1000, 100, key="custom_dose")
            
            with col3:
                intervention_time = st.number_input("Time (seconds)", 0, duration, 0, key="custom_drug_time")
            
            if st.button("Add Drug", key="add_drug_confirm"):
                interventions.append({
                    "type": "drug",
                    "name": drug_name,
                    "dose": dose,
                    "time": intervention_time
                })
                st.session_state['add_drug_intervention'] = False
                st.success("Drug intervention added!")
        
        # Run custom experiment
        if st.button("Run Custom Experiment", key="run_custom"):
            if body_simulator and experiment_name:
                custom_experiment = {
                    "name": experiment_name,
                    "duration": duration,
                    "acceleration": acceleration,
                    "interventions": interventions
                }
                self._run_experiment(body_simulator, custom_experiment)
            else:
                st.error("Please provide experiment name and initialize body simulator")
    
    def render_intervention_history(self):
        """Render intervention history"""
        st.markdown("### 📋 Intervention History")
        
        if self.intervention_history:
            for i, intervention in enumerate(reversed(self.intervention_history[-10:])):  # Show last 10
                with st.expander(f"Intervention {len(self.intervention_history) - i}: {intervention['type'].title()}"):
                    st.json(intervention)
        else:
            st.info("No interventions recorded yet.")
    
    def render_experiment_results(self):
        """Render experiment results"""
        st.markdown("### 📊 Experiment Results")
        
        if self.active_experiments:
            for i, experiment in enumerate(reversed(self.active_experiments[-5:])):  # Show last 5
                with st.expander(f"Experiment {len(self.active_experiments) - i}: {experiment['name']}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Duration", f"{experiment['duration']/3600:.1f} hours")
                        st.metric("Acceleration", f"{experiment['acceleration']:.1f}x")
                        st.metric("Interventions", len(experiment['interventions']))
                    
                    with col2:
                        final_health = experiment['final_health']
                        st.metric("Final Health", f"{final_health['overall_health_score']:.2f}")
                        st.metric("Age", f"{final_health['age']:.1f} years")
                        st.metric("Heart Rate", f"{final_health['vital_signs']['heart_rate']:.0f} bpm")
        else:
            st.info("No experiments completed yet.")