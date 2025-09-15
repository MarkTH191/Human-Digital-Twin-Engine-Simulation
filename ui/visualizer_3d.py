"""
3D Human Body Visualizer - Interactive 3D visualization of the human body

This module provides 3D visualization capabilities for the human body,
including organs, tissues, cells, and molecular structures.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Any, Tuple
import json

logger = logging.getLogger(__name__)

class HumanBody3DVisualizer:
    """
    3D visualizer for the human body and its components.
    
    This class provides interactive 3D visualization of the human body,
    including organs, tissues, cells, and molecular structures with
    zoom capabilities from body level to molecular level.
    """
    
    def __init__(self):
        """Initialize the 3D visualizer"""
        self.current_zoom_level = "body"  # body, organ, tissue, cell, molecule
        self.selected_organ = None
        self.selected_tissue = None
        self.selected_cell = None
        
        # 3D body model data
        self.body_model = self._create_body_model()
        self.organ_models = self._create_organ_models()
        self.tissue_models = self._create_tissue_models()
        self.cell_models = self._create_cell_models()
        self.molecule_models = self._create_molecule_models()
        
        logger.info("Initialized 3D Human Body Visualizer")
    
    def _create_body_model(self) -> Dict[str, Any]:
        """Create 3D model of the human body"""
        return {
            'organs': {
                'heart': {'position': [0, 0, 0], 'size': [2, 2, 2], 'color': '#ff6b6b'},
                'brain': {'position': [0, 0, 8], 'size': [3, 3, 3], 'color': '#4ecdc4'},
                'liver': {'position': [3, 0, 0], 'size': [4, 2, 2], 'color': '#45b7d1'},
                'lungs': {'position': [-2, 0, 2], 'size': [2, 3, 2], 'color': '#96ceb4'},
                'kidneys': {'position': [2, 0, -2], 'size': [1, 1, 1], 'color': '#feca57'},
                'pancreas': {'position': [1, 0, -1], 'size': [1, 0.5, 0.5], 'color': '#ff9ff3'},
                'spleen': {'position': [-1, 0, 0], 'size': [1, 1, 1], 'color': '#54a0ff'},
                'stomach': {'position': [0, 0, -1], 'size': [2, 1, 1], 'color': '#5f27cd'},
                'intestines': {'position': [0, 0, -3], 'size': [3, 0.5, 0.5], 'color': '#00d2d3'},
                'skin': {'position': [0, 0, 0], 'size': [10, 10, 10], 'color': '#ff9f43', 'opacity': 0.1}
            },
            'skeleton': {
                'skull': {'position': [0, 0, 8], 'size': [3, 3, 3], 'color': '#f8f9fa'},
                'spine': {'position': [0, 0, 0], 'size': [0.5, 0.5, 8], 'color': '#e9ecef'},
                'ribs': {'position': [0, 0, 2], 'size': [4, 0.2, 2], 'color': '#dee2e6'},
                'pelvis': {'position': [0, 0, -4], 'size': [3, 2, 1], 'color': '#ced4da'},
                'femur': {'position': [0, 0, -6], 'size': [0.3, 0.3, 4], 'color': '#adb5bd'}
            }
        }
    
    def _create_organ_models(self) -> Dict[str, Dict[str, Any]]:
        """Create 3D models for individual organs"""
        return {
            'heart': {
                'chambers': {
                    'left_ventricle': {'position': [0, 0, 0], 'size': [1, 1, 1], 'color': '#ff6b6b'},
                    'right_ventricle': {'position': [0.5, 0, 0], 'size': [1, 1, 1], 'color': '#ff8e8e'},
                    'left_atrium': {'position': [0, 0, 0.5], 'size': [1, 1, 0.5], 'color': '#ff9f9f'},
                    'right_atrium': {'position': [0.5, 0, 0.5], 'size': [1, 1, 0.5], 'color': '#ffb3b3'}
                },
                'valves': {
                    'mitral': {'position': [0, 0, 0.25], 'size': [0.2, 0.2, 0.1], 'color': '#ff4757'},
                    'tricuspid': {'position': [0.5, 0, 0.25], 'size': [0.2, 0.2, 0.1], 'color': '#ff4757'},
                    'aortic': {'position': [0, 0, -0.5], 'size': [0.2, 0.2, 0.1], 'color': '#ff4757'},
                    'pulmonary': {'position': [0.5, 0, -0.5], 'size': [0.2, 0.2, 0.1], 'color': '#ff4757'}
                }
            },
            'brain': {
                'regions': {
                    'cerebrum': {'position': [0, 0, 0], 'size': [3, 3, 2], 'color': '#4ecdc4'},
                    'cerebellum': {'position': [0, 0, -1], 'size': [2, 2, 1], 'color': '#45b7d1'},
                    'brainstem': {'position': [0, 0, -1.5], 'size': [1, 1, 1], 'color': '#96ceb4'},
                    'hippocampus': {'position': [0, 0, 0.5], 'size': [0.5, 0.5, 0.5], 'color': '#feca57'}
                }
            },
            'liver': {
                'lobes': {
                    'right_lobe': {'position': [1, 0, 0], 'size': [2, 2, 2], 'color': '#45b7d1'},
                    'left_lobe': {'position': [-1, 0, 0], 'size': [2, 2, 2], 'color': '#5dade2'},
                    'caudate_lobe': {'position': [0, 0, 1], 'size': [1, 1, 1], 'color': '#85c1e9'},
                    'quadrate_lobe': {'position': [0, 0, -1], 'size': [1, 1, 1], 'color': '#aed6f1'}
                }
            }
        }
    
    def _create_tissue_models(self) -> Dict[str, Dict[str, Any]]:
        """Create 3D models for tissues"""
        return {
            'cardiac_muscle': {
                'fibers': [
                    {'position': [i*0.1, 0, 0], 'size': [0.05, 0.05, 1], 'color': '#ff6b6b'}
                    for i in range(20)
                ]
            },
            'nervous_tissue': {
                'neurons': [
                    {'position': [i*0.2, j*0.2, 0], 'size': [0.1, 0.1, 0.1], 'color': '#4ecdc4'}
                    for i in range(10) for j in range(10)
                ]
            },
            'epithelial_tissue': {
                'cells': [
                    {'position': [i*0.1, j*0.1, 0], 'size': [0.08, 0.08, 0.08], 'color': '#96ceb4'}
                    for i in range(15) for j in range(15)
                ]
            }
        }
    
    def _create_cell_models(self) -> Dict[str, Dict[str, Any]]:
        """Create 3D models for cells"""
        return {
            'cardiomyocyte': {
                'nucleus': {'position': [0, 0, 0], 'size': [0.3, 0.3, 0.3], 'color': '#ff6b6b'},
                'mitochondria': [
                    {'position': [i*0.1, 0, 0], 'size': [0.05, 0.05, 0.05], 'color': '#feca57'}
                    for i in range(10)
                ],
                'sarcomeres': [
                    {'position': [i*0.05, 0, 0], 'size': [0.02, 0.02, 0.5], 'color': '#ff9ff3'}
                    for i in range(20)
                ]
            },
            'neuron': {
                'soma': {'position': [0, 0, 0], 'size': [0.2, 0.2, 0.2], 'color': '#4ecdc4'},
                'dendrites': [
                    {'position': [i*0.1, 0, 0], 'size': [0.02, 0.02, 0.5], 'color': '#45b7d1'}
                    for i in range(5)
                ],
                'axon': {'position': [0, 0, 0.3], 'size': [0.01, 0.01, 1], 'color': '#96ceb4'}
            }
        }
    
    def _create_molecule_models(self) -> Dict[str, Dict[str, Any]]:
        """Create 3D models for molecules"""
        return {
            'dna': {
                'strands': [
                    {'position': [i*0.1, 0, 0], 'size': [0.05, 0.05, 0.05], 'color': '#ff6b6b'}
                    for i in range(20)
                ]
            },
            'protein': {
                'atoms': [
                    {'position': [i*0.05, j*0.05, k*0.05], 'size': [0.02, 0.02, 0.02], 'color': '#4ecdc4'}
                    for i in range(10) for j in range(10) for k in range(10)
                ]
            },
            'atp': {
                'atoms': [
                    {'position': [0, 0, 0], 'size': [0.1, 0.1, 0.1], 'color': '#feca57'},
                    {'position': [0.1, 0, 0], 'size': [0.08, 0.08, 0.08], 'color': '#ff9ff3'},
                    {'position': [0.2, 0, 0], 'size': [0.08, 0.08, 0.08], 'color': '#ff9ff3'}
                ]
            }
        }
    
    def render_3d_body(self, body_simulator):
        """Render the 3D human body visualization"""
        st.markdown("### 🧬 Interactive 3D Human Body")
        
        # Zoom level controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            zoom_level = st.selectbox(
                "Zoom Level",
                ["body", "organ", "tissue", "cell", "molecule"],
                index=["body", "organ", "tissue", "cell", "molecule"].index(self.current_zoom_level)
            )
            self.current_zoom_level = zoom_level
        
        with col2:
            if self.current_zoom_level == "organ":
                organ_options = list(self.body_model['organs'].keys())
                self.selected_organ = st.selectbox("Select Organ", organ_options)
            elif self.current_zoom_level == "tissue":
                tissue_options = list(self.tissue_models.keys())
                self.selected_tissue = st.selectbox("Select Tissue", tissue_options)
            elif self.current_zoom_level == "cell":
                cell_options = list(self.cell_models.keys())
                self.selected_cell = st.selectbox("Select Cell Type", cell_options)
        
        with col3:
            if st.button("🔄 Reset View"):
                self.current_zoom_level = "body"
                self.selected_organ = None
                self.selected_tissue = None
                self.selected_cell = None
        
        # Render 3D visualization based on zoom level
        if self.current_zoom_level == "body":
            self._render_body_view(body_simulator)
        elif self.current_zoom_level == "organ":
            self._render_organ_view(body_simulator)
        elif self.current_zoom_level == "tissue":
            self._render_tissue_view(body_simulator)
        elif self.current_zoom_level == "cell":
            self._render_cell_view(body_simulator)
        elif self.current_zoom_level == "molecule":
            self._render_molecule_view(body_simulator)
    
    def _render_body_view(self, body_simulator):
        """Render the full body view"""
        fig = go.Figure()
        
        # Add organs
        for organ_name, organ_data in self.body_model['organs'].keys():
            if organ_name in self.body_model['organs']:
                organ = self.body_model['organs'][organ_name]
                
                # Get health status from simulator
                health_score = 1.0
                if body_simulator and organ_name in body_simulator.organs:
                    health_score = body_simulator.organs[organ_name].get_health_score()
                
                # Adjust color based on health
                color = self._get_health_color(organ['color'], health_score)
                
                fig.add_trace(go.Scatter3d(
                    x=[organ['position'][0]],
                    y=[organ['position'][1]],
                    z=[organ['position'][2]],
                    mode='markers',
                    marker=dict(
                        size=organ['size'][0] * 10,
                        color=color,
                        opacity=0.8
                    ),
                    name=organ_name.title(),
                    text=f"{organ_name.title()}<br>Health: {health_score:.2f}",
                    hovertemplate="%{text}<extra></extra>"
                ))
        
        # Add skeleton
        for bone_name, bone_data in self.body_model['skeleton'].items():
            fig.add_trace(go.Scatter3d(
                x=[bone_data['position'][0]],
                y=[bone_data['position'][1]],
                z=[bone_data['position'][2]],
                mode='markers',
                marker=dict(
                    size=bone_data['size'][0] * 5,
                    color=bone_data['color'],
                    opacity=0.6
                ),
                name=bone_name.title(),
                showlegend=False
            ))
        
        fig.update_layout(
            title="3D Human Body - Click on organs to zoom in",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                bgcolor='rgba(0,0,0,0)',
                xaxis=dict(backgroundcolor='rgba(0,0,0,0)'),
                yaxis=dict(backgroundcolor='rgba(0,0,0,0)'),
                zaxis=dict(backgroundcolor='rgba(0,0,0,0)')
            ),
            template="plotly_dark",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Organ health summary
        if body_simulator:
            st.markdown("#### 🏥 Organ Health Status")
            health_summary = body_simulator.get_health_summary()
            
            col1, col2, col3, col4 = st.columns(4)
            organs = list(health_summary['organ_health'].items())
            
            for i, (organ, health) in enumerate(organs[:8]):  # Show first 8 organs
                col = [col1, col2, col3, col4][i % 4]
                with col:
                    status_color = self._get_status_color(health)
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="status-indicator status-{status_color}"></div>
                        <strong>{organ.title()}</strong><br>
                        Health: {health:.2f}
                    </div>
                    """, unsafe_allow_html=True)
    
    def _render_organ_view(self, body_simulator):
        """Render detailed organ view"""
        if not self.selected_organ:
            st.info("Please select an organ to view details.")
            return
        
        fig = go.Figure()
        
        # Get organ model
        if self.selected_organ in self.organ_models:
            organ_model = self.organ_models[self.selected_organ]
            
            # Add organ components
            for component_type, components in organ_model.items():
                if isinstance(components, dict):
                    for component_name, component_data in components.items():
                        fig.add_trace(go.Scatter3d(
                            x=[component_data['position'][0]],
                            y=[component_data['position'][1]],
                            z=[component_data['position'][2]],
                            mode='markers',
                            marker=dict(
                                size=component_data['size'][0] * 20,
                                color=component_data['color'],
                                opacity=0.8
                            ),
                            name=f"{component_name.replace('_', ' ').title()}",
                            text=f"{component_name.replace('_', ' ').title()}",
                            hovertemplate="%{text}<extra></extra>"
                        ))
        
        fig.update_layout(
            title=f"3D {self.selected_organ.title()} - Detailed View",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                bgcolor='rgba(0,0,0,0)'
            ),
            template="plotly_dark",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Organ-specific information
        if body_simulator and self.selected_organ in body_simulator.organs:
            organ_simulator = body_simulator.organs[self.selected_organ]
            health_score = organ_simulator.get_health_score()
            
            st.markdown(f"#### {self.selected_organ.title()} Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Health Score", f"{health_score:.2f}")
                st.metric("Blood Flow", f"{organ_simulator.metrics.blood_flow:.2f} L/min")
                st.metric("Oxygen Consumption", f"{organ_simulator.metrics.oxygen_consumption:.2f} L/min")
            
            with col2:
                st.metric("Metabolic Rate", f"{organ_simulator.metrics.metabolic_rate:.2f}")
                st.metric("Function Score", f"{organ_simulator.metrics.function_score:.2f}")
                st.metric("Cellular Density", f"{organ_simulator.metrics.cellular_density:.2f}")
    
    def _render_tissue_view(self, body_simulator):
        """Render tissue view"""
        if not self.selected_tissue:
            st.info("Please select a tissue type to view details.")
            return
        
        fig = go.Figure()
        
        # Get tissue model
        if self.selected_tissue in self.tissue_models:
            tissue_model = self.tissue_models[self.selected_tissue]
            
            # Add tissue components
            for component_type, components in tissue_model.items():
                if isinstance(components, list):
                    for component in components:
                        fig.add_trace(go.Scatter3d(
                            x=[component['position'][0]],
                            y=[component['position'][1]],
                            z=[component['position'][2]],
                            mode='markers',
                            marker=dict(
                                size=component['size'][0] * 50,
                                color=component['color'],
                                opacity=0.7
                            ),
                            name=component_type.title(),
                            showlegend=False
                        ))
        
        fig.update_layout(
            title=f"3D {self.selected_tissue.replace('_', ' ').title()} - Tissue View",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                bgcolor='rgba(0,0,0,0)'
            ),
            template="plotly_dark",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Tissue-specific information
        st.markdown(f"#### {self.selected_tissue.replace('_', ' ').title()} Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Elasticity", "0.7")
            st.metric("Density", "1.05 g/cm³")
            st.metric("Vascularity", "0.8")
        
        with col2:
            st.metric("Cellularity", "0.9")
            st.metric("Metabolic Rate", "0.8")
            st.metric("Repair Capacity", "0.6")
    
    def _render_cell_view(self, body_simulator):
        """Render cell view"""
        if not self.selected_cell:
            st.info("Please select a cell type to view details.")
            return
        
        fig = go.Figure()
        
        # Get cell model
        if self.selected_cell in self.cell_models:
            cell_model = self.cell_models[self.selected_cell]
            
            # Add cell components
            for component_name, component_data in cell_model.items():
                if isinstance(component_data, dict):
                    # Single component
                    fig.add_trace(go.Scatter3d(
                        x=[component_data['position'][0]],
                        y=[component_data['position'][1]],
                        z=[component_data['position'][2]],
                        mode='markers',
                        marker=dict(
                            size=component_data['size'][0] * 100,
                            color=component_data['color'],
                            opacity=0.8
                        ),
                        name=component_name.title(),
                        text=component_name.title(),
                        hovertemplate="%{text}<extra></extra>"
                    ))
                elif isinstance(component_data, list):
                    # Multiple components
                    for component in component_data:
                        fig.add_trace(go.Scatter3d(
                            x=[component['position'][0]],
                            y=[component['position'][1]],
                            z=[component['position'][2]],
                            mode='markers',
                            marker=dict(
                                size=component['size'][0] * 100,
                                color=component['color'],
                                opacity=0.7
                            ),
                            name=component_name.title(),
                            showlegend=False
                        ))
        
        fig.update_layout(
            title=f"3D {self.selected_cell.title()} - Cell View",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                bgcolor='rgba(0,0,0,0)'
            ),
            template="plotly_dark",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Cell-specific information
        st.markdown(f"#### {self.selected_cell.title()} Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Viability", "0.95")
            st.metric("Metabolic Activity", "0.8")
            st.metric("Protein Synthesis", "0.7")
        
        with col2:
            st.metric("DNA Repair", "0.6")
            st.metric("Antioxidant Capacity", "0.5")
            st.metric("Membrane Integrity", "0.8")
    
    def _render_molecule_view(self, body_simulator):
        """Render molecule view"""
        fig = go.Figure()
        
        # Add different molecule types
        for molecule_name, molecule_data in self.molecule_models.items():
            if isinstance(molecule_data, dict):
                for component_name, components in molecule_data.items():
                    if isinstance(components, list):
                        for component in components:
                            fig.add_trace(go.Scatter3d(
                                x=[component['position'][0]],
                                y=[component['position'][1]],
                                z=[component['position'][2]],
                                mode='markers',
                                marker=dict(
                                    size=component['size'][0] * 200,
                                    color=component['color'],
                                    opacity=0.8
                                ),
                                name=f"{molecule_name.upper()} - {component_name.title()}",
                                text=f"{molecule_name.upper()} - {component_name.title()}",
                                hovertemplate="%{text}<extra></extra>"
                            ))
        
        fig.update_layout(
            title="3D Molecular Structures - Molecular View",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                bgcolor='rgba(0,0,0,0)'
            ),
            template="plotly_dark",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Molecular information
        st.markdown("#### Molecular Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**DNA**")
            st.metric("Length", "3.2 billion bp")
            st.metric("Genes", "~20,000")
        
        with col2:
            st.markdown("**Proteins**")
            st.metric("Total Proteins", "~20,000")
            st.metric("Average Length", "400 aa")
        
        with col3:
            st.markdown("**Metabolites**")
            st.metric("Total Metabolites", "~5,000")
            st.metric("Pathways", "~1,000")
    
    def _get_health_color(self, base_color: str, health_score: float) -> str:
        """Get color based on health score"""
        if health_score >= 0.8:
            return base_color
        elif health_score >= 0.6:
            return "#ffa500"  # Orange
        else:
            return "#ff4444"  # Red
    
    def _get_status_color(self, health_score: float) -> str:
        """Get status color for health indicators"""
        if health_score >= 0.8:
            return "healthy"
        elif health_score >= 0.6:
            return "warning"
        else:
            return "critical"