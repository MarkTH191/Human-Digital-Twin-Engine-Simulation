"""
Organ Simulator - Simulates individual human organs

This module simulates the function and health of individual human organs,
including their cellular composition, metabolic processes, and responses
to interventions.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class OrganHealth(Enum):
    """Organ health states"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"

@dataclass
class OrganMetrics:
    """Metrics for organ function and health"""
    blood_flow: float
    oxygen_consumption: float
    metabolic_rate: float
    waste_clearance: float
    function_score: float
    cellular_density: float
    inflammation_level: float
    age_related_damage: float

class OrganSimulator:
    """
    Simulates the function and health of a human organ.
    
    Each organ simulator tracks cellular composition, metabolic processes,
    blood flow, and responses to various interventions and aging.
    """
    
    def __init__(self, organ_name: str, age: float, sex: str):
        """
        Initialize organ simulator.
        
        Args:
            organ_name: Name of the organ
            age: Age in years
            sex: Biological sex
        """
        self.organ_name = organ_name
        self.age = age
        self.sex = sex
        
        # Organ-specific properties
        self.properties = self._get_organ_properties()
        
        # Current metrics
        self.metrics = self._calculate_initial_metrics()
        
        # Cellular composition
        self.cell_types = self._initialize_cell_types()
        
        # Metabolic processes
        self.metabolic_pathways = self._initialize_metabolic_pathways()
        
        # Blood flow and circulation
        self.blood_flow_rate = self.properties['base_blood_flow']
        self.oxygen_saturation = 0.95
        
        # Health tracking
        self.health_history = []
        self.damage_factors = {}
        
        logger.info(f"Initialized {organ_name} simulator for {age}y {sex}")
    
    def _get_organ_properties(self) -> Dict[str, Any]:
        """Get organ-specific properties and parameters"""
        properties = {
            'heart': {
                'base_blood_flow': 5.0,  # L/min
                'oxygen_consumption': 0.3,  # L/min
                'metabolic_rate': 1.0,
                'cell_count': 2e9,
                'mass': 0.3,  # kg
                'critical_functions': ['pumping', 'electrical_conduction']
            },
            'brain': {
                'base_blood_flow': 0.75,  # L/min
                'oxygen_consumption': 0.05,  # L/min
                'metabolic_rate': 0.2,
                'cell_count': 86e9,
                'mass': 1.4,  # kg
                'critical_functions': ['neural_processing', 'memory', 'motor_control']
            },
            'liver': {
                'base_blood_flow': 1.5,  # L/min
                'oxygen_consumption': 0.1,  # L/min
                'metabolic_rate': 0.8,
                'cell_count': 2.5e11,
                'mass': 1.5,  # kg
                'critical_functions': ['detoxification', 'protein_synthesis', 'glucose_regulation']
            },
            'kidneys': {
                'base_blood_flow': 1.2,  # L/min
                'oxygen_consumption': 0.08,  # L/min
                'metabolic_rate': 0.6,
                'cell_count': 1e6,  # nephrons
                'mass': 0.3,  # kg
                'critical_functions': ['filtration', 'waste_clearance', 'electrolyte_balance']
            },
            'lungs': {
                'base_blood_flow': 5.0,  # L/min
                'oxygen_consumption': 0.0,  # L/min (produces oxygen)
                'metabolic_rate': 0.3,
                'cell_count': 3e8,
                'mass': 1.0,  # kg
                'critical_functions': ['gas_exchange', 'oxygenation', 'co2_removal']
            },
            'pancreas': {
                'base_blood_flow': 0.1,  # L/min
                'oxygen_consumption': 0.01,  # L/min
                'metabolic_rate': 0.4,
                'cell_count': 1e9,
                'mass': 0.1,  # kg
                'critical_functions': ['insulin_production', 'digestive_enzymes']
            },
            'spleen': {
                'base_blood_flow': 0.2,  # L/min
                'oxygen_consumption': 0.02,  # L/min
                'metabolic_rate': 0.2,
                'cell_count': 1e8,
                'mass': 0.2,  # kg
                'critical_functions': ['immune_function', 'blood_filtering']
            },
            'stomach': {
                'base_blood_flow': 0.3,  # L/min
                'oxygen_consumption': 0.03,  # L/min
                'metabolic_rate': 0.3,
                'cell_count': 1e8,
                'mass': 0.3,  # kg
                'critical_functions': ['digestion', 'acid_production']
            },
            'intestines': {
                'base_blood_flow': 0.8,  # L/min
                'oxygen_consumption': 0.08,  # L/min
                'metabolic_rate': 0.5,
                'cell_count': 1e9,
                'mass': 1.0,  # kg
                'critical_functions': ['nutrient_absorption', 'waste_processing']
            },
            'skin': {
                'base_blood_flow': 0.5,  # L/min
                'oxygen_consumption': 0.05,  # L/min
                'metabolic_rate': 0.2,
                'cell_count': 1e10,
                'mass': 4.0,  # kg
                'critical_functions': ['protection', 'thermoregulation', 'sensation']
            },
            'muscles': {
                'base_blood_flow': 1.0,  # L/min
                'oxygen_consumption': 0.1,  # L/min
                'metabolic_rate': 0.6,
                'cell_count': 2.5e9,
                'mass': 25.0,  # kg
                'critical_functions': ['contraction', 'movement', 'posture']
            },
            'bones': {
                'base_blood_flow': 0.2,  # L/min
                'oxygen_consumption': 0.02,  # L/min
                'metabolic_rate': 0.1,
                'cell_count': 2e9,
                'mass': 10.0,  # kg
                'critical_functions': ['support', 'protection', 'mineral_storage']
            },
            'immune_system': {
                'base_blood_flow': 0.1,  # L/min
                'oxygen_consumption': 0.01,  # L/min
                'metabolic_rate': 0.3,
                'cell_count': 1e12,
                'mass': 1.0,  # kg
                'critical_functions': ['pathogen_detection', 'immune_response', 'memory']
            },
            'endocrine_system': {
                'base_blood_flow': 0.1,  # L/min
                'oxygen_consumption': 0.01,  # L/min
                'metabolic_rate': 0.2,
                'cell_count': 1e8,
                'mass': 0.5,  # kg
                'critical_functions': ['hormone_production', 'metabolic_regulation']
            },
            'nervous_system': {
                'base_blood_flow': 0.5,  # L/min
                'oxygen_consumption': 0.05,  # L/min
                'metabolic_rate': 0.3,
                'cell_count': 1e11,
                'mass': 2.0,  # kg
                'critical_functions': ['signal_transmission', 'coordination', 'reflexes']
            }
        }
        
        return properties.get(self.organ_name, {
            'base_blood_flow': 0.1,
            'oxygen_consumption': 0.01,
            'metabolic_rate': 0.3,
            'cell_count': 1e8,
            'mass': 0.5,
            'critical_functions': ['basic_function']
        })
    
    def _calculate_initial_metrics(self) -> OrganMetrics:
        """Calculate initial organ metrics based on age and properties"""
        # Age-related decline factors
        age_factor = max(0.5, 1.0 - (self.age - 25) * 0.01)
        
        # Sex-based adjustments
        sex_factor = 1.0
        if self.sex == "female" and self.organ_name in ['heart', 'muscles']:
            sex_factor = 0.9
        elif self.sex == "male" and self.organ_name in ['bones']:
            sex_factor = 1.1
        
        return OrganMetrics(
            blood_flow=self.properties['base_blood_flow'] * age_factor * sex_factor,
            oxygen_consumption=self.properties['oxygen_consumption'] * age_factor,
            metabolic_rate=self.properties['metabolic_rate'] * age_factor,
            waste_clearance=0.9 * age_factor,
            function_score=0.95 * age_factor,
            cellular_density=1.0 * age_factor,
            inflammation_level=0.1 + (self.age - 25) * 0.005,
            age_related_damage=(self.age - 25) * 0.01
        )
    
    def _initialize_cell_types(self) -> Dict[str, Dict[str, float]]:
        """Initialize cell type composition for the organ"""
        # Simplified cell type models
        cell_types = {
            'heart': {
                'cardiomyocytes': 0.7,
                'fibroblasts': 0.2,
                'endothelial_cells': 0.1
            },
            'brain': {
                'neurons': 0.8,
                'glial_cells': 0.15,
                'endothelial_cells': 0.05
            },
            'liver': {
                'hepatocytes': 0.8,
                'kupffer_cells': 0.1,
                'stellate_cells': 0.1
            },
            'kidneys': {
                'nephron_cells': 0.9,
                'interstitial_cells': 0.1
            },
            'lungs': {
                'alveolar_cells': 0.6,
                'bronchial_cells': 0.3,
                'immune_cells': 0.1
            },
            'pancreas': {
                'beta_cells': 0.3,
                'acinar_cells': 0.6,
                'ductal_cells': 0.1
            }
        }
        
        return cell_types.get(self.organ_name, {
            'primary_cells': 0.8,
            'support_cells': 0.2
        })
    
    def _initialize_metabolic_pathways(self) -> Dict[str, float]:
        """Initialize metabolic pathway activities"""
        return {
            'glycolysis': 1.0,
            'oxidative_phosphorylation': 1.0,
            'fatty_acid_oxidation': 1.0,
            'protein_synthesis': 1.0,
            'detoxification': 1.0,
            'antioxidant_defense': 1.0
        }
    
    def simulate_step(self, dt: float, body_metrics: Any):
        """
        Simulate one time step of organ function.
        
        Args:
            dt: Time step in seconds
            body_metrics: Current body metrics
        """
        # Update blood flow based on body demand
        self._update_blood_flow(body_metrics, dt)
        
        # Update metabolic processes
        self._update_metabolism(dt)
        
        # Update cellular health
        self._update_cellular_health(dt)
        
        # Update organ metrics
        self._update_metrics(dt)
        
        # Store health history
        self.health_history.append(self.metrics.function_score)
        if len(self.health_history) > 1000:
            self.health_history = self.health_history[-1000:]
    
    def _update_blood_flow(self, body_metrics: Any, dt: float):
        """Update blood flow based on body demand and health"""
        # Base blood flow demand
        demand_factor = 1.0
        
        # Adjust based on heart rate
        if hasattr(body_metrics, 'heart_rate'):
            hr_factor = body_metrics.heart_rate / 70.0
            demand_factor *= hr_factor
        
        # Adjust based on organ health
        health_factor = self.metrics.function_score
        
        # Apply changes
        target_flow = self.properties['base_blood_flow'] * demand_factor * health_factor
        flow_change = (target_flow - self.blood_flow_rate) * 0.1
        self.blood_flow_rate = max(0.1, self.blood_flow_rate + flow_change)
    
    def _update_metabolism(self, dt: float):
        """Update metabolic pathway activities"""
        # Age-related metabolic decline
        age_factor = max(0.5, 1.0 - (self.age - 25) * 0.005)
        
        # Update each pathway
        for pathway in self.metabolic_pathways:
            # Random fluctuations
            fluctuation = np.random.normal(0, 0.01)
            self.metabolic_pathways[pathway] = max(0.1, 
                self.metabolic_pathways[pathway] + fluctuation)
            
            # Apply age factor
            self.metabolic_pathways[pathway] *= age_factor
    
    def _update_cellular_health(self, dt: float):
        """Update cellular health and composition"""
        # Age-related cellular damage
        damage_rate = 1e-6 * dt  # Very slow damage accumulation
        
        # Update cellular density
        self.metrics.cellular_density -= damage_rate
        self.metrics.cellular_density = max(0.3, self.metrics.cellular_density)
        
        # Update inflammation based on damage
        if self.metrics.age_related_damage > 0.1:
            inflammation_increase = damage_rate * 10
            self.metrics.inflammation_level += inflammation_increase
            self.metrics.inflammation_level = min(1.0, self.metrics.inflammation_level)
    
    def _update_metrics(self, dt: float):
        """Update organ metrics based on current state"""
        # Calculate function score based on multiple factors
        blood_flow_factor = self.blood_flow_rate / self.properties['base_blood_flow']
        cellular_factor = self.metrics.cellular_density
        inflammation_factor = 1.0 - self.metrics.inflammation_level
        metabolic_factor = np.mean(list(self.metabolic_pathways.values()))
        
        # Weighted function score
        self.metrics.function_score = (
            0.3 * blood_flow_factor +
            0.3 * cellular_factor +
            0.2 * inflammation_factor +
            0.2 * metabolic_factor
        )
        
        # Update other metrics
        self.metrics.blood_flow = self.blood_flow_rate
        self.metrics.oxygen_consumption = (
            self.properties['oxygen_consumption'] * 
            self.metrics.function_score
        )
        self.metrics.metabolic_rate = (
            self.properties['metabolic_rate'] * 
            self.metrics.function_score
        )
        self.metrics.waste_clearance = (
            0.9 * self.metrics.function_score
        )
    
    def apply_drug_effect(self, drug_name: str, effect_strength: float):
        """Apply drug effects to the organ"""
        if drug_name.lower() in ['aspirin', 'ibuprofen']:
            # Anti-inflammatory effect
            self.metrics.inflammation_level *= (1.0 - effect_strength * 0.5)
        elif drug_name.lower() in ['metformin'] and self.organ_name == 'pancreas':
            # Improve pancreatic function
            self.metrics.function_score += effect_strength * 0.1
        elif drug_name.lower() in ['statin'] and self.organ_name == 'liver':
            # Improve liver function
            self.metrics.function_score += effect_strength * 0.05
    
    def get_health_score(self) -> float:
        """Get overall health score of the organ"""
        return self.metrics.function_score
    
    def get_health_status(self) -> OrganHealth:
        """Get health status category"""
        score = self.get_health_score()
        if score >= 0.9:
            return OrganHealth.EXCELLENT
        elif score >= 0.8:
            return OrganHealth.GOOD
        elif score >= 0.7:
            return OrganHealth.FAIR
        elif score >= 0.6:
            return OrganHealth.POOR
        else:
            return OrganHealth.CRITICAL
    
    def export_state(self) -> Dict[str, Any]:
        """Export current organ state"""
        return {
            'organ_name': self.organ_name,
            'age': self.age,
            'sex': self.sex,
            'metrics': self.metrics.__dict__,
            'cell_types': self.cell_types,
            'metabolic_pathways': self.metabolic_pathways,
            'blood_flow_rate': self.blood_flow_rate,
            'oxygen_saturation': self.oxygen_saturation,
            'damage_factors': self.damage_factors
        }
    
    def import_state(self, state: Dict[str, Any]):
        """Import organ state"""
        self.age = state['age']
        self.sex = state['sex']
        self.cell_types = state['cell_types']
        self.metabolic_pathways = state['metabolic_pathways']
        self.blood_flow_rate = state['blood_flow_rate']
        self.oxygen_saturation = state['oxygen_saturation']
        self.damage_factors = state['damage_factors']
        
        # Reconstruct metrics
        metrics_dict = state['metrics']
        self.metrics = OrganMetrics(**metrics_dict)