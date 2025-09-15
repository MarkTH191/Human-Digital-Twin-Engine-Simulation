"""
Tissue Simulator - Simulates human tissue types and their properties

This module simulates various human tissue types, their cellular composition,
mechanical properties, and responses to physiological changes.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class TissueType(Enum):
    """Types of human tissues"""
    EPITHELIAL = "epithelial"
    CONNECTIVE = "connective"
    MUSCLE = "muscle"
    NERVOUS = "nervous"
    BLOOD = "blood"
    ADIPOSE = "adipose"
    BONE = "bone"
    CARTILAGE = "cartilage"

@dataclass
class TissueProperties:
    """Physical and biological properties of tissue"""
    elasticity: float
    density: float
    vascularity: float
    cellularity: float
    metabolic_rate: float
    repair_capacity: float
    oxygen_consumption: float
    waste_production: float

class TissueSimulator:
    """
    Simulates the properties and behavior of human tissue types.
    
    Each tissue simulator tracks cellular composition, mechanical properties,
    metabolic activity, and responses to aging and interventions.
    """
    
    def __init__(self, tissue_name: str, age: float):
        """
        Initialize tissue simulator.
        
        Args:
            tissue_name: Name of the tissue type
            age: Age in years
        """
        self.tissue_name = tissue_name
        self.age = age
        
        # Tissue properties
        self.properties = self._get_tissue_properties()
        
        # Current state
        self.current_properties = self._calculate_initial_properties()
        
        # Cellular composition
        self.cell_composition = self._initialize_cell_composition()
        
        # Extracellular matrix
        self.ecm_composition = self._initialize_ecm_composition()
        
        # Mechanical properties
        self.mechanical_properties = self._initialize_mechanical_properties()
        
        # Metabolic state
        self.metabolic_state = self._initialize_metabolic_state()
        
        # Health tracking
        self.health_history = []
        self.damage_accumulation = 0.0
        
        logger.info(f"Initialized {tissue_name} tissue simulator for {age}y")
    
    def _get_tissue_properties(self) -> Dict[str, Any]:
        """Get tissue-specific properties and parameters"""
        properties = {
            'cardiac_muscle': {
                'tissue_type': TissueType.MUSCLE,
                'base_elasticity': 0.8,
                'base_density': 1.05,
                'base_vascularity': 0.9,
                'base_cellularity': 0.8,
                'base_metabolic_rate': 0.9,
                'repair_capacity': 0.3,
                'oxygen_consumption': 0.8,
                'waste_production': 0.7
            },
            'skeletal_muscle': {
                'tissue_type': TissueType.MUSCLE,
                'base_elasticity': 0.7,
                'base_density': 1.06,
                'base_vascularity': 0.7,
                'base_cellularity': 0.9,
                'base_metabolic_rate': 0.8,
                'repair_capacity': 0.6,
                'oxygen_consumption': 0.7,
                'waste_production': 0.6
            },
            'smooth_muscle': {
                'tissue_type': TissueType.MUSCLE,
                'base_elasticity': 0.6,
                'base_density': 1.04,
                'base_vascularity': 0.5,
                'base_cellularity': 0.7,
                'base_metabolic_rate': 0.5,
                'repair_capacity': 0.8,
                'oxygen_consumption': 0.4,
                'waste_production': 0.3
            },
            'nervous_tissue': {
                'tissue_type': TissueType.NERVOUS,
                'base_elasticity': 0.3,
                'base_density': 1.03,
                'base_vascularity': 0.8,
                'base_cellularity': 0.9,
                'base_metabolic_rate': 1.0,
                'repair_capacity': 0.1,
                'oxygen_consumption': 1.0,
                'waste_production': 0.8
            },
            'epithelial_tissue': {
                'tissue_type': TissueType.EPITHELIAL,
                'base_elasticity': 0.4,
                'base_density': 1.02,
                'base_vascularity': 0.3,
                'base_cellularity': 0.95,
                'base_metabolic_rate': 0.7,
                'repair_capacity': 0.9,
                'oxygen_consumption': 0.5,
                'waste_production': 0.4
            },
            'connective_tissue': {
                'tissue_type': TissueType.CONNECTIVE,
                'base_elasticity': 0.5,
                'base_density': 1.01,
                'base_vascularity': 0.4,
                'base_cellularity': 0.3,
                'base_metabolic_rate': 0.3,
                'repair_capacity': 0.7,
                'oxygen_consumption': 0.2,
                'waste_production': 0.2
            },
            'adipose_tissue': {
                'tissue_type': TissueType.ADIPOSE,
                'base_elasticity': 0.2,
                'base_density': 0.92,
                'base_vascularity': 0.3,
                'base_cellularity': 0.4,
                'base_metabolic_rate': 0.2,
                'repair_capacity': 0.5,
                'oxygen_consumption': 0.1,
                'waste_production': 0.1
            },
            'bone_tissue': {
                'tissue_type': TissueType.BONE,
                'base_elasticity': 0.9,
                'base_density': 1.8,
                'base_vascularity': 0.2,
                'base_cellularity': 0.1,
                'base_metabolic_rate': 0.1,
                'repair_capacity': 0.4,
                'oxygen_consumption': 0.1,
                'waste_production': 0.1
            },
            'cartilage_tissue': {
                'tissue_type': TissueType.CARTILAGE,
                'base_elasticity': 0.6,
                'base_density': 1.1,
                'base_vascularity': 0.1,
                'base_cellularity': 0.2,
                'base_metabolic_rate': 0.2,
                'repair_capacity': 0.2,
                'oxygen_consumption': 0.1,
                'waste_production': 0.1
            },
            'blood_tissue': {
                'tissue_type': TissueType.BLOOD,
                'base_elasticity': 0.1,
                'base_density': 1.05,
                'base_vascularity': 1.0,
                'base_cellularity': 0.4,
                'base_metabolic_rate': 0.6,
                'repair_capacity': 0.8,
                'oxygen_consumption': 0.0,
                'waste_production': 0.3
            }
        }
        
        return properties.get(self.tissue_name, {
            'tissue_type': TissueType.CONNECTIVE,
            'base_elasticity': 0.5,
            'base_density': 1.0,
            'base_vascularity': 0.5,
            'base_cellularity': 0.5,
            'base_metabolic_rate': 0.5,
            'repair_capacity': 0.5,
            'oxygen_consumption': 0.5,
            'waste_production': 0.5
        })
    
    def _calculate_initial_properties(self) -> TissueProperties:
        """Calculate initial tissue properties based on age"""
        # Age-related decline factors
        age_factor = max(0.3, 1.0 - (self.age - 25) * 0.015)
        
        return TissueProperties(
            elasticity=self.properties['base_elasticity'] * age_factor,
            density=self.properties['base_density'],
            vascularity=self.properties['base_vascularity'] * age_factor,
            cellularity=self.properties['base_cellularity'] * age_factor,
            metabolic_rate=self.properties['base_metabolic_rate'] * age_factor,
            repair_capacity=self.properties['repair_capacity'] * age_factor,
            oxygen_consumption=self.properties['oxygen_consumption'] * age_factor,
            waste_production=self.properties['waste_production'] * age_factor
        )
    
    def _initialize_cell_composition(self) -> Dict[str, float]:
        """Initialize cellular composition of the tissue"""
        compositions = {
            'cardiac_muscle': {
                'cardiomyocytes': 0.7,
                'fibroblasts': 0.2,
                'endothelial_cells': 0.1
            },
            'skeletal_muscle': {
                'myocytes': 0.8,
                'satellite_cells': 0.1,
                'fibroblasts': 0.1
            },
            'smooth_muscle': {
                'smooth_muscle_cells': 0.9,
                'fibroblasts': 0.1
            },
            'nervous_tissue': {
                'neurons': 0.6,
                'glial_cells': 0.3,
                'endothelial_cells': 0.1
            },
            'epithelial_tissue': {
                'epithelial_cells': 0.9,
                'basal_cells': 0.1
            },
            'connective_tissue': {
                'fibroblasts': 0.4,
                'collagen_fibers': 0.4,
                'ground_substance': 0.2
            },
            'adipose_tissue': {
                'adipocytes': 0.8,
                'stromal_cells': 0.2
            },
            'bone_tissue': {
                'osteocytes': 0.1,
                'osteoblasts': 0.05,
                'osteoclasts': 0.05,
                'bone_matrix': 0.8
            },
            'cartilage_tissue': {
                'chondrocytes': 0.1,
                'cartilage_matrix': 0.9
            },
            'blood_tissue': {
                'red_blood_cells': 0.45,
                'white_blood_cells': 0.01,
                'platelets': 0.01,
                'plasma': 0.53
            }
        }
        
        return compositions.get(self.tissue_name, {
            'primary_cells': 0.7,
            'support_cells': 0.3
        })
    
    def _initialize_ecm_composition(self) -> Dict[str, float]:
        """Initialize extracellular matrix composition"""
        ecm_compositions = {
            'cardiac_muscle': {
                'collagen': 0.3,
                'elastin': 0.1,
                'proteoglycans': 0.1,
                'water': 0.5
            },
            'skeletal_muscle': {
                'collagen': 0.2,
                'elastin': 0.05,
                'proteoglycans': 0.05,
                'water': 0.7
            },
            'nervous_tissue': {
                'collagen': 0.1,
                'proteoglycans': 0.2,
                'water': 0.7
            },
            'bone_tissue': {
                'collagen': 0.3,
                'hydroxyapatite': 0.6,
                'water': 0.1
            },
            'cartilage_tissue': {
                'collagen': 0.2,
                'proteoglycans': 0.3,
                'water': 0.5
            }
        }
        
        return ecm_compositions.get(self.tissue_name, {
            'collagen': 0.2,
            'water': 0.8
        })
    
    def _initialize_mechanical_properties(self) -> Dict[str, float]:
        """Initialize mechanical properties of the tissue"""
        return {
            'tensile_strength': self.properties['base_elasticity'] * 1000,  # Pa
            'compressive_strength': self.properties['base_elasticity'] * 500,  # Pa
            'shear_modulus': self.properties['base_elasticity'] * 200,  # Pa
            'poisson_ratio': 0.3,
            'fatigue_resistance': self.properties['repair_capacity'] * 1000
        }
    
    def _initialize_metabolic_state(self) -> Dict[str, float]:
        """Initialize metabolic state of the tissue"""
        return {
            'glucose_uptake': self.properties['base_metabolic_rate'],
            'oxygen_consumption': self.properties['oxygen_consumption'],
            'atp_production': self.properties['base_metabolic_rate'],
            'waste_clearance': 1.0 - self.properties['waste_production'],
            'antioxidant_capacity': self.properties['repair_capacity'],
            'inflammation_level': 0.1
        }
    
    def simulate_step(self, dt: float, body_metrics: Any):
        """
        Simulate one time step of tissue function.
        
        Args:
            dt: Time step in seconds
            body_metrics: Current body metrics
        """
        # Update metabolic processes
        self._update_metabolism(dt)
        
        # Update mechanical properties
        self._update_mechanical_properties(dt)
        
        # Update cellular composition
        self._update_cellular_composition(dt)
        
        # Update tissue properties
        self._update_tissue_properties(dt)
        
        # Store health history
        health_score = self._calculate_health_score()
        self.health_history.append(health_score)
        if len(self.health_history) > 1000:
            self.health_history = self.health_history[-1000:]
    
    def _update_metabolism(self, dt: float):
        """Update metabolic processes in the tissue"""
        # Age-related metabolic decline
        age_factor = max(0.3, 1.0 - (self.age - 25) * 0.01)
        
        # Update metabolic state
        for process in self.metabolic_state:
            if process != 'inflammation_level':
                # Random fluctuations
                fluctuation = np.random.normal(0, 0.01)
                self.metabolic_state[process] += fluctuation
                
                # Apply age factor
                self.metabolic_state[process] *= age_factor
                
                # Keep within bounds
                self.metabolic_state[process] = max(0.1, min(2.0, self.metabolic_state[process]))
        
        # Update inflammation based on damage
        if self.damage_accumulation > 0.1:
            inflammation_increase = self.damage_accumulation * 0.1
            self.metabolic_state['inflammation_level'] += inflammation_increase
            self.metabolic_state['inflammation_level'] = min(1.0, self.metabolic_state['inflammation_level'])
    
    def _update_mechanical_properties(self, dt: float):
        """Update mechanical properties based on aging and damage"""
        # Age-related mechanical decline
        age_factor = max(0.4, 1.0 - (self.age - 25) * 0.02)
        
        # Damage-related decline
        damage_factor = 1.0 - self.damage_accumulation * 0.5
        
        # Update mechanical properties
        for property_name in self.mechanical_properties:
            if property_name != 'poisson_ratio':
                self.mechanical_properties[property_name] *= age_factor * damage_factor
                self.mechanical_properties[property_name] = max(0.1, self.mechanical_properties[property_name])
    
    def _update_cellular_composition(self, dt: float):
        """Update cellular composition based on aging and health"""
        # Age-related cellular changes
        age_factor = max(0.5, 1.0 - (self.age - 25) * 0.01)
        
        # Update cell composition
        for cell_type in self.cell_composition:
            # Random fluctuations
            fluctuation = np.random.normal(0, 0.001)
            self.cell_composition[cell_type] += fluctuation
            
            # Apply age factor
            self.cell_composition[cell_type] *= age_factor
            
            # Keep within bounds
            self.cell_composition[cell_type] = max(0.0, min(1.0, self.cell_composition[cell_type]))
        
        # Normalize composition
        total = sum(self.cell_composition.values())
        if total > 0:
            for cell_type in self.cell_composition:
                self.cell_composition[cell_type] /= total
    
    def _update_tissue_properties(self, dt: float):
        """Update tissue properties based on current state"""
        # Calculate property factors
        cellularity_factor = np.mean(list(self.cell_composition.values()))
        metabolic_factor = np.mean([v for k, v in self.metabolic_state.items() if k != 'inflammation_level'])
        damage_factor = 1.0 - self.damage_accumulation
        
        # Update properties
        self.current_properties.cellularity = self.properties['base_cellularity'] * cellularity_factor
        self.current_properties.metabolic_rate = self.properties['base_metabolic_rate'] * metabolic_factor
        self.current_properties.elasticity = self.properties['base_elasticity'] * damage_factor
        self.current_properties.vascularity = self.properties['base_vascularity'] * damage_factor
        self.current_properties.repair_capacity = self.properties['repair_capacity'] * damage_factor
        self.current_properties.oxygen_consumption = self.properties['oxygen_consumption'] * metabolic_factor
        self.current_properties.waste_production = self.properties['waste_production'] * metabolic_factor
    
    def _calculate_health_score(self) -> float:
        """Calculate overall health score of the tissue"""
        # Weighted combination of various factors
        cellularity_score = self.current_properties.cellularity
        metabolic_score = self.current_properties.metabolic_rate
        mechanical_score = self.current_properties.elasticity
        inflammation_score = 1.0 - self.metabolic_state['inflammation_level']
        damage_score = 1.0 - self.damage_accumulation
        
        health_score = (
            0.25 * cellularity_score +
            0.25 * metabolic_score +
            0.2 * mechanical_score +
            0.15 * inflammation_score +
            0.15 * damage_score
        )
        
        return max(0.0, min(1.0, health_score))
    
    def apply_damage(self, damage_amount: float):
        """Apply damage to the tissue"""
        self.damage_accumulation += damage_amount
        self.damage_accumulation = min(1.0, self.damage_accumulation)
        
        # Increase inflammation
        self.metabolic_state['inflammation_level'] += damage_amount * 0.5
        self.metabolic_state['inflammation_level'] = min(1.0, self.metabolic_state['inflammation_level'])
    
    def apply_repair(self, repair_amount: float):
        """Apply repair to the tissue"""
        self.damage_accumulation -= repair_amount
        self.damage_accumulation = max(0.0, self.damage_accumulation)
        
        # Decrease inflammation
        self.metabolic_state['inflammation_level'] -= repair_amount * 0.3
        self.metabolic_state['inflammation_level'] = max(0.0, self.metabolic_state['inflammation_level'])
    
    def get_health_score(self) -> float:
        """Get overall health score of the tissue"""
        return self._calculate_health_score()
    
    def get_mechanical_properties(self) -> Dict[str, float]:
        """Get current mechanical properties"""
        return self.mechanical_properties.copy()
    
    def get_metabolic_state(self) -> Dict[str, float]:
        """Get current metabolic state"""
        return self.metabolic_state.copy()
    
    def export_state(self) -> Dict[str, Any]:
        """Export current tissue state"""
        return {
            'tissue_name': self.tissue_name,
            'age': self.age,
            'properties': self.properties,
            'current_properties': self.current_properties.__dict__,
            'cell_composition': self.cell_composition,
            'ecm_composition': self.ecm_composition,
            'mechanical_properties': self.mechanical_properties,
            'metabolic_state': self.metabolic_state,
            'damage_accumulation': self.damage_accumulation
        }
    
    def import_state(self, state: Dict[str, Any]):
        """Import tissue state"""
        self.age = state['age']
        self.properties = state['properties']
        self.cell_composition = state['cell_composition']
        self.ecm_composition = state['ecm_composition']
        self.mechanical_properties = state['mechanical_properties']
        self.metabolic_state = state['metabolic_state']
        self.damage_accumulation = state['damage_accumulation']
        
        # Reconstruct current properties
        properties_dict = state['current_properties']
        self.current_properties = TissueProperties(**properties_dict)