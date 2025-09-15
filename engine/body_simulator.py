"""
Human Body Simulator - Top-level simulation orchestrator

This module simulates the entire human body, coordinating all organs,
tissues, and cellular processes in a unified digital twin.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import time
import logging
from dataclasses import dataclass
from enum import Enum

from .organ_simulator import OrganSimulator
from .tissue_simulator import TissueSimulator
from .cell_simulator import CellSimulator
from .molecule_simulator import MoleculeSimulator

logger = logging.getLogger(__name__)

class BodyState(Enum):
    """Human body simulation states"""
    HEALTHY = "healthy"
    AGING = "aging"
    DISEASED = "diseased"
    DRUG_TREATED = "drug_treated"
    STRESSED = "stressed"

@dataclass
class BodyMetrics:
    """Comprehensive body health metrics"""
    age: float
    bmi: float
    heart_rate: float
    blood_pressure_systolic: float
    blood_pressure_diastolic: float
    temperature: float
    oxygen_saturation: float
    glucose_level: float
    cholesterol_total: float
    inflammation_markers: Dict[str, float]
    epigenetic_age: float
    cellular_health_score: float
    organ_function_scores: Dict[str, float]

class HumanBodySimulator:
    """
    The main human body simulator that orchestrates all biological processes.
    
    This class simulates the entire human body as a complex system of
    interacting organs, tissues, and cells, with real-time biological
    processes and responses to interventions.
    """
    
    def __init__(self, age: float = 25.0, sex: str = "male", weight: float = 70.0, height: float = 175.0):
        """
        Initialize the human body simulator.
        
        Args:
            age: Age in years
            sex: Biological sex ("male" or "female")
            weight: Weight in kg
            height: Height in cm
        """
        self.age = age
        self.sex = sex
        self.weight = weight
        self.height = height
        self.bmi = weight / ((height / 100) ** 2)
        
        # Initialize simulation state
        self.state = BodyState.HEALTHY
        self.simulation_time = 0.0  # in seconds
        self.time_acceleration = 1.0  # 1.0 = real time, 10.0 = 10x faster
        
        # Initialize organ simulators
        self.organs = self._initialize_organs()
        
        # Initialize tissue simulators
        self.tissues = self._initialize_tissues()
        
        # Initialize cell simulators
        self.cells = self._initialize_cells()
        
        # Initialize molecule simulator
        self.molecules = MoleculeSimulator()
        
        # Body metrics tracking
        self.metrics_history = []
        self.current_metrics = self._calculate_initial_metrics()
        
        # Drug and intervention tracking
        self.active_drugs = {}
        self.interventions = []
        
        logger.info(f"Initialized Human Body Simulator: {age}y, {sex}, {weight}kg, {height}cm")
    
    def _initialize_organs(self) -> Dict[str, OrganSimulator]:
        """Initialize all major organ simulators"""
        organs = {
            'heart': OrganSimulator('heart', self.age, self.sex),
            'brain': OrganSimulator('brain', self.age, self.sex),
            'liver': OrganSimulator('liver', self.age, self.sex),
            'kidneys': OrganSimulator('kidneys', self.age, self.sex),
            'lungs': OrganSimulator('lungs', self.age, self.sex),
            'pancreas': OrganSimulator('pancreas', self.age, self.sex),
            'spleen': OrganSimulator('spleen', self.age, self.sex),
            'stomach': OrganSimulator('stomach', self.age, self.sex),
            'intestines': OrganSimulator('intestines', self.age, self.sex),
            'skin': OrganSimulator('skin', self.age, self.sex),
            'muscles': OrganSimulator('muscles', self.age, self.sex),
            'bones': OrganSimulator('bones', self.age, self.sex),
            'immune_system': OrganSimulator('immune_system', self.age, self.sex),
            'endocrine_system': OrganSimulator('endocrine_system', self.age, self.sex),
            'nervous_system': OrganSimulator('nervous_system', self.age, self.sex)
        }
        return organs
    
    def _initialize_tissues(self) -> Dict[str, TissueSimulator]:
        """Initialize tissue simulators for major tissue types"""
        tissues = {
            'cardiac_muscle': TissueSimulator('cardiac_muscle', self.age),
            'skeletal_muscle': TissueSimulator('skeletal_muscle', self.age),
            'smooth_muscle': TissueSimulator('smooth_muscle', self.age),
            'nervous_tissue': TissueSimulator('nervous_tissue', self.age),
            'epithelial_tissue': TissueSimulator('epithelial_tissue', self.age),
            'connective_tissue': TissueSimulator('connective_tissue', self.age),
            'adipose_tissue': TissueSimulator('adipose_tissue', self.age),
            'bone_tissue': TissueSimulator('bone_tissue', self.age),
            'cartilage_tissue': TissueSimulator('cartilage_tissue', self.age),
            'blood_tissue': TissueSimulator('blood_tissue', self.age)
        }
        return tissues
    
    def _initialize_cells(self) -> Dict[str, CellSimulator]:
        """Initialize cell simulators for major cell types"""
        cells = {
            'cardiomyocytes': CellSimulator('cardiomyocytes', self.age),
            'neurons': CellSimulator('neurons', self.age),
            'hepatocytes': CellSimulator('hepatocytes', self.age),
            'nephron_cells': CellSimulator('nephron_cells', self.age),
            'alveolar_cells': CellSimulator('alveolar_cells', self.age),
            'beta_cells': CellSimulator('beta_cells', self.age),
            't_cells': CellSimulator('t_cells', self.age),
            'b_cells': CellSimulator('b_cells', self.age),
            'macrophages': CellSimulator('macrophages', self.age),
            'fibroblasts': CellSimulator('fibroblasts', self.age),
            'keratinocytes': CellSimulator('keratinocytes', self.age),
            'osteocytes': CellSimulator('osteocytes', self.age),
            'adipocytes': CellSimulator('adipocytes', self.age),
            'stem_cells': CellSimulator('stem_cells', self.age)
        }
        return cells
    
    def _calculate_initial_metrics(self) -> BodyMetrics:
        """Calculate initial body metrics based on age, sex, and physical parameters"""
        # Base metrics with age-related adjustments
        base_heart_rate = 70 - (self.age - 25) * 0.5
        base_systolic = 120 + (self.age - 25) * 0.5
        base_diastolic = 80 + (self.age - 25) * 0.3
        
        # Sex-based adjustments
        if self.sex == "female":
            base_heart_rate += 5
            base_systolic -= 5
        
        # BMI-based adjustments
        if self.bmi > 25:
            base_systolic += (self.bmi - 25) * 2
            base_diastolic += (self.bmi - 25) * 1.5
        
        return BodyMetrics(
            age=self.age,
            bmi=self.bmi,
            heart_rate=max(50, base_heart_rate),
            blood_pressure_systolic=max(90, base_systolic),
            blood_pressure_diastolic=max(60, base_diastolic),
            temperature=36.5 + np.random.normal(0, 0.2),
            oxygen_saturation=98.0 + np.random.normal(0, 1.0),
            glucose_level=90 + np.random.normal(0, 10),
            cholesterol_total=180 + (self.age - 25) * 2,
            inflammation_markers={
                'CRP': 1.0 + np.random.exponential(0.5),
                'IL6': 2.0 + np.random.exponential(1.0),
                'TNF_alpha': 1.5 + np.random.exponential(0.8)
            },
            epigenetic_age=self.age + np.random.normal(0, 2),
            cellular_health_score=0.85 + np.random.normal(0, 0.1),
            organ_function_scores={organ: 0.9 + np.random.normal(0, 0.05) for organ in self.organs.keys()}
        )
    
    def simulate_step(self, dt: float = 1.0) -> BodyMetrics:
        """
        Simulate one time step of the human body.
        
        Args:
            dt: Time step in seconds
            
        Returns:
            Updated body metrics
        """
        # Apply time acceleration
        actual_dt = dt * self.time_acceleration
        self.simulation_time += actual_dt
        
        # Update all organ simulators
        for organ_name, organ in self.organs.items():
            organ.simulate_step(actual_dt, self.current_metrics)
        
        # Update all tissue simulators
        for tissue_name, tissue in self.tissues.items():
            tissue.simulate_step(actual_dt, self.current_metrics)
        
        # Update all cell simulators
        for cell_name, cell in self.cells.items():
            cell.simulate_step(actual_dt, self.current_metrics)
        
        # Update molecule simulator
        self.molecules.simulate_step(actual_dt, self.current_metrics)
        
        # Update body metrics based on organ/tissue/cell states
        self.current_metrics = self._update_metrics(actual_dt)
        
        # Store metrics history
        self.metrics_history.append(self.current_metrics)
        
        # Limit history size
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
        
        return self.current_metrics
    
    def _update_metrics(self, dt: float) -> BodyMetrics:
        """Update body metrics based on current organ/tissue/cell states"""
        # Calculate organ function scores
        organ_scores = {}
        for organ_name, organ in self.organs.items():
            organ_scores[organ_name] = organ.get_health_score()
        
        # Calculate average organ health
        avg_organ_health = np.mean(list(organ_scores.values()))
        
        # Update heart rate based on organ health and age
        heart_rate_change = (avg_organ_health - 0.9) * 10
        new_heart_rate = max(50, self.current_metrics.heart_rate + heart_rate_change)
        
        # Update blood pressure based on age and health
        age_factor = (self.age - 25) * 0.5
        health_factor = (avg_organ_health - 0.9) * 20
        new_systolic = max(90, 120 + age_factor - health_factor)
        new_diastolic = max(60, 80 + age_factor * 0.6 - health_factor * 0.6)
        
        # Update glucose based on pancreatic function
        pancreas_health = organ_scores.get('pancreas', 0.9)
        glucose_change = (pancreas_health - 0.9) * 20
        new_glucose = max(70, self.current_metrics.glucose_level + glucose_change)
        
        # Update inflammation markers based on immune system health
        immune_health = organ_scores.get('immune_system', 0.9)
        inflammation_factor = 1.0 - (immune_health - 0.9) * 0.5
        
        new_inflammation = {
            'CRP': self.current_metrics.inflammation_markers['CRP'] * inflammation_factor,
            'IL6': self.current_metrics.inflammation_markers['IL6'] * inflammation_factor,
            'TNF_alpha': self.current_metrics.inflammation_markers['TNF_alpha'] * inflammation_factor
        }
        
        # Update cellular health score
        cell_health_scores = [cell.get_health_score() for cell in self.cells.values()]
        new_cellular_health = np.mean(cell_health_scores)
        
        # Update epigenetic age (slow aging process)
        aging_rate = 1.0 / (365.25 * 24 * 3600)  # 1 year per real year
        new_epigenetic_age = self.current_metrics.epigenetic_age + aging_rate * dt
        
        return BodyMetrics(
            age=self.current_metrics.age + dt / (365.25 * 24 * 3600),
            bmi=self.bmi,
            heart_rate=new_heart_rate,
            blood_pressure_systolic=new_systolic,
            blood_pressure_diastolic=new_diastolic,
            temperature=self.current_metrics.temperature + np.random.normal(0, 0.1),
            oxygen_saturation=self.current_metrics.oxygen_saturation + np.random.normal(0, 0.5),
            glucose_level=new_glucose,
            cholesterol_total=self.current_metrics.cholesterol_total + np.random.normal(0, 1),
            inflammation_markers=new_inflammation,
            epigenetic_age=new_epigenetic_age,
            cellular_health_score=new_cellular_health,
            organ_function_scores=organ_scores
        )
    
    def apply_drug(self, drug_name: str, dose: float, duration: float = 3600.0):
        """
        Apply a drug intervention to the body.
        
        Args:
            drug_name: Name of the drug
            dose: Dose in mg
            duration: Duration in seconds
        """
        self.active_drugs[drug_name] = {
            'dose': dose,
            'start_time': self.simulation_time,
            'duration': duration,
            'effects': self._calculate_drug_effects(drug_name, dose)
        }
        
        logger.info(f"Applied drug {drug_name} at dose {dose}mg for {duration}s")
    
    def _calculate_drug_effects(self, drug_name: str, dose: float) -> Dict[str, float]:
        """Calculate the effects of a drug on various body systems"""
        # Simplified drug effects model
        effects = {}
        
        if drug_name.lower() in ['aspirin', 'ibuprofen']:
            effects['inflammation_reduction'] = min(0.5, dose / 100)
            effects['pain_reduction'] = min(0.7, dose / 50)
        elif drug_name.lower() in ['metformin']:
            effects['glucose_reduction'] = min(0.3, dose / 200)
        elif drug_name.lower() in ['statin']:
            effects['cholesterol_reduction'] = min(0.4, dose / 100)
        elif drug_name.lower() in ['ace_inhibitor']:
            effects['blood_pressure_reduction'] = min(0.2, dose / 50)
        
        return effects
    
    def edit_gene(self, gene_name: str, expression_change: float):
        """
        Edit gene expression in the body.
        
        Args:
            gene_name: Name of the gene
            expression_change: Change in expression level (-1.0 to 1.0)
        """
        # Apply gene expression changes to relevant cells
        for cell_name, cell in self.cells.items():
            if gene_name in cell.gene_expression:
                cell.gene_expression[gene_name] += expression_change
                cell.gene_expression[gene_name] = max(0, min(2.0, cell.gene_expression[gene_name]))
        
        logger.info(f"Edited gene {gene_name} expression by {expression_change}")
    
    def set_time_acceleration(self, acceleration: float):
        """Set the time acceleration factor for simulation"""
        self.time_acceleration = max(0.1, min(1000.0, acceleration))
        logger.info(f"Set time acceleration to {self.time_acceleration}x")
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get a comprehensive health summary of the body"""
        return {
            'age': self.current_metrics.age,
            'overall_health_score': np.mean(list(self.current_metrics.organ_function_scores.values())),
            'organ_health': self.current_metrics.organ_function_scores,
            'vital_signs': {
                'heart_rate': self.current_metrics.heart_rate,
                'blood_pressure': f"{self.current_metrics.blood_pressure_systolic}/{self.current_metrics.blood_pressure_diastolic}",
                'temperature': self.current_metrics.temperature,
                'oxygen_saturation': self.current_metrics.oxygen_saturation
            },
            'biomarkers': {
                'glucose': self.current_metrics.glucose_level,
                'cholesterol': self.current_metrics.cholesterol_total,
                'inflammation': self.current_metrics.inflammation_markers
            },
            'cellular_health': self.current_metrics.cellular_health_score,
            'epigenetic_age': self.current_metrics.epigenetic_age,
            'active_drugs': list(self.active_drugs.keys()),
            'simulation_time': self.simulation_time
        }
    
    def export_state(self) -> Dict[str, Any]:
        """Export the current state of the body simulator"""
        return {
            'age': self.age,
            'sex': self.sex,
            'weight': self.weight,
            'height': self.height,
            'current_metrics': self.current_metrics.__dict__,
            'active_drugs': self.active_drugs,
            'simulation_time': self.simulation_time,
            'time_acceleration': self.time_acceleration,
            'organ_states': {name: organ.export_state() for name, organ in self.organs.items()},
            'tissue_states': {name: tissue.export_state() for name, tissue in self.tissues.items()},
            'cell_states': {name: cell.export_state() for name, cell in self.cells.items()}
        }
    
    def import_state(self, state: Dict[str, Any]):
        """Import a previously exported state"""
        self.age = state['age']
        self.sex = state['sex']
        self.weight = state['weight']
        self.height = state['height']
        self.simulation_time = state['simulation_time']
        self.time_acceleration = state['time_acceleration']
        self.active_drugs = state['active_drugs']
        
        # Reconstruct metrics
        metrics_dict = state['current_metrics']
        self.current_metrics = BodyMetrics(**metrics_dict)
        
        logger.info("Imported body simulator state successfully")