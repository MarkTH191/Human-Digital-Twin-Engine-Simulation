"""
Cell Simulator - Simulates individual human cell types and their processes

This module simulates individual human cell types, their gene expression,
protein synthesis, metabolic pathways, and responses to environmental changes.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import random

logger = logging.getLogger(__name__)

class CellState(Enum):
    """Cell states"""
    HEALTHY = "healthy"
    STRESSED = "stressed"
    DAMAGED = "damaged"
    APOPTOTIC = "apoptotic"
    SENESCENT = "senescent"
    CANCEROUS = "cancerous"

@dataclass
class CellMetrics:
    """Metrics for cell function and health"""
    viability: float
    metabolic_activity: float
    protein_synthesis_rate: float
    dna_repair_capacity: float
    antioxidant_capacity: float
    membrane_integrity: float
    mitochondrial_function: float
    cell_cycle_progression: float

class CellSimulator:
    """
    Simulates individual human cell types and their biological processes.
    
    Each cell simulator tracks gene expression, protein synthesis, metabolic
    pathways, DNA integrity, and responses to various stimuli and interventions.
    """
    
    def __init__(self, cell_type: str, age: float):
        """
        Initialize cell simulator.
        
        Args:
            cell_type: Type of cell
            age: Age in years
        """
        self.cell_type = cell_type
        self.age = age
        
        # Cell properties
        self.properties = self._get_cell_properties()
        
        # Current state
        self.state = CellState.HEALTHY
        self.metrics = self._calculate_initial_metrics()
        
        # Gene expression
        self.gene_expression = self._initialize_gene_expression()
        
        # Protein levels
        self.protein_levels = self._initialize_protein_levels()
        
        # Metabolic pathways
        self.metabolic_pathways = self._initialize_metabolic_pathways()
        
        # DNA integrity
        self.dna_integrity = 1.0
        self.mutations = []
        
        # Mitochondrial function
        self.mitochondrial_health = 1.0
        self.atp_production = 1.0
        
        # Cell cycle
        self.cell_cycle_phase = "G1"
        self.cell_cycle_progress = 0.0
        
        # Stress responses
        self.stress_markers = {}
        self.heat_shock_proteins = {}
        
        # Health tracking
        self.health_history = []
        self.damage_history = []
        
        logger.info(f"Initialized {cell_type} cell simulator for {age}y")
    
    def _get_cell_properties(self) -> Dict[str, Any]:
        """Get cell-specific properties and parameters"""
        properties = {
            'cardiomyocytes': {
                'lifespan': 100,  # years
                'division_rate': 0.0,  # non-dividing
                'metabolic_rate': 0.9,
                'oxygen_consumption': 0.8,
                'protein_synthesis': 0.7,
                'dna_repair': 0.6,
                'antioxidant_capacity': 0.5,
                'membrane_stability': 0.8,
                'mitochondrial_density': 0.9
            },
            'neurons': {
                'lifespan': 100,
                'division_rate': 0.0,
                'metabolic_rate': 1.0,
                'oxygen_consumption': 1.0,
                'protein_synthesis': 0.8,
                'dna_repair': 0.4,
                'antioxidant_capacity': 0.3,
                'membrane_stability': 0.7,
                'mitochondrial_density': 0.8
            },
            'hepatocytes': {
                'lifespan': 1,  # years
                'division_rate': 0.1,
                'metabolic_rate': 0.8,
                'oxygen_consumption': 0.7,
                'protein_synthesis': 0.9,
                'dna_repair': 0.7,
                'antioxidant_capacity': 0.8,
                'membrane_stability': 0.6,
                'mitochondrial_density': 0.7
            },
            'nephron_cells': {
                'lifespan': 1,
                'division_rate': 0.05,
                'metabolic_rate': 0.6,
                'oxygen_consumption': 0.5,
                'protein_synthesis': 0.6,
                'dna_repair': 0.5,
                'antioxidant_capacity': 0.6,
                'membrane_stability': 0.7,
                'mitochondrial_density': 0.6
            },
            'alveolar_cells': {
                'lifespan': 0.5,
                'division_rate': 0.2,
                'metabolic_rate': 0.5,
                'oxygen_consumption': 0.3,
                'protein_synthesis': 0.5,
                'dna_repair': 0.6,
                'antioxidant_capacity': 0.7,
                'membrane_stability': 0.5,
                'mitochondrial_density': 0.4
            },
            'beta_cells': {
                'lifespan': 2,
                'division_rate': 0.02,
                'metabolic_rate': 0.7,
                'oxygen_consumption': 0.6,
                'protein_synthesis': 0.8,
                'dna_repair': 0.5,
                'antioxidant_capacity': 0.6,
                'membrane_stability': 0.6,
                'mitochondrial_density': 0.7
            },
            't_cells': {
                'lifespan': 0.1,
                'division_rate': 0.5,
                'metabolic_rate': 0.8,
                'oxygen_consumption': 0.7,
                'protein_synthesis': 0.7,
                'dna_repair': 0.6,
                'antioxidant_capacity': 0.5,
                'membrane_stability': 0.5,
                'mitochondrial_density': 0.6
            },
            'b_cells': {
                'lifespan': 0.2,
                'division_rate': 0.3,
                'metabolic_rate': 0.7,
                'oxygen_consumption': 0.6,
                'protein_synthesis': 0.8,
                'dna_repair': 0.6,
                'antioxidant_capacity': 0.5,
                'membrane_stability': 0.5,
                'mitochondrial_density': 0.5
            },
            'macrophages': {
                'lifespan': 0.1,
                'division_rate': 0.1,
                'metabolic_rate': 0.8,
                'oxygen_consumption': 0.8,
                'protein_synthesis': 0.7,
                'dna_repair': 0.5,
                'antioxidant_capacity': 0.7,
                'membrane_stability': 0.6,
                'mitochondrial_density': 0.7
            },
            'fibroblasts': {
                'lifespan': 0.5,
                'division_rate': 0.1,
                'metabolic_rate': 0.6,
                'oxygen_consumption': 0.5,
                'protein_synthesis': 0.8,
                'dna_repair': 0.7,
                'antioxidant_capacity': 0.6,
                'membrane_stability': 0.7,
                'mitochondrial_density': 0.5
            },
            'keratinocytes': {
                'lifespan': 0.1,
                'division_rate': 0.3,
                'metabolic_rate': 0.5,
                'oxygen_consumption': 0.4,
                'protein_synthesis': 0.6,
                'dna_repair': 0.6,
                'antioxidant_capacity': 0.7,
                'membrane_stability': 0.8,
                'mitochondrial_density': 0.4
            },
            'osteocytes': {
                'lifespan': 10,
                'division_rate': 0.0,
                'metabolic_rate': 0.3,
                'oxygen_consumption': 0.2,
                'protein_synthesis': 0.4,
                'dna_repair': 0.5,
                'antioxidant_capacity': 0.4,
                'membrane_stability': 0.9,
                'mitochondrial_density': 0.3
            },
            'adipocytes': {
                'lifespan': 10,
                'division_rate': 0.01,
                'metabolic_rate': 0.3,
                'oxygen_consumption': 0.2,
                'protein_synthesis': 0.3,
                'dna_repair': 0.4,
                'antioxidant_capacity': 0.3,
                'membrane_stability': 0.6,
                'mitochondrial_density': 0.2
            },
            'stem_cells': {
                'lifespan': 100,
                'division_rate': 0.05,
                'metabolic_rate': 0.6,
                'oxygen_consumption': 0.4,
                'protein_synthesis': 0.7,
                'dna_repair': 0.9,
                'antioxidant_capacity': 0.8,
                'membrane_stability': 0.8,
                'mitochondrial_density': 0.6
            }
        }
        
        return properties.get(self.cell_type, {
            'lifespan': 1,
            'division_rate': 0.1,
            'metabolic_rate': 0.5,
            'oxygen_consumption': 0.5,
            'protein_synthesis': 0.5,
            'dna_repair': 0.5,
            'antioxidant_capacity': 0.5,
            'membrane_stability': 0.5,
            'mitochondrial_density': 0.5
        })
    
    def _calculate_initial_metrics(self) -> CellMetrics:
        """Calculate initial cell metrics based on age and properties"""
        # Age-related decline factors
        age_factor = max(0.3, 1.0 - (self.age - 25) * 0.02)
        
        return CellMetrics(
            viability=0.95 * age_factor,
            metabolic_activity=self.properties['metabolic_rate'] * age_factor,
            protein_synthesis_rate=self.properties['protein_synthesis'] * age_factor,
            dna_repair_capacity=self.properties['dna_repair'] * age_factor,
            antioxidant_capacity=self.properties['antioxidant_capacity'] * age_factor,
            membrane_integrity=self.properties['membrane_stability'] * age_factor,
            mitochondrial_function=self.properties['mitochondrial_density'] * age_factor,
            cell_cycle_progression=0.0
        )
    
    def _initialize_gene_expression(self) -> Dict[str, float]:
        """Initialize gene expression levels"""
        # Common genes with cell-type specific expression
        genes = {
            'TP53': 1.0,  # Tumor suppressor
            'MYC': 0.5,   # Transcription factor
            'BCL2': 0.7,  # Anti-apoptotic
            'BAX': 0.3,   # Pro-apoptotic
            'SOD1': 0.8,  # Antioxidant
            'CAT': 0.6,   # Antioxidant
            'GPX1': 0.7,  # Antioxidant
            'HSP70': 0.5, # Heat shock protein
            'HSP90': 0.4, # Heat shock protein
            'TERT': 0.1,  # Telomerase
            'CDK4': 0.6,  # Cell cycle
            'CCND1': 0.5, # Cell cycle
            'PCNA': 0.4,  # DNA replication
            'BRCA1': 0.6, # DNA repair
            'ATM': 0.5,   # DNA repair
            'PARP1': 0.7, # DNA repair
            'NFKB': 0.4,  # Inflammation
            'IL6': 0.2,   # Inflammation
            'TNF': 0.1,   # Inflammation
            'INS': 0.0,   # Insulin (cell-type specific)
            'GLUT4': 0.3, # Glucose transporter
            'MTOR': 0.6,  # Growth signaling
            'AKT1': 0.5,  # Survival signaling
            'MAPK1': 0.4, # Signaling
            'STAT3': 0.3  # Signaling
        }
        
        # Cell-type specific adjustments
        if self.cell_type == 'beta_cells':
            genes['INS'] = 1.0
            genes['GLUT2'] = 1.0
        elif self.cell_type == 'cardiomyocytes':
            genes['MYH7'] = 1.0  # Myosin heavy chain
            genes['TNNT2'] = 1.0  # Troponin
        elif self.cell_type == 'neurons':
            genes['SYN1'] = 1.0  # Synapsin
            genes['MAP2'] = 1.0  # Microtubule protein
        elif self.cell_type == 'hepatocytes':
            genes['CYP3A4'] = 1.0  # Cytochrome P450
            genes['ALB'] = 1.0     # Albumin
        
        return genes
    
    def _initialize_protein_levels(self) -> Dict[str, float]:
        """Initialize protein levels based on gene expression"""
        proteins = {}
        for gene, expression in self.gene_expression.items():
            # Protein level is related to gene expression but with some delay/noise
            protein_level = expression * (0.8 + 0.4 * random.random())
            proteins[gene] = max(0.0, min(2.0, protein_level))
        
        return proteins
    
    def _initialize_metabolic_pathways(self) -> Dict[str, float]:
        """Initialize metabolic pathway activities"""
        return {
            'glycolysis': 1.0,
            'oxidative_phosphorylation': 1.0,
            'fatty_acid_oxidation': 1.0,
            'protein_synthesis': 1.0,
            'dna_synthesis': 0.5,
            'rna_synthesis': 0.8,
            'lipid_synthesis': 0.6,
            'antioxidant_defense': 1.0,
            'detoxification': 0.7,
            'autophagy': 0.5
        }
    
    def simulate_step(self, dt: float, body_metrics: Any):
        """
        Simulate one time step of cell function.
        
        Args:
            dt: Time step in seconds
            body_metrics: Current body metrics
        """
        # Update gene expression
        self._update_gene_expression(dt)
        
        # Update protein levels
        self._update_protein_levels(dt)
        
        # Update metabolic pathways
        self._update_metabolic_pathways(dt)
        
        # Update DNA integrity
        self._update_dna_integrity(dt)
        
        # Update mitochondrial function
        self._update_mitochondrial_function(dt)
        
        # Update cell cycle
        self._update_cell_cycle(dt)
        
        # Update stress responses
        self._update_stress_responses(dt)
        
        # Update cell metrics
        self._update_metrics(dt)
        
        # Update cell state
        self._update_cell_state()
        
        # Store health history
        health_score = self._calculate_health_score()
        self.health_history.append(health_score)
        if len(self.health_history) > 1000:
            self.health_history = self.health_history[-1000:]
    
    def _update_gene_expression(self, dt: float):
        """Update gene expression levels"""
        for gene in self.gene_expression:
            # Random fluctuations
            fluctuation = np.random.normal(0, 0.01)
            self.gene_expression[gene] += fluctuation
            
            # Age-related changes
            if gene in ['TERT', 'SOD1', 'CAT']:
                # Telomerase and antioxidants decrease with age
                age_decline = (self.age - 25) * 0.001
                self.gene_expression[gene] -= age_decline
            
            # Keep within bounds
            self.gene_expression[gene] = max(0.0, min(2.0, self.gene_expression[gene]))
    
    def _update_protein_levels(self, dt: float):
        """Update protein levels based on gene expression"""
        for gene, expression in self.gene_expression.items():
            if gene in self.protein_levels:
                # Protein level approaches gene expression level with some delay
                target_level = expression * (0.9 + 0.2 * random.random())
                current_level = self.protein_levels[gene]
                
                # Gradual change towards target
                change_rate = 0.1
                new_level = current_level + (target_level - current_level) * change_rate
                self.protein_levels[gene] = max(0.0, min(2.0, new_level))
    
    def _update_metabolic_pathways(self, dt: float):
        """Update metabolic pathway activities"""
        for pathway in self.metabolic_pathways:
            # Random fluctuations
            fluctuation = np.random.normal(0, 0.01)
            self.metabolic_pathways[pathway] += fluctuation
            
            # Age-related decline
            age_factor = max(0.3, 1.0 - (self.age - 25) * 0.01)
            self.metabolic_pathways[pathway] *= age_factor
            
            # Keep within bounds
            self.metabolic_pathways[pathway] = max(0.1, min(2.0, self.metabolic_pathways[pathway]))
    
    def _update_dna_integrity(self, dt: float):
        """Update DNA integrity and accumulate damage"""
        # Base damage rate
        damage_rate = 1e-6 * dt
        
        # Increase damage rate with age
        age_factor = 1.0 + (self.age - 25) * 0.02
        damage_rate *= age_factor
        
        # Increase damage rate if antioxidant capacity is low
        antioxidant_factor = 1.0 / max(0.1, self.metrics.antioxidant_capacity)
        damage_rate *= antioxidant_factor
        
        # Apply damage
        self.dna_integrity -= damage_rate
        self.dna_integrity = max(0.0, self.dna_integrity)
        
        # DNA repair
        repair_rate = self.metrics.dna_repair_capacity * 1e-5 * dt
        self.dna_integrity += repair_rate
        self.dna_integrity = min(1.0, self.dna_integrity)
    
    def _update_mitochondrial_function(self, dt: float):
        """Update mitochondrial function and ATP production"""
        # Mitochondrial damage accumulation
        damage_rate = 1e-5 * dt
        
        # Increase damage with age
        age_factor = 1.0 + (self.age - 25) * 0.01
        damage_rate *= age_factor
        
        # Apply damage
        self.mitochondrial_health -= damage_rate
        self.mitochondrial_health = max(0.0, self.mitochondrial_health)
        
        # ATP production based on mitochondrial health
        self.atp_production = self.mitochondrial_health * self.properties['mitochondrial_density']
    
    def _update_cell_cycle(self, dt: float):
        """Update cell cycle progression"""
        if self.properties['division_rate'] > 0:
            # Cell cycle phases: G1 -> S -> G2 -> M
            cycle_duration = 24 * 3600  # 24 hours in seconds
            
            # Progress through cell cycle
            self.cell_cycle_progress += dt / cycle_duration
            
            if self.cell_cycle_progress >= 1.0:
                # Cell division
                self.cell_cycle_progress = 0.0
                self._cell_division()
            
            # Update cell cycle phase
            if self.cell_cycle_progress < 0.4:
                self.cell_cycle_phase = "G1"
            elif self.cell_cycle_progress < 0.6:
                self.cell_cycle_phase = "S"
            elif self.cell_cycle_progress < 0.9:
                self.cell_cycle_phase = "G2"
            else:
                self.cell_cycle_phase = "M"
    
    def _cell_division(self):
        """Simulate cell division"""
        # Reset some metrics after division
        self.metrics.viability = min(1.0, self.metrics.viability + 0.1)
        self.dna_integrity = min(1.0, self.dna_integrity + 0.05)
        
        # Small chance of mutation
        if random.random() < 0.001:
            self._add_mutation()
    
    def _add_mutation(self):
        """Add a random mutation to the cell"""
        mutation_types = ['point_mutation', 'deletion', 'insertion', 'translocation']
        mutation_type = random.choice(mutation_types)
        
        mutation = {
            'type': mutation_type,
            'gene': random.choice(list(self.gene_expression.keys())),
            'effect': random.uniform(-0.5, 0.5),
            'time': self.age
        }
        
        self.mutations.append(mutation)
        
        # Apply mutation effect
        if mutation['gene'] in self.gene_expression:
            self.gene_expression[mutation['gene']] += mutation['effect']
            self.gene_expression[mutation['gene']] = max(0.0, min(2.0, self.gene_expression[mutation['gene']]))
    
    def _update_stress_responses(self, dt: float):
        """Update cellular stress responses"""
        # Calculate stress level
        stress_level = 0.0
        
        # DNA damage stress
        stress_level += (1.0 - self.dna_integrity) * 0.3
        
        # Mitochondrial stress
        stress_level += (1.0 - self.mitochondrial_health) * 0.2
        
        # Metabolic stress
        metabolic_stress = 1.0 - np.mean(list(self.metabolic_pathways.values()))
        stress_level += metabolic_stress * 0.2
        
        # Update stress markers
        self.stress_markers['overall_stress'] = stress_level
        self.stress_markers['dna_damage'] = 1.0 - self.dna_integrity
        self.stress_markers['mitochondrial_dysfunction'] = 1.0 - self.mitochondrial_health
        
        # Update heat shock proteins based on stress
        if stress_level > 0.3:
            self.heat_shock_proteins['HSP70'] = min(2.0, stress_level * 2)
            self.heat_shock_proteins['HSP90'] = min(2.0, stress_level * 1.5)
        else:
            self.heat_shock_proteins['HSP70'] = max(0.1, self.heat_shock_proteins.get('HSP70', 0.5) * 0.99)
            self.heat_shock_proteins['HSP90'] = max(0.1, self.heat_shock_proteins.get('HSP90', 0.4) * 0.99)
    
    def _update_metrics(self, dt: float):
        """Update cell metrics based on current state"""
        # Viability based on multiple factors
        viability_factors = [
            self.dna_integrity,
            self.mitochondrial_health,
            self.metrics.membrane_integrity,
            np.mean(list(self.metabolic_pathways.values()))
        ]
        self.metrics.viability = np.mean(viability_factors)
        
        # Metabolic activity
        self.metrics.metabolic_activity = np.mean(list(self.metabolic_pathways.values()))
        
        # Protein synthesis rate
        self.metrics.protein_synthesis_rate = self.metabolic_pathways['protein_synthesis']
        
        # DNA repair capacity
        self.metrics.dna_repair_capacity = self.properties['dna_repair'] * self.dna_integrity
        
        # Antioxidant capacity
        self.metrics.antioxidant_capacity = self.properties['antioxidant_capacity'] * (1.0 - self.stress_markers.get('overall_stress', 0.0))
        
        # Membrane integrity
        stress_factor = 1.0 - self.stress_markers.get('overall_stress', 0.0)
        self.metrics.membrane_integrity = self.properties['membrane_stability'] * stress_factor
        
        # Mitochondrial function
        self.metrics.mitochondrial_function = self.mitochondrial_health * self.properties['mitochondrial_density']
        
        # Cell cycle progression
        self.metrics.cell_cycle_progression = self.cell_cycle_progress
    
    def _update_cell_state(self):
        """Update cell state based on current metrics"""
        viability = self.metrics.viability
        stress_level = self.stress_markers.get('overall_stress', 0.0)
        
        if viability < 0.3:
            self.state = CellState.APOPTOTIC
        elif viability < 0.5:
            self.state = CellState.DAMAGED
        elif stress_level > 0.7:
            self.state = CellState.STRESSED
        elif self.age > 80 and viability < 0.7:
            self.state = CellState.SENESCENT
        elif len(self.mutations) > 10:
            self.state = CellState.CANCEROUS
        else:
            self.state = CellState.HEALTHY
    
    def _calculate_health_score(self) -> float:
        """Calculate overall health score of the cell"""
        # Weighted combination of various factors
        viability_score = self.metrics.viability
        metabolic_score = self.metrics.metabolic_activity
        dna_score = self.dna_integrity
        mitochondrial_score = self.mitochondrial_health
        stress_score = 1.0 - self.stress_markers.get('overall_stress', 0.0)
        
        health_score = (
            0.3 * viability_score +
            0.2 * metabolic_score +
            0.2 * dna_score +
            0.15 * mitochondrial_score +
            0.15 * stress_score
        )
        
        return max(0.0, min(1.0, health_score))
    
    def apply_drug_effect(self, drug_name: str, effect_strength: float):
        """Apply drug effects to the cell"""
        if drug_name.lower() in ['aspirin', 'ibuprofen']:
            # Anti-inflammatory effect
            if 'NFKB' in self.gene_expression:
                self.gene_expression['NFKB'] *= (1.0 - effect_strength * 0.3)
        elif drug_name.lower() in ['metformin']:
            # Improve mitochondrial function
            self.mitochondrial_health += effect_strength * 0.1
            self.mitochondrial_health = min(1.0, self.mitochondrial_health)
        elif drug_name.lower() in ['statin']:
            # Reduce cholesterol synthesis
            if 'HMGCR' in self.gene_expression:
                self.gene_expression['HMGCR'] *= (1.0 - effect_strength * 0.5)
    
    def apply_gene_edit(self, gene_name: str, expression_change: float):
        """Apply gene editing to the cell"""
        if gene_name in self.gene_expression:
            self.gene_expression[gene_name] += expression_change
            self.gene_expression[gene_name] = max(0.0, min(2.0, self.gene_expression[gene_name]))
    
    def get_health_score(self) -> float:
        """Get overall health score of the cell"""
        return self._calculate_health_score()
    
    def get_gene_expression(self) -> Dict[str, float]:
        """Get current gene expression levels"""
        return self.gene_expression.copy()
    
    def get_protein_levels(self) -> Dict[str, float]:
        """Get current protein levels"""
        return self.protein_levels.copy()
    
    def get_metabolic_state(self) -> Dict[str, float]:
        """Get current metabolic pathway activities"""
        return self.metabolic_pathways.copy()
    
    def get_mutations(self) -> List[Dict[str, Any]]:
        """Get list of mutations"""
        return self.mutations.copy()
    
    def export_state(self) -> Dict[str, Any]:
        """Export current cell state"""
        return {
            'cell_type': self.cell_type,
            'age': self.age,
            'state': self.state.value,
            'metrics': self.metrics.__dict__,
            'gene_expression': self.gene_expression,
            'protein_levels': self.protein_levels,
            'metabolic_pathways': self.metabolic_pathways,
            'dna_integrity': self.dna_integrity,
            'mitochondrial_health': self.mitochondrial_health,
            'atp_production': self.atp_production,
            'cell_cycle_phase': self.cell_cycle_phase,
            'cell_cycle_progress': self.cell_cycle_progress,
            'stress_markers': self.stress_markers,
            'heat_shock_proteins': self.heat_shock_proteins,
            'mutations': self.mutations
        }
    
    def import_state(self, state: Dict[str, Any]):
        """Import cell state"""
        self.age = state['age']
        self.state = CellState(state['state'])
        self.gene_expression = state['gene_expression']
        self.protein_levels = state['protein_levels']
        self.metabolic_pathways = state['metabolic_pathways']
        self.dna_integrity = state['dna_integrity']
        self.mitochondrial_health = state['mitochondrial_health']
        self.atp_production = state['atp_production']
        self.cell_cycle_phase = state['cell_cycle_phase']
        self.cell_cycle_progress = state['cell_cycle_progress']
        self.stress_markers = state['stress_markers']
        self.heat_shock_proteins = state['heat_shock_proteins']
        self.mutations = state['mutations']
        
        # Reconstruct metrics
        metrics_dict = state['metrics']
        self.metrics = CellMetrics(**metrics_dict)