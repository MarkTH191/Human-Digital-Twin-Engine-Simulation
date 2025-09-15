"""
Molecule Simulator - Simulates molecular interactions and biochemical processes

This module simulates molecular interactions, protein folding, enzyme kinetics,
metabolic pathways, and biochemical reactions within the human body.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import random

logger = logging.getLogger(__name__)

class MoleculeType(Enum):
    """Types of molecules"""
    PROTEIN = "protein"
    DNA = "dna"
    RNA = "rna"
    LIPID = "lipid"
    CARBOHYDRATE = "carbohydrate"
    METABOLITE = "metabolite"
    HORMONE = "hormone"
    NEUROTRANSMITTER = "neurotransmitter"
    DRUG = "drug"
    ION = "ion"

@dataclass
class Molecule:
    """Represents a molecule with its properties"""
    name: str
    molecule_type: MoleculeType
    concentration: float
    molecular_weight: float
    charge: float
    solubility: float
    stability: float
    binding_affinity: Dict[str, float]

@dataclass
class Reaction:
    """Represents a biochemical reaction"""
    reactants: List[str]
    products: List[str]
    rate_constant: float
    equilibrium_constant: float
    activation_energy: float
    enzyme: Optional[str]

class MoleculeSimulator:
    """
    Simulates molecular interactions and biochemical processes.
    
    This simulator tracks molecular concentrations, protein folding,
    enzyme kinetics, metabolic pathways, and drug interactions.
    """
    
    def __init__(self):
        """Initialize molecule simulator"""
        # Molecular database
        self.molecules = self._initialize_molecules()
        
        # Reaction database
        self.reactions = self._initialize_reactions()
        
        # Protein folding
        self.protein_structures = {}
        
        # Enzyme kinetics
        self.enzyme_activities = {}
        
        # Metabolic pathways
        self.metabolic_pathways = self._initialize_metabolic_pathways()
        
        # Drug interactions
        self.drug_interactions = {}
        
        # Molecular transport
        self.transport_rates = {}
        
        # pH and ionic environment
        self.ph = 7.4
        self.ionic_strength = 0.15
        
        # Temperature
        self.temperature = 310.15  # 37°C in Kelvin
        
        logger.info("Initialized molecule simulator")
    
    def _initialize_molecules(self) -> Dict[str, Molecule]:
        """Initialize common biological molecules"""
        molecules = {}
        
        # Proteins
        proteins = {
            'insulin': Molecule('insulin', MoleculeType.PROTEIN, 0.1, 5808, 0, 0.8, 0.9, {'insulin_receptor': 0.95}),
            'glucagon': Molecule('glucagon', MoleculeType.PROTEIN, 0.05, 3483, 0, 0.7, 0.8, {'glucagon_receptor': 0.9}),
            'cortisol': Molecule('cortisol', MoleculeType.HORMONE, 0.2, 362.47, 0, 0.6, 0.7, {'glucocorticoid_receptor': 0.85}),
            'adrenaline': Molecule('adrenaline', MoleculeType.HORMONE, 0.01, 183.2, 1, 0.9, 0.6, {'adrenergic_receptor': 0.8}),
            'dopamine': Molecule('dopamine', MoleculeType.NEUROTRANSMITTER, 0.001, 153.18, 1, 0.8, 0.5, {'dopamine_receptor': 0.75}),
            'serotonin': Molecule('serotonin', MoleculeType.NEUROTRANSMITTER, 0.0005, 176.21, 1, 0.7, 0.6, {'serotonin_receptor': 0.7}),
            'hemoglobin': Molecule('hemoglobin', MoleculeType.PROTEIN, 2.5, 64500, 0, 0.9, 0.95, {'oxygen': 0.9}),
            'albumin': Molecule('albumin', MoleculeType.PROTEIN, 4.0, 66500, -15, 0.95, 0.9, {'fatty_acids': 0.8}),
            'glucose': Molecule('glucose', MoleculeType.CARBOHYDRATE, 5.0, 180.16, 0, 1.0, 0.8, {'glucose_transporter': 0.7}),
            'atp': Molecule('atp', MoleculeType.METABOLITE, 2.0, 507.18, -4, 0.9, 0.7, {'atpase': 0.8}),
            'cholesterol': Molecule('cholesterol', MoleculeType.LIPID, 0.2, 386.65, 0, 0.1, 0.9, {'ldl_receptor': 0.6}),
            'sodium': Molecule('sodium', MoleculeType.ION, 140.0, 22.99, 1, 1.0, 1.0, {'sodium_channel': 0.9}),
            'potassium': Molecule('potassium', MoleculeType.ION, 4.0, 39.10, 1, 1.0, 1.0, {'potassium_channel': 0.9}),
            'calcium': Molecule('calcium', MoleculeType.ION, 2.5, 40.08, 2, 0.8, 0.9, {'calcium_channel': 0.8}),
            'oxygen': Molecule('oxygen', MoleculeType.METABOLITE, 0.1, 32.0, 0, 0.3, 0.5, {'hemoglobin': 0.9}),
            'carbon_dioxide': Molecule('carbon_dioxide', MoleculeType.METABOLITE, 1.2, 44.01, 0, 0.7, 0.6, {'carbonic_anhydrase': 0.7})
        }
        
        molecules.update(proteins)
        
        # Add some common drugs
        drugs = {
            'aspirin': Molecule('aspirin', MoleculeType.DRUG, 0.0, 180.16, 0, 0.6, 0.7, {'cox_enzyme': 0.8}),
            'metformin': Molecule('metformin', MoleculeType.DRUG, 0.0, 129.16, 1, 0.9, 0.8, {'ampk': 0.7}),
            'statin': Molecule('statin', MoleculeType.DRUG, 0.0, 400.0, 0, 0.5, 0.8, {'hmgcr': 0.9})
        }
        
        molecules.update(drugs)
        
        return molecules
    
    def _initialize_reactions(self) -> Dict[str, Reaction]:
        """Initialize common biochemical reactions"""
        reactions = {}
        
        # Glycolysis
        reactions['glucose_to_glucose6p'] = Reaction(
            ['glucose', 'atp'], ['glucose6p', 'adp'], 1.0, 1000, 50, 'hexokinase'
        )
        
        reactions['glucose6p_to_fructose6p'] = Reaction(
            ['glucose6p'], ['fructose6p'], 0.5, 0.5, 30, 'phosphoglucose_isomerase'
        )
        
        reactions['fructose6p_to_fructose16bp'] = Reaction(
            ['fructose6p', 'atp'], ['fructose16bp', 'adp'], 0.3, 100, 40, 'phosphofructokinase'
        )
        
        # ATP synthesis
        reactions['adp_to_atp'] = Reaction(
            ['adp', 'phosphate'], ['atp'], 2.0, 1000, 60, 'atp_synthase'
        )
        
        # Protein synthesis
        reactions['amino_acids_to_protein'] = Reaction(
            ['amino_acids', 'atp'], ['protein'], 0.1, 100, 80, 'ribosome'
        )
        
        # Drug metabolism
        reactions['aspirin_metabolism'] = Reaction(
            ['aspirin'], ['salicylic_acid'], 0.2, 10, 30, 'esterase'
        )
        
        # Hormone binding
        reactions['insulin_binding'] = Reaction(
            ['insulin', 'insulin_receptor'], ['insulin_receptor_complex'], 5.0, 1000, 20, None
        )
        
        # Oxygen binding
        reactions['oxygen_binding'] = Reaction(
            ['oxygen', 'hemoglobin'], ['oxyhemoglobin'], 10.0, 1000, 10, None
        )
        
        return reactions
    
    def _initialize_metabolic_pathways(self) -> Dict[str, Dict[str, Any]]:
        """Initialize metabolic pathways"""
        pathways = {
            'glycolysis': {
                'reactions': ['glucose_to_glucose6p', 'glucose6p_to_fructose6p', 'fructose6p_to_fructose16bp'],
                'flux': 1.0,
                'regulation': {'insulin': 1.5, 'glucagon': 0.5}
            },
            'atp_synthesis': {
                'reactions': ['adp_to_atp'],
                'flux': 2.0,
                'regulation': {'oxygen': 2.0, 'glucose': 1.5}
            },
            'protein_synthesis': {
                'reactions': ['amino_acids_to_protein'],
                'flux': 0.1,
                'regulation': {'insulin': 1.2, 'growth_hormone': 1.3}
            },
            'drug_metabolism': {
                'reactions': ['aspirin_metabolism'],
                'flux': 0.2,
                'regulation': {'cyp_enzymes': 1.0}
            }
        }
        
        return pathways
    
    def simulate_step(self, dt: float, body_metrics: Any):
        """
        Simulate one time step of molecular processes.
        
        Args:
            dt: Time step in seconds
            body_metrics: Current body metrics
        """
        # Update molecular concentrations
        self._update_concentrations(dt)
        
        # Simulate reactions
        self._simulate_reactions(dt)
        
        # Update metabolic pathways
        self._update_metabolic_pathways(dt, body_metrics)
        
        # Update protein folding
        self._update_protein_folding(dt)
        
        # Update enzyme activities
        self._update_enzyme_activities(dt)
        
        # Update drug interactions
        self._update_drug_interactions(dt)
        
        # Update molecular transport
        self._update_transport(dt)
    
    def _update_concentrations(self, dt: float):
        """Update molecular concentrations based on production and degradation"""
        for molecule_name, molecule in self.molecules.items():
            # Random fluctuations
            fluctuation = np.random.normal(0, 0.01)
            molecule.concentration += fluctuation * dt
            
            # Degradation (first-order kinetics)
            if molecule.molecule_type in [MoleculeType.PROTEIN, MoleculeType.HORMONE]:
                degradation_rate = 0.01 * dt  # 1% per second
                molecule.concentration *= (1.0 - degradation_rate)
            
            # Keep concentrations positive
            molecule.concentration = max(0.0, molecule.concentration)
    
    def _simulate_reactions(self, dt: float):
        """Simulate biochemical reactions"""
        for reaction_name, reaction in self.reactions.items():
            # Calculate reaction rate using mass action kinetics
            reactant_concentrations = []
            for reactant in reaction.reactants:
                if reactant in self.molecules:
                    reactant_concentrations.append(self.molecules[reactant].concentration)
                else:
                    reactant_concentrations.append(0.0)
            
            # Forward reaction rate
            forward_rate = reaction.rate_constant
            for conc in reactant_concentrations:
                forward_rate *= conc
            
            # Apply enzyme activity
            if reaction.enzyme and reaction.enzyme in self.enzyme_activities:
                forward_rate *= self.enzyme_activities[reaction.enzyme]
            
            # Temperature effect (Arrhenius equation)
            temp_factor = np.exp(-reaction.activation_energy / (8.314 * self.temperature))
            forward_rate *= temp_factor
            
            # Apply reaction
            reaction_flux = forward_rate * dt
            
            # Update reactant concentrations
            for reactant in reaction.reactants:
                if reactant in self.molecules:
                    self.molecules[reactant].concentration -= reaction_flux
                    self.molecules[reactant].concentration = max(0.0, self.molecules[reactant].concentration)
            
            # Update product concentrations
            for product in reaction.products:
                if product in self.molecules:
                    self.molecules[product].concentration += reaction_flux
    
    def _update_metabolic_pathways(self, dt: float, body_metrics: Any):
        """Update metabolic pathway fluxes"""
        for pathway_name, pathway in self.metabolic_pathways.items():
            # Base flux
            base_flux = pathway['flux']
            
            # Apply regulation
            regulation_factor = 1.0
            for regulator, effect in pathway['regulation'].items():
                if regulator in self.molecules:
                    concentration = self.molecules[regulator].concentration
                    # Simple regulation model
                    regulation_factor *= (1.0 + (concentration - 1.0) * (effect - 1.0))
            
            # Update pathway flux
            pathway['flux'] = base_flux * regulation_factor
            
            # Apply to individual reactions
            for reaction_name in pathway['reactions']:
                if reaction_name in self.reactions:
                    # Update reaction rate based on pathway flux
                    self.reactions[reaction_name].rate_constant *= regulation_factor
    
    def _update_protein_folding(self, dt: float):
        """Update protein folding states"""
        for protein_name, molecule in self.molecules.items():
            if molecule.molecule_type == MoleculeType.PROTEIN:
                # Initialize protein structure if not exists
                if protein_name not in self.protein_structures:
                    self.protein_structures[protein_name] = {
                        'folded_fraction': 0.8,
                        'stability': molecule.stability,
                        'denaturation_rate': 0.001
                    }
                
                structure = self.protein_structures[protein_name]
                
                # Protein folding/unfolding dynamics
                folding_rate = 0.1 * dt
                unfolding_rate = structure['denaturation_rate'] * dt
                
                # Temperature effect on stability
                temp_factor = np.exp(-(self.temperature - 310.15) / 10.0)
                unfolding_rate *= temp_factor
                
                # Update folded fraction
                structure['folded_fraction'] += folding_rate - unfolding_rate
                structure['folded_fraction'] = max(0.0, min(1.0, structure['folded_fraction']))
    
    def _update_enzyme_activities(self, dt: float):
        """Update enzyme activities"""
        # Initialize enzyme activities if not exists
        if not self.enzyme_activities:
            for reaction in self.reactions.values():
                if reaction.enzyme:
                    self.enzyme_activities[reaction.enzyme] = 1.0
        
        # Update enzyme activities
        for enzyme_name in self.enzyme_activities:
            # Random fluctuations
            fluctuation = np.random.normal(0, 0.01)
            self.enzyme_activities[enzyme_name] += fluctuation * dt
            
            # pH effect
            ph_factor = 1.0 - abs(self.ph - 7.4) * 0.1
            self.enzyme_activities[enzyme_name] *= ph_factor
            
            # Keep within bounds
            self.enzyme_activities[enzyme_name] = max(0.1, min(2.0, self.enzyme_activities[enzyme_name]))
    
    def _update_drug_interactions(self, dt: float):
        """Update drug interactions and metabolism"""
        for drug_name, drug in self.molecules.items():
            if drug.molecule_type == MoleculeType.DRUG and drug.concentration > 0:
                # Drug metabolism
                if drug_name in ['aspirin', 'metformin', 'statin']:
                    metabolism_rate = 0.1 * dt
                    drug.concentration *= (1.0 - metabolism_rate)
                
                # Drug-protein binding
                for target, affinity in drug.binding_affinity.items():
                    if target in self.molecules:
                        binding_rate = affinity * drug.concentration * dt
                        # Simple binding model
                        bound_fraction = min(0.9, binding_rate)
                        # Update target activity
                        if target in self.enzyme_activities:
                            self.enzyme_activities[target] *= (1.0 - bound_fraction * 0.5)
    
    def _update_transport(self, dt: float):
        """Update molecular transport between compartments"""
        # Simple transport model
        transport_molecules = ['glucose', 'oxygen', 'sodium', 'potassium', 'calcium']
        
        for molecule_name in transport_molecules:
            if molecule_name in self.molecules:
                molecule = self.molecules[molecule_name]
                
                # Transport rate depends on concentration gradient
                transport_rate = 0.01 * dt
                
                # Random transport events
                if random.random() < transport_rate:
                    # Simulate transport between compartments
                    transport_amount = molecule.concentration * 0.1
                    molecule.concentration -= transport_amount
                    molecule.concentration = max(0.0, molecule.concentration)
    
    def add_drug(self, drug_name: str, dose: float):
        """Add a drug to the system"""
        if drug_name in self.molecules:
            self.molecules[drug_name].concentration += dose
            logger.info(f"Added {dose} units of {drug_name}")
    
    def remove_drug(self, drug_name: str):
        """Remove a drug from the system"""
        if drug_name in self.molecules:
            self.molecules[drug_name].concentration = 0.0
            logger.info(f"Removed {drug_name}")
    
    def get_molecule_concentration(self, molecule_name: str) -> float:
        """Get concentration of a specific molecule"""
        if molecule_name in self.molecules:
            return self.molecules[molecule_name].concentration
        return 0.0
    
    def get_metabolic_flux(self, pathway_name: str) -> float:
        """Get flux of a specific metabolic pathway"""
        if pathway_name in self.metabolic_pathways:
            return self.metabolic_pathways[pathway_name]['flux']
        return 0.0
    
    def get_enzyme_activity(self, enzyme_name: str) -> float:
        """Get activity of a specific enzyme"""
        if enzyme_name in self.enzyme_activities:
            return self.enzyme_activities[enzyme_name]
        return 0.0
    
    def get_protein_folding_state(self, protein_name: str) -> Dict[str, float]:
        """Get folding state of a specific protein"""
        if protein_name in self.protein_structures:
            return self.protein_structures[protein_name].copy()
        return {'folded_fraction': 0.0, 'stability': 0.0, 'denaturation_rate': 0.0}
    
    def export_state(self) -> Dict[str, Any]:
        """Export current molecular state"""
        return {
            'molecules': {name: {
                'name': mol.name,
                'type': mol.molecule_type.value,
                'concentration': mol.concentration,
                'molecular_weight': mol.molecular_weight,
                'charge': mol.charge,
                'solubility': mol.solubility,
                'stability': mol.stability,
                'binding_affinity': mol.binding_affinity
            } for name, mol in self.molecules.items()},
            'reactions': {name: {
                'reactants': r.reactants,
                'products': r.products,
                'rate_constant': r.rate_constant,
                'equilibrium_constant': r.equilibrium_constant,
                'activation_energy': r.activation_energy,
                'enzyme': r.enzyme
            } for name, r in self.reactions.items()},
            'metabolic_pathways': self.metabolic_pathways,
            'enzyme_activities': self.enzyme_activities,
            'protein_structures': self.protein_structures,
            'ph': self.ph,
            'ionic_strength': self.ionic_strength,
            'temperature': self.temperature
        }
    
    def import_state(self, state: Dict[str, Any]):
        """Import molecular state"""
        # Reconstruct molecules
        self.molecules = {}
        for name, mol_data in state['molecules'].items():
            molecule = Molecule(
                name=mol_data['name'],
                molecule_type=MoleculeType(mol_data['type']),
                concentration=mol_data['concentration'],
                molecular_weight=mol_data['molecular_weight'],
                charge=mol_data['charge'],
                solubility=mol_data['solubility'],
                stability=mol_data['stability'],
                binding_affinity=mol_data['binding_affinity']
            )
            self.molecules[name] = molecule
        
        # Reconstruct reactions
        self.reactions = {}
        for name, r_data in state['reactions'].items():
            reaction = Reaction(
                reactants=r_data['reactants'],
                products=r_data['products'],
                rate_constant=r_data['rate_constant'],
                equilibrium_constant=r_data['equilibrium_constant'],
                activation_energy=r_data['activation_energy'],
                enzyme=r_data['enzyme']
            )
            self.reactions[name] = reaction
        
        # Import other state
        self.metabolic_pathways = state['metabolic_pathways']
        self.enzyme_activities = state['enzyme_activities']
        self.protein_structures = state['protein_structures']
        self.ph = state['ph']
        self.ionic_strength = state['ionic_strength']
        self.temperature = state['temperature']