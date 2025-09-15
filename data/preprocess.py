"""
Data Preprocessor - Preprocess and integrate biological data

This module provides functionality to preprocess and integrate data from
various biological databases and sources for the Human Digital Twin Engine.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import json
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import networkx as nx

from .loaders.human_cell_atlas_loader import HumanCellAtlasLoader
from .loaders.uniprot_loader import UniProtLoader
from .loaders.pdb_loader import PDBLoader

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Data preprocessor for biological data integration.
    
    This class provides functionality to preprocess and integrate data from
    various biological databases and sources for the Human Digital Twin Engine.
    """
    
    def __init__(self, data_dir: str = "data", cache_dir: str = "cache"):
        """
        Initialize data preprocessor.
        
        Args:
            data_dir: Directory to store data
            cache_dir: Directory to store processed data
        """
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data loaders
        self.hca_loader = HumanCellAtlasLoader(str(self.data_dir / "hca"), str(self.cache_dir / "hca"))
        self.uniprot_loader = UniProtLoader(str(self.data_dir / "uniprot"), str(self.cache_dir / "uniprot"))
        self.pdb_loader = PDBLoader(str(self.data_dir / "pdb"), str(self.cache_dir / "pdb"))
        
        # Preprocessing tools
        self.scalers = {}
        self.feature_selectors = {}
        self.dimension_reducers = {}
        
        # Integrated data
        self.integrated_data = {}
        
        logger.info("Initialized data preprocessor")
    
    def preprocess_expression_data(self, expression_data: np.ndarray, 
                                 method: str = 'standard') -> np.ndarray:
        """
        Preprocess gene expression data.
        
        Args:
            expression_data: Raw expression data
            method: Preprocessing method ('standard', 'minmax', 'log')
            
        Returns:
            Preprocessed expression data
        """
        try:
            if method == 'standard':
                scaler = StandardScaler()
                processed_data = scaler.fit_transform(expression_data)
            elif method == 'minmax':
                scaler = MinMaxScaler()
                processed_data = scaler.fit_transform(expression_data)
            elif method == 'log':
                # Log transform with pseudocount
                processed_data = np.log1p(expression_data)
            else:
                processed_data = expression_data.copy()
            
            # Store scaler for later use
            self.scalers['expression'] = scaler
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error preprocessing expression data: {e}")
            return expression_data
    
    def preprocess_protein_data(self, protein_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess protein data.
        
        Args:
            protein_data: Raw protein data
            
        Returns:
            Preprocessed protein data
        """
        try:
            processed_data = {}
            
            # Process protein sequences
            if 'sequences' in protein_data:
                sequences = protein_data['sequences']
                # Convert sequences to numerical features
                processed_data['sequence_features'] = self._sequence_to_features(sequences)
            
            # Process protein properties
            if 'properties' in protein_data:
                properties = protein_data['properties']
                # Normalize properties
                scaler = StandardScaler()
                processed_data['normalized_properties'] = scaler.fit_transform(properties)
                self.scalers['protein_properties'] = scaler
            
            # Process protein interactions
            if 'interactions' in protein_data:
                interactions = protein_data['interactions']
                processed_data['interaction_network'] = self._build_interaction_network(interactions)
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error preprocessing protein data: {e}")
            return protein_data
    
    def _sequence_to_features(self, sequences: List[str]) -> np.ndarray:
        """Convert protein sequences to numerical features"""
        features = []
        
        for sequence in sequences:
            # Basic sequence features
            seq_features = [
                len(sequence),  # Length
                sequence.count('A') / len(sequence),  # Amino acid composition
                sequence.count('R') / len(sequence),
                sequence.count('N') / len(sequence),
                sequence.count('D') / len(sequence),
                sequence.count('C') / len(sequence),
                sequence.count('Q') / len(sequence),
                sequence.count('E') / len(sequence),
                sequence.count('G') / len(sequence),
                sequence.count('H') / len(sequence),
                sequence.count('I') / len(sequence),
                sequence.count('L') / len(sequence),
                sequence.count('K') / len(sequence),
                sequence.count('M') / len(sequence),
                sequence.count('F') / len(sequence),
                sequence.count('P') / len(sequence),
                sequence.count('S') / len(sequence),
                sequence.count('T') / len(sequence),
                sequence.count('W') / len(sequence),
                sequence.count('Y') / len(sequence),
                sequence.count('V') / len(sequence)
            ]
            features.append(seq_features)
        
        return np.array(features)
    
    def _build_interaction_network(self, interactions: List[Dict[str, Any]]) -> nx.Graph:
        """Build protein interaction network"""
        G = nx.Graph()
        
        for interaction in interactions:
            protein1 = interaction.get('protein1')
            protein2 = interaction.get('protein2')
            confidence = interaction.get('confidence', 1.0)
            
            if protein1 and protein2:
                G.add_edge(protein1, protein2, weight=confidence)
        
        return G
    
    def integrate_multiomics_data(self, data_sources: Dict[str, Any]) -> Dict[str, Any]:
        """
        Integrate multi-omics data from different sources.
        
        Args:
            data_sources: Dictionary of data from different sources
            
        Returns:
            Integrated data
        """
        try:
            integrated_data = {}
            
            # Integrate genomics data
            if 'genomics' in data_sources:
                genomics_data = data_sources['genomics']
                integrated_data['genomics'] = self._integrate_genomics_data(genomics_data)
            
            # Integrate transcriptomics data
            if 'transcriptomics' in data_sources:
                transcriptomics_data = data_sources['transcriptomics']
                integrated_data['transcriptomics'] = self._integrate_transcriptomics_data(transcriptomics_data)
            
            # Integrate proteomics data
            if 'proteomics' in data_sources:
                proteomics_data = data_sources['proteomics']
                integrated_data['proteomics'] = self._integrate_proteomics_data(proteomics_data)
            
            # Integrate metabolomics data
            if 'metabolomics' in data_sources:
                metabolomics_data = data_sources['metabolomics']
                integrated_data['metabolomics'] = self._integrate_metabolomics_data(metabolomics_data)
            
            # Create cross-omics correlations
            integrated_data['cross_omics_correlations'] = self._calculate_cross_omics_correlations(integrated_data)
            
            return integrated_data
            
        except Exception as e:
            logger.error(f"Error integrating multi-omics data: {e}")
            return {}
    
    def _integrate_genomics_data(self, genomics_data: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate genomics data"""
        integrated = {}
        
        # Process gene variants
        if 'variants' in genomics_data:
            variants = genomics_data['variants']
            integrated['variant_features'] = self._process_variants(variants)
        
        # Process gene expression
        if 'expression' in genomics_data:
            expression = genomics_data['expression']
            integrated['expression_features'] = self.preprocess_expression_data(expression)
        
        return integrated
    
    def _integrate_transcriptomics_data(self, transcriptomics_data: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate transcriptomics data"""
        integrated = {}
        
        # Process RNA-seq data
        if 'rna_seq' in transcriptomics_data:
            rna_seq = transcriptomics_data['rna_seq']
            integrated['rna_seq_features'] = self.preprocess_expression_data(rna_seq)
        
        # Process microRNA data
        if 'mirna' in transcriptomics_data:
            mirna = transcriptomics_data['mirna']
            integrated['mirna_features'] = self.preprocess_expression_data(mirna)
        
        return integrated
    
    def _integrate_proteomics_data(self, proteomics_data: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate proteomics data"""
        integrated = {}
        
        # Process protein abundance
        if 'abundance' in proteomics_data:
            abundance = proteomics_data['abundance']
            integrated['abundance_features'] = self.preprocess_expression_data(abundance)
        
        # Process protein modifications
        if 'modifications' in proteomics_data:
            modifications = proteomics_data['modifications']
            integrated['modification_features'] = self._process_modifications(modifications)
        
        return integrated
    
    def _integrate_metabolomics_data(self, metabolomics_data: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate metabolomics data"""
        integrated = {}
        
        # Process metabolite levels
        if 'metabolites' in metabolomics_data:
            metabolites = metabolomics_data['metabolites']
            integrated['metabolite_features'] = self.preprocess_expression_data(metabolites)
        
        return integrated
    
    def _process_variants(self, variants: List[Dict[str, Any]]) -> np.ndarray:
        """Process genetic variants"""
        features = []
        
        for variant in variants:
            variant_features = [
                variant.get('position', 0),
                len(variant.get('ref_allele', '')),
                len(variant.get('alt_allele', '')),
                1.0 if variant.get('is_snp', False) else 0.0,
                variant.get('quality_score', 0.0)
            ]
            features.append(variant_features)
        
        return np.array(features)
    
    def _process_modifications(self, modifications: List[Dict[str, Any]]) -> np.ndarray:
        """Process protein modifications"""
        features = []
        
        for modification in modifications:
            modification_features = [
                modification.get('position', 0),
                1.0 if modification.get('type') == 'phosphorylation' else 0.0,
                1.0 if modification.get('type') == 'acetylation' else 0.0,
                1.0 if modification.get('type') == 'ubiquitination' else 0.0,
                modification.get('confidence', 0.0)
            ]
            features.append(modification_features)
        
        return np.array(features)
    
    def _calculate_cross_omics_correlations(self, integrated_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate correlations between different omics types"""
        correlations = {}
        
        # Get feature matrices
        feature_matrices = {}
        for omics_type, data in integrated_data.items():
            if omics_type != 'cross_omics_correlations':
                for feature_type, features in data.items():
                    if isinstance(features, np.ndarray):
                        feature_matrices[f"{omics_type}_{feature_type}"] = features
        
        # Calculate pairwise correlations
        feature_names = list(feature_matrices.keys())
        for i, name1 in enumerate(feature_names):
            for j, name2 in enumerate(feature_names[i+1:], i+1):
                matrix1 = feature_matrices[name1]
                matrix2 = feature_matrices[name2]
                
                # Flatten matrices and calculate correlation
                if matrix1.size > 0 and matrix2.size > 0:
                    corr = np.corrcoef(matrix1.flatten(), matrix2.flatten())[0, 1]
                    if not np.isnan(corr):
                        correlations[f"{name1}_vs_{name2}"] = float(corr)
        
        return correlations
    
    def create_knowledge_graph(self, integrated_data: Dict[str, Any]) -> nx.Graph:
        """
        Create a knowledge graph from integrated data.
        
        Args:
            integrated_data: Integrated multi-omics data
            
        Returns:
            Knowledge graph
        """
        try:
            G = nx.Graph()
            
            # Add nodes for different entities
            if 'genomics' in integrated_data:
                G.add_node('genome', type='system', level='molecular')
            
            if 'transcriptomics' in integrated_data:
                G.add_node('transcriptome', type='system', level='molecular')
            
            if 'proteomics' in integrated_data:
                G.add_node('proteome', type='system', level='molecular')
            
            if 'metabolomics' in integrated_data:
                G.add_node('metabolome', type='system', level='molecular')
            
            # Add edges based on correlations
            if 'cross_omics_correlations' in integrated_data:
                correlations = integrated_data['cross_omics_correlations']
                for correlation_name, correlation_value in correlations.items():
                    if abs(correlation_value) > 0.3:  # Threshold for significant correlation
                        parts = correlation_name.split('_vs_')
                        if len(parts) == 2:
                            source = parts[0].split('_')[0]  # Extract omics type
                            target = parts[1].split('_')[0]  # Extract omics type
                            G.add_edge(source, target, correlation=correlation_value)
            
            return G
            
        except Exception as e:
            logger.error(f"Error creating knowledge graph: {e}")
            return nx.Graph()
    
    def save_processed_data(self, data: Dict[str, Any], filename: str):
        """
        Save processed data to file.
        
        Args:
            data: Processed data
            filename: Output filename
        """
        try:
            output_file = self.cache_dir / filename
            
            if filename.endswith('.json'):
                with open(output_file, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
            elif filename.endswith('.pkl'):
                with open(output_file, 'wb') as f:
                    pickle.dump(data, f)
            else:
                logger.error(f"Unsupported file format: {filename}")
                return
            
            logger.info(f"Saved processed data to {output_file}")
            
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
    
    def load_processed_data(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Load processed data from file.
        
        Args:
            filename: Input filename
            
        Returns:
            Processed data or None if not found
        """
        try:
            input_file = self.cache_dir / filename
            
            if not input_file.exists():
                return None
            
            if filename.endswith('.json'):
                with open(input_file, 'r') as f:
                    data = json.load(f)
            elif filename.endswith('.pkl'):
                with open(input_file, 'rb') as f:
                    data = pickle.load(f)
            else:
                logger.error(f"Unsupported file format: {filename}")
                return None
            
            logger.info(f"Loaded processed data from {input_file}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading processed data: {e}")
            return None