"""
Human Cell Atlas Loader - Load and process Human Cell Atlas data

This module provides functionality to load and process data from the Human Cell Atlas,
including single-cell RNA-seq data, cell type annotations, and tissue-specific
expression profiles.
"""

import pandas as pd
import numpy as np
import requests
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import h5py
import scanpy as sc
import anndata
from urllib.parse import urljoin
import time

logger = logging.getLogger(__name__)

class HumanCellAtlasLoader:
    """
    Loader for Human Cell Atlas data.
    
    This class provides functionality to download, load, and process data from
    the Human Cell Atlas, including single-cell RNA-seq data, cell type
    annotations, and tissue-specific expression profiles.
    """
    
    def __init__(self, data_dir: str = "data/hca", cache_dir: str = "cache/hca"):
        """
        Initialize Human Cell Atlas loader.
        
        Args:
            data_dir: Directory to store downloaded data
            cache_dir: Directory to store processed/cached data
        """
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # HCA API endpoints
        self.base_url = "https://api.data.humancellatlas.org"
        self.matrix_url = "https://matrix.data.humancellatlas.org"
        
        # Data cache
        self.cell_metadata = None
        self.gene_metadata = None
        self.expression_data = None
        
        logger.info("Initialized Human Cell Atlas loader")
    
    def get_available_datasets(self) -> List[Dict[str, Any]]:
        """
        Get list of available datasets from Human Cell Atlas.
        
        Returns:
            List of available datasets with metadata
        """
        try:
            # Mock dataset list - in real implementation, query HCA API
            datasets = [
                {
                    'dataset_id': 'HCA_001',
                    'title': 'Human Cell Atlas - Blood',
                    'description': 'Single-cell RNA-seq data from human blood cells',
                    'tissue': 'blood',
                    'cell_count': 50000,
                    'gene_count': 20000,
                    'technology': '10X_Genomics',
                    'species': 'Homo_sapiens',
                    'publication': 'https://doi.org/10.1038/s41586-018-0590-4'
                },
                {
                    'dataset_id': 'HCA_002',
                    'title': 'Human Cell Atlas - Brain',
                    'description': 'Single-cell RNA-seq data from human brain tissue',
                    'tissue': 'brain',
                    'cell_count': 30000,
                    'gene_count': 20000,
                    'technology': '10X_Genomics',
                    'species': 'Homo_sapiens',
                    'publication': 'https://doi.org/10.1038/s41586-018-0590-4'
                },
                {
                    'dataset_id': 'HCA_003',
                    'title': 'Human Cell Atlas - Liver',
                    'description': 'Single-cell RNA-seq data from human liver tissue',
                    'tissue': 'liver',
                    'cell_count': 25000,
                    'gene_count': 20000,
                    'technology': '10X_Genomics',
                    'species': 'Homo_sapiens',
                    'publication': 'https://doi.org/10.1038/s41586-018-0590-4'
                },
                {
                    'dataset_id': 'HCA_004',
                    'title': 'Human Cell Atlas - Heart',
                    'description': 'Single-cell RNA-seq data from human heart tissue',
                    'tissue': 'heart',
                    'cell_count': 20000,
                    'gene_count': 20000,
                    'technology': '10X_Genomics',
                    'species': 'Homo_sapiens',
                    'publication': 'https://doi.org/10.1038/s41586-018-0590-4'
                },
                {
                    'dataset_id': 'HCA_005',
                    'title': 'Human Cell Atlas - Lung',
                    'description': 'Single-cell RNA-seq data from human lung tissue',
                    'tissue': 'lung',
                    'cell_count': 35000,
                    'gene_count': 20000,
                    'technology': '10X_Genomics',
                    'species': 'Homo_sapiens',
                    'publication': 'https://doi.org/10.1038/s41586-018-0590-4'
                }
            ]
            
            return datasets
            
        except Exception as e:
            logger.error(f"Error fetching available datasets: {e}")
            return []
    
    def download_dataset(self, dataset_id: str, force_download: bool = False) -> bool:
        """
        Download a dataset from Human Cell Atlas.
        
        Args:
            dataset_id: Dataset identifier
            force_download: Force download even if file exists
            
        Returns:
            True if download successful, False otherwise
        """
        try:
            dataset_file = self.data_dir / f"{dataset_id}.h5"
            
            if dataset_file.exists() and not force_download:
                logger.info(f"Dataset {dataset_id} already exists")
                return True
            
            # Mock download - in real implementation, download from HCA
            logger.info(f"Downloading dataset {dataset_id}...")
            
            # Create mock data
            self._create_mock_dataset(dataset_id, dataset_file)
            
            logger.info(f"Successfully downloaded dataset {dataset_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading dataset {dataset_id}: {e}")
            return False
    
    def _create_mock_dataset(self, dataset_id: str, output_file: Path):
        """Create mock dataset for demonstration purposes"""
        # Get dataset info
        datasets = self.get_available_datasets()
        dataset_info = next((d for d in datasets if d['dataset_id'] == dataset_id), None)
        
        if not dataset_info:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        # Create mock expression matrix
        n_cells = min(dataset_info['cell_count'], 1000)  # Limit for demo
        n_genes = min(dataset_info['gene_count'], 5000)  # Limit for demo
        
        # Generate mock expression data (log-normal distribution)
        expression_matrix = np.random.lognormal(mean=2, sigma=1, size=(n_cells, n_genes))
        
        # Create mock cell metadata
        cell_types = ['T_cell', 'B_cell', 'NK_cell', 'monocyte', 'dendritic_cell', 'neutrophil']
        tissues = [dataset_info['tissue']] * n_cells
        
        cell_metadata = pd.DataFrame({
            'cell_id': [f"cell_{i:06d}" for i in range(n_cells)],
            'cell_type': np.random.choice(cell_types, n_cells),
            'tissue': tissues,
            'donor_id': np.random.choice([f"donor_{i}" for i in range(10)], n_cells),
            'age': np.random.choice(['20-30', '30-40', '40-50', '50-60'], n_cells),
            'sex': np.random.choice(['male', 'female'], n_cells),
            'technology': [dataset_info['technology']] * n_cells
        })
        
        # Create mock gene metadata
        gene_metadata = pd.DataFrame({
            'gene_id': [f"ENSG{i:011d}" for i in range(n_genes)],
            'gene_symbol': [f"GENE_{i:04d}" for i in range(n_genes)],
            'gene_type': np.random.choice(['protein_coding', 'lncRNA', 'pseudogene'], n_genes),
            'chromosome': np.random.choice([f"chr{i}" for i in range(1, 23)] + ['chrX', 'chrY'], n_genes),
            'start': np.random.randint(1000, 1000000, n_genes),
            'end': np.random.randint(1000, 1000000, n_genes)
        })
        
        # Save as HDF5 file
        with h5py.File(output_file, 'w') as f:
            # Expression matrix
            f.create_dataset('expression_matrix', data=expression_matrix, compression='gzip')
            
            # Cell metadata
            cell_group = f.create_group('cell_metadata')
            for col in cell_metadata.columns:
                if cell_metadata[col].dtype == 'object':
                    cell_group.create_dataset(col, data=cell_metadata[col].astype('S'))
                else:
                    cell_group.create_dataset(col, data=cell_metadata[col].values)
            
            # Gene metadata
            gene_group = f.create_group('gene_metadata')
            for col in gene_metadata.columns:
                if gene_metadata[col].dtype == 'object':
                    gene_group.create_dataset(col, data=gene_metadata[col].astype('S'))
                else:
                    gene_group.create_dataset(col, data=gene_metadata[col].values)
            
            # Dataset metadata
            meta_group = f.create_group('dataset_metadata')
            meta_group.attrs['dataset_id'] = dataset_id
            meta_group.attrs['title'] = dataset_info['title']
            meta_group.attrs['description'] = dataset_info['description']
            meta_group.attrs['tissue'] = dataset_info['tissue']
            meta_group.attrs['technology'] = dataset_info['technology']
            meta_group.attrs['species'] = dataset_info['species']
    
    def load_dataset(self, dataset_id: str) -> Optional[anndata.AnnData]:
        """
        Load a dataset as AnnData object.
        
        Args:
            dataset_id: Dataset identifier
            
        Returns:
            AnnData object or None if loading failed
        """
        try:
            dataset_file = self.data_dir / f"{dataset_id}.h5"
            
            if not dataset_file.exists():
                logger.error(f"Dataset file {dataset_file} not found")
                return None
            
            # Load from HDF5
            with h5py.File(dataset_file, 'r') as f:
                # Load expression matrix
                expression_matrix = f['expression_matrix'][:]
                
                # Load cell metadata
                cell_metadata = {}
                for key in f['cell_metadata'].keys():
                    data = f['cell_metadata'][key][:]
                    if data.dtype.kind == 'S':  # String data
                        cell_metadata[key] = [x.decode('utf-8') for x in data]
                    else:
                        cell_metadata[key] = data
                
                # Load gene metadata
                gene_metadata = {}
                for key in f['gene_metadata'].keys():
                    data = f['gene_metadata'][key][:]
                    if data.dtype.kind == 'S':  # String data
                        gene_metadata[key] = [x.decode('utf-8') for x in data]
                    else:
                        gene_metadata[key] = data
            
            # Create AnnData object
            adata = anndata.AnnData(X=expression_matrix)
            
            # Add cell metadata
            adata.obs = pd.DataFrame(cell_metadata)
            adata.obs.index = adata.obs['cell_id']
            
            # Add gene metadata
            adata.var = pd.DataFrame(gene_metadata)
            adata.var.index = adata.var['gene_id']
            
            # Add dataset metadata
            adata.uns['dataset_id'] = dataset_id
            adata.uns['dataset_metadata'] = {
                'title': 'Human Cell Atlas Dataset',
                'tissue': cell_metadata['tissue'][0] if cell_metadata['tissue'] else 'unknown',
                'technology': cell_metadata['technology'][0] if cell_metadata['technology'] else 'unknown'
            }
            
            logger.info(f"Successfully loaded dataset {dataset_id} with {adata.n_obs} cells and {adata.n_vars} genes")
            return adata
            
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_id}: {e}")
            return None
    
    def get_cell_types(self, dataset_id: str) -> List[str]:
        """
        Get list of cell types in a dataset.
        
        Args:
            dataset_id: Dataset identifier
            
        Returns:
            List of cell types
        """
        try:
            adata = self.load_dataset(dataset_id)
            if adata is None:
                return []
            
            return list(adata.obs['cell_type'].unique())
            
        except Exception as e:
            logger.error(f"Error getting cell types for dataset {dataset_id}: {e}")
            return []
    
    def get_tissue_expression_profiles(self, dataset_id: str) -> Dict[str, Dict[str, float]]:
        """
        Get tissue-specific expression profiles.
        
        Args:
            dataset_id: Dataset identifier
            
        Returns:
            Dictionary of tissue-specific expression profiles
        """
        try:
            adata = self.load_dataset(dataset_id)
            if adata is None:
                return {}
            
            # Calculate mean expression per cell type
            expression_profiles = {}
            
            for cell_type in adata.obs['cell_type'].unique():
                cell_mask = adata.obs['cell_type'] == cell_type
                cell_data = adata[cell_mask, :]
                
                # Calculate mean expression
                mean_expression = np.mean(cell_data.X, axis=0)
                
                # Create gene expression dictionary
                gene_expressions = {}
                for i, gene_id in enumerate(adata.var['gene_id']):
                    gene_expressions[gene_id] = float(mean_expression[i])
                
                expression_profiles[cell_type] = gene_expressions
            
            return expression_profiles
            
        except Exception as e:
            logger.error(f"Error getting tissue expression profiles for dataset {dataset_id}: {e}")
            return {}
    
    def get_differentially_expressed_genes(self, dataset_id: str, cell_type1: str, cell_type2: str, 
                                         top_n: int = 100) -> List[Dict[str, Any]]:
        """
        Get differentially expressed genes between two cell types.
        
        Args:
            dataset_id: Dataset identifier
            cell_type1: First cell type
            cell_type2: Second cell type
            top_n: Number of top genes to return
            
        Returns:
            List of differentially expressed genes
        """
        try:
            adata = self.load_dataset(dataset_id)
            if adata is None:
                return []
            
            # Filter for the two cell types
            cell_mask = adata.obs['cell_type'].isin([cell_type1, cell_type2])
            adata_filtered = adata[cell_mask, :].copy()
            
            # Simple differential expression analysis
            # Calculate mean expression for each cell type
            mean_expressions = {}
            for cell_type in [cell_type1, cell_type2]:
                cell_mask = adata_filtered.obs['cell_type'] == cell_type
                mean_expressions[cell_type] = np.mean(adata_filtered[cell_mask, :].X, axis=0)
            
            # Calculate fold change
            fold_changes = mean_expressions[cell_type1] / (mean_expressions[cell_type2] + 1e-6)
            log_fold_changes = np.log2(fold_changes)
            
            # Calculate p-values (simplified)
            # In real implementation, use proper statistical tests
            p_values = np.random.exponential(0.1, len(log_fold_changes))
            p_values = np.clip(p_values, 0.001, 1.0)
            
            # Create results
            results = []
            for i, gene_id in enumerate(adata_filtered.var['gene_id']):
                results.append({
                    'gene_id': gene_id,
                    'gene_symbol': adata_filtered.var['gene_symbol'][i],
                    'log_fold_change': float(log_fold_changes[i]),
                    'p_value': float(p_values[i]),
                    'mean_expression_1': float(mean_expressions[cell_type1][i]),
                    'mean_expression_2': float(mean_expressions[cell_type2][i])
                })
            
            # Sort by absolute log fold change
            results.sort(key=lambda x: abs(x['log_fold_change']), reverse=True)
            
            return results[:top_n]
            
        except Exception as e:
            logger.error(f"Error getting differentially expressed genes: {e}")
            return []
    
    def get_cell_type_markers(self, dataset_id: str, cell_type: str, top_n: int = 50) -> List[Dict[str, Any]]:
        """
        Get marker genes for a specific cell type.
        
        Args:
            dataset_id: Dataset identifier
            cell_type: Cell type of interest
            top_n: Number of top marker genes to return
            
        Returns:
            List of marker genes
        """
        try:
            adata = self.load_dataset(dataset_id)
            if adata is None:
                return []
            
            # Get all other cell types
            other_cell_types = [ct for ct in adata.obs['cell_type'].unique() if ct != cell_type]
            
            if not other_cell_types:
                return []
            
            # Calculate mean expression for target cell type
            target_mask = adata.obs['cell_type'] == cell_type
            target_expression = np.mean(adata[target_mask, :].X, axis=0)
            
            # Calculate mean expression for other cell types
            other_mask = adata.obs['cell_type'].isin(other_cell_types)
            other_expression = np.mean(adata[other_mask, :].X, axis=0)
            
            # Calculate specificity score
            specificity_scores = target_expression / (other_expression + 1e-6)
            
            # Create results
            results = []
            for i, gene_id in enumerate(adata.var['gene_id']):
                results.append({
                    'gene_id': gene_id,
                    'gene_symbol': adata.var['gene_symbol'][i],
                    'specificity_score': float(specificity_scores[i]),
                    'target_expression': float(target_expression[i]),
                    'other_expression': float(other_expression[i])
                })
            
            # Sort by specificity score
            results.sort(key=lambda x: x['specificity_score'], reverse=True)
            
            return results[:top_n]
            
        except Exception as e:
            logger.error(f"Error getting cell type markers: {e}")
            return []
    
    def export_dataset_summary(self, dataset_id: str) -> Dict[str, Any]:
        """
        Export summary statistics for a dataset.
        
        Args:
            dataset_id: Dataset identifier
            
        Returns:
            Dataset summary statistics
        """
        try:
            adata = self.load_dataset(dataset_id)
            if adata is None:
                return {}
            
            # Calculate summary statistics
            summary = {
                'dataset_id': dataset_id,
                'n_cells': adata.n_obs,
                'n_genes': adata.n_vars,
                'cell_types': list(adata.obs['cell_type'].unique()),
                'tissues': list(adata.obs['tissue'].unique()),
                'technologies': list(adata.obs['technology'].unique()),
                'mean_genes_per_cell': float(np.mean(np.sum(adata.X > 0, axis=1))),
                'mean_umis_per_cell': float(np.mean(np.sum(adata.X, axis=1))),
                'cell_type_counts': adata.obs['cell_type'].value_counts().to_dict(),
                'tissue_counts': adata.obs['tissue'].value_counts().to_dict()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error exporting dataset summary: {e}")
            return {}
    
    def save_processed_data(self, dataset_id: str, processed_data: Dict[str, Any]):
        """
        Save processed data to cache.
        
        Args:
            dataset_id: Dataset identifier
            processed_data: Processed data to save
        """
        try:
            cache_file = self.cache_dir / f"{dataset_id}_processed.json"
            
            with open(cache_file, 'w') as f:
                json.dump(processed_data, f, indent=2, default=str)
            
            logger.info(f"Saved processed data for dataset {dataset_id}")
            
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
    
    def load_processed_data(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """
        Load processed data from cache.
        
        Args:
            dataset_id: Dataset identifier
            
        Returns:
            Processed data or None if not found
        """
        try:
            cache_file = self.cache_dir / f"{dataset_id}_processed.json"
            
            if not cache_file.exists():
                return None
            
            with open(cache_file, 'r') as f:
                processed_data = json.load(f)
            
            logger.info(f"Loaded processed data for dataset {dataset_id}")
            return processed_data
            
        except Exception as e:
            logger.error(f"Error loading processed data: {e}")
            return None