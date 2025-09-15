"""
UniProt Loader - Load and process UniProt protein data

This module provides functionality to load and process data from UniProt,
including protein sequences, annotations, functional information, and
post-translational modifications.
"""

import pandas as pd
import numpy as np
import requests
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import xml.etree.ElementTree as ET
from urllib.parse import urljoin
import time
import gzip
import io

logger = logging.getLogger(__name__)

class UniProtLoader:
    """
    Loader for UniProt protein data.
    
    This class provides functionality to download, load, and process data from
    UniProt, including protein sequences, annotations, functional information,
    and post-translational modifications.
    """
    
    def __init__(self, data_dir: str = "data/uniprot", cache_dir: str = "cache/uniprot"):
        """
        Initialize UniProt loader.
        
        Args:
            data_dir: Directory to store downloaded data
            cache_dir: Directory to store processed/cached data
        """
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # UniProt API endpoints
        self.base_url = "https://www.uniprot.org"
        self.rest_url = "https://rest.uniprot.org"
        
        # Data cache
        self.protein_database = {}
        self.sequence_database = {}
        
        logger.info("Initialized UniProt loader")
    
    def search_proteins(self, query: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Search for proteins in UniProt.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of protein search results
        """
        try:
            # Mock search results - in real implementation, query UniProt API
            mock_results = [
                {
                    'accession': 'P04637',
                    'name': 'Cellular tumor antigen p53',
                    'gene_names': ['TP53', 'TRP53'],
                    'organism': 'Homo sapiens',
                    'length': 393,
                    'molecular_weight': 43653,
                    'function': 'Acts as a tumor suppressor in many tumor types',
                    'keywords': ['DNA-binding', 'Transcription regulation', 'Tumor suppressor'],
                    'go_terms': ['GO:0005634', 'GO:0003700', 'GO:0006915'],
                    'pathways': ['KEGG:04110', 'Reactome:R-HSA-5633007']
                },
                {
                    'accession': 'P01106',
                    'name': 'Myc proto-oncogene protein',
                    'gene_names': ['MYC', 'bHLHe39'],
                    'organism': 'Homo sapiens',
                    'length': 439,
                    'molecular_weight': 49434,
                    'function': 'Transcription factor that binds DNA in a non-specific manner',
                    'keywords': ['DNA-binding', 'Transcription regulation', 'Oncogene'],
                    'go_terms': ['GO:0005634', 'GO:0003700', 'GO:0007049'],
                    'pathways': ['KEGG:04110', 'Reactome:R-HSA-212436']
                },
                {
                    'accession': 'P38398',
                    'name': 'Breast cancer type 1 susceptibility protein',
                    'gene_names': ['BRCA1', 'BRCC1'],
                    'organism': 'Homo sapiens',
                    'length': 1863,
                    'molecular_weight': 207691,
                    'function': 'E3 ubiquitin-protein ligase that specifically mediates the formation of Lys-6-linked polyubiquitin chains',
                    'keywords': ['DNA repair', 'Ubiquitin ligase', 'Tumor suppressor'],
                    'go_terms': ['GO:0005634', 'GO:0006281', 'GO:0004842'],
                    'pathways': ['KEGG:03440', 'Reactome:R-HSA-73857']
                }
            ]
            
            # Filter results based on query
            filtered_results = []
            query_lower = query.lower()
            
            for result in mock_results:
                if (query_lower in result['name'].lower() or 
                    query_lower in result['accession'].lower() or
                    any(query_lower in gene.lower() for gene in result['gene_names'])):
                    filtered_results.append(result)
            
            return filtered_results[:limit]
            
        except Exception as e:
            logger.error(f"Error searching proteins: {e}")
            return []
    
    def get_protein_info(self, accession: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information for a specific protein.
        
        Args:
            accession: UniProt accession number
            
        Returns:
            Detailed protein information or None if not found
        """
        try:
            # Check cache first
            if accession in self.protein_database:
                return self.protein_database[accession]
            
            # Mock protein data - in real implementation, fetch from UniProt API
            mock_protein_data = {
                'P04637': {
                    'accession': 'P04637',
                    'name': 'Cellular tumor antigen p53',
                    'gene_names': ['TP53', 'TRP53'],
                    'organism': 'Homo sapiens',
                    'taxonomy_id': 9606,
                    'length': 393,
                    'molecular_weight': 43653,
                    'function': 'Acts as a tumor suppressor in many tumor types; induces growth arrest or apoptosis depending on the physiological circumstances and cell type',
                    'keywords': ['DNA-binding', 'Transcription regulation', 'Tumor suppressor', 'Apoptosis'],
                    'go_terms': {
                        'GO:0005634': 'nucleus',
                        'GO:0003700': 'DNA-binding transcription factor activity',
                        'GO:0006915': 'apoptotic process',
                        'GO:0007049': 'cell cycle',
                        'GO:0006281': 'DNA repair'
                    },
                    'pathways': {
                        'KEGG:04110': 'Cell cycle',
                        'KEGG:04115': 'p53 signaling pathway',
                        'Reactome:R-HSA-5633007': 'TP53 Regulates Transcription of Cell Death Genes'
                    },
                    'domains': [
                        {'name': 'DNA-binding domain', 'start': 102, 'end': 292},
                        {'name': 'Transactivation domain', 'start': 1, 'end': 42},
                        {'name': 'Tetramerization domain', 'start': 325, 'end': 355}
                    ],
                    'post_translational_modifications': [
                        {'type': 'phosphorylation', 'position': 15, 'residue': 'S', 'description': 'Phosphorylated by ATM'},
                        {'type': 'acetylation', 'position': 120, 'residue': 'K', 'description': 'Acetylated by p300'},
                        {'type': 'ubiquitination', 'position': 386, 'residue': 'K', 'description': 'Ubiquitinated by MDM2'}
                    ],
                    'interactions': [
                        {'partner': 'MDM2', 'type': 'physical', 'confidence': 'high'},
                        {'partner': 'ATM', 'type': 'physical', 'confidence': 'high'},
                        {'partner': 'BRCA1', 'type': 'physical', 'confidence': 'medium'}
                    ],
                    'diseases': [
                        {'name': 'Li-Fraumeni syndrome', 'omim': '151623'},
                        {'name': 'Breast cancer', 'omim': '114480'},
                        {'name': 'Colorectal cancer', 'omim': '114500'}
                    ]
                },
                'P01106': {
                    'accession': 'P01106',
                    'name': 'Myc proto-oncogene protein',
                    'gene_names': ['MYC', 'bHLHe39'],
                    'organism': 'Homo sapiens',
                    'taxonomy_id': 9606,
                    'length': 439,
                    'molecular_weight': 49434,
                    'function': 'Transcription factor that binds DNA in a non-specific manner, yet also specifically recognizes the core sequence 5\'-CAC[GA]TG-3\'',
                    'keywords': ['DNA-binding', 'Transcription regulation', 'Oncogene', 'Helix-loop-helix'],
                    'go_terms': {
                        'GO:0005634': 'nucleus',
                        'GO:0003700': 'DNA-binding transcription factor activity',
                        'GO:0007049': 'cell cycle',
                        'GO:0006355': 'regulation of transcription, DNA-templated'
                    },
                    'pathways': {
                        'KEGG:04110': 'Cell cycle',
                        'KEGG:04010': 'MAPK signaling pathway',
                        'Reactome:R-HSA-212436': 'Transcriptional Regulation by E2F6'
                    },
                    'domains': [
                        {'name': 'Helix-loop-helix domain', 'start': 355, 'end': 413},
                        {'name': 'Leucine zipper domain', 'start': 413, 'end': 439}
                    ],
                    'post_translational_modifications': [
                        {'type': 'phosphorylation', 'position': 58, 'residue': 'T', 'description': 'Phosphorylated by GSK3'},
                        {'type': 'acetylation', 'position': 143, 'residue': 'K', 'description': 'Acetylated by p300'}
                    ],
                    'interactions': [
                        {'partner': 'MAX', 'type': 'physical', 'confidence': 'high'},
                        {'partner': 'MAD', 'type': 'physical', 'confidence': 'high'},
                        {'partner': 'MXI1', 'type': 'physical', 'confidence': 'medium'}
                    ],
                    'diseases': [
                        {'name': 'Burkitt lymphoma', 'omim': '113970'},
                        {'name': 'Neuroblastoma', 'omim': '256700'}
                    ]
                }
            }
            
            protein_info = mock_protein_data.get(accession)
            if protein_info:
                # Cache the result
                self.protein_database[accession] = protein_info
                return protein_info
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting protein info for {accession}: {e}")
            return None
    
    def get_protein_sequence(self, accession: str) -> Optional[str]:
        """
        Get protein sequence for a specific accession.
        
        Args:
            accession: UniProt accession number
            
        Returns:
            Protein sequence or None if not found
        """
        try:
            # Check cache first
            if accession in self.sequence_database:
                return self.sequence_database[accession]
            
            # Mock sequences - in real implementation, fetch from UniProt API
            mock_sequences = {
                'P04637': 'MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD',
                'P01106': 'MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD',
                'P38398': 'MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD'
            }
            
            sequence = mock_sequences.get(accession)
            if sequence:
                # Cache the result
                self.sequence_database[accession] = sequence
                return sequence
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting protein sequence for {accession}: {e}")
            return None
    
    def get_protein_family(self, accession: str) -> Optional[Dict[str, Any]]:
        """
        Get protein family information.
        
        Args:
            accession: UniProt accession number
            
        Returns:
            Protein family information or None if not found
        """
        try:
            protein_info = self.get_protein_info(accession)
            if not protein_info:
                return None
            
            # Mock family data - in real implementation, fetch from UniProt
            family_data = {
                'P04637': {
                    'family': 'p53 family',
                    'subfamily': 'p53',
                    'pfam_domains': ['PF00870', 'PF08563'],
                    'interpro_domains': ['IPR002117', 'IPR012346'],
                    'smart_domains': ['SM00291'],
                    'superfamily': 'DNA-binding protein'
                },
                'P01106': {
                    'family': 'Myc family',
                    'subfamily': 'c-Myc',
                    'pfam_domains': ['PF00010', 'PF00130'],
                    'interpro_domains': ['IPR001092', 'IPR011598'],
                    'smart_domains': ['SM00353'],
                    'superfamily': 'Helix-loop-helix DNA-binding protein'
                },
                'P38398': {
                    'family': 'BRCA1 family',
                    'subfamily': 'BRCA1',
                    'pfam_domains': ['PF00533', 'PF00643'],
                    'interpro_domains': ['IPR001357', 'IPR003890'],
                    'smart_domains': ['SM00292'],
                    'superfamily': 'E3 ubiquitin ligase'
                }
            }
            
            return family_data.get(accession)
            
        except Exception as e:
            logger.error(f"Error getting protein family for {accession}: {e}")
            return None
    
    def get_protein_expression(self, accession: str) -> Optional[Dict[str, Any]]:
        """
        Get protein expression data.
        
        Args:
            accession: UniProt accession number
            
        Returns:
            Protein expression data or None if not found
        """
        try:
            # Mock expression data - in real implementation, fetch from UniProt
            expression_data = {
                'P04637': {
                    'tissue_expression': {
                        'liver': 'high',
                        'lung': 'high',
                        'breast': 'high',
                        'brain': 'medium',
                        'heart': 'medium',
                        'kidney': 'medium'
                    },
                    'cell_line_expression': {
                        'HeLa': 'high',
                        'HEK293': 'medium',
                        'MCF7': 'high',
                        'A549': 'high'
                    },
                    'developmental_stage': {
                        'embryo': 'high',
                        'adult': 'high',
                        'aged': 'medium'
                    }
                },
                'P01106': {
                    'tissue_expression': {
                        'brain': 'high',
                        'liver': 'medium',
                        'muscle': 'medium',
                        'heart': 'low',
                        'kidney': 'low'
                    },
                    'cell_line_expression': {
                        'HeLa': 'high',
                        'HEK293': 'medium',
                        'MCF7': 'high',
                        'A549': 'medium'
                    },
                    'developmental_stage': {
                        'embryo': 'high',
                        'adult': 'medium',
                        'aged': 'low'
                    }
                }
            }
            
            return expression_data.get(accession)
            
        except Exception as e:
            logger.error(f"Error getting protein expression for {accession}: {e}")
            return None
    
    def get_protein_interactions(self, accession: str) -> List[Dict[str, Any]]:
        """
        Get protein-protein interactions.
        
        Args:
            accession: UniProt accession number
            
        Returns:
            List of protein interactions
        """
        try:
            protein_info = self.get_protein_info(accession)
            if not protein_info:
                return []
            
            # Mock interaction data - in real implementation, fetch from UniProt
            interaction_data = {
                'P04637': [
                    {
                        'partner_accession': 'Q00987',
                        'partner_name': 'MDM2',
                        'interaction_type': 'physical',
                        'confidence': 'high',
                        'method': 'two hybrid',
                        'pubmed_id': '12345678'
                    },
                    {
                        'partner_accession': 'Q13315',
                        'partner_name': 'ATM',
                        'interaction_type': 'physical',
                        'confidence': 'high',
                        'method': 'coimmunoprecipitation',
                        'pubmed_id': '12345679'
                    },
                    {
                        'partner_accession': 'P38398',
                        'partner_name': 'BRCA1',
                        'interaction_type': 'physical',
                        'confidence': 'medium',
                        'method': 'two hybrid',
                        'pubmed_id': '12345680'
                    }
                ],
                'P01106': [
                    {
                        'partner_accession': 'P61244',
                        'partner_name': 'MAX',
                        'interaction_type': 'physical',
                        'confidence': 'high',
                        'method': 'two hybrid',
                        'pubmed_id': '12345681'
                    },
                    {
                        'partner_accession': 'Q05195',
                        'partner_name': 'MAD',
                        'interaction_type': 'physical',
                        'confidence': 'high',
                        'method': 'coimmunoprecipitation',
                        'pubmed_id': '12345682'
                    }
                ]
            }
            
            return interaction_data.get(accession, [])
            
        except Exception as e:
            logger.error(f"Error getting protein interactions for {accession}: {e}")
            return []
    
    def get_protein_diseases(self, accession: str) -> List[Dict[str, Any]]:
        """
        Get disease associations for a protein.
        
        Args:
            accession: UniProt accession number
            
        Returns:
            List of disease associations
        """
        try:
            protein_info = self.get_protein_info(accession)
            if not protein_info:
                return []
            
            # Mock disease data - in real implementation, fetch from UniProt
            disease_data = {
                'P04637': [
                    {
                        'disease_name': 'Li-Fraumeni syndrome',
                        'omim_id': '151623',
                        'association_type': 'causative',
                        'confidence': 'high',
                        'description': 'Autosomal dominant disorder characterized by early onset of multiple primary cancers'
                    },
                    {
                        'disease_name': 'Breast cancer',
                        'omim_id': '114480',
                        'association_type': 'susceptibility',
                        'confidence': 'high',
                        'description': 'Increased risk of breast cancer development'
                    }
                ],
                'P01106': [
                    {
                        'disease_name': 'Burkitt lymphoma',
                        'omim_id': '113970',
                        'association_type': 'causative',
                        'confidence': 'high',
                        'description': 'Chromosomal translocation involving MYC gene'
                    }
                ]
            }
            
            return disease_data.get(accession, [])
            
        except Exception as e:
            logger.error(f"Error getting protein diseases for {accession}: {e}")
            return []
    
    def export_protein_data(self, accession: str) -> Dict[str, Any]:
        """
        Export comprehensive protein data.
        
        Args:
            accession: UniProt accession number
            
        Returns:
            Comprehensive protein data
        """
        try:
            # Get all protein data
            protein_info = self.get_protein_info(accession)
            sequence = self.get_protein_sequence(accession)
            family = self.get_protein_family(accession)
            expression = self.get_protein_expression(accession)
            interactions = self.get_protein_interactions(accession)
            diseases = self.get_protein_diseases(accession)
            
            if not protein_info:
                return {}
            
            # Combine all data
            comprehensive_data = {
                'basic_info': protein_info,
                'sequence': sequence,
                'family': family,
                'expression': expression,
                'interactions': interactions,
                'diseases': diseases,
                'export_timestamp': time.time()
            }
            
            return comprehensive_data
            
        except Exception as e:
            logger.error(f"Error exporting protein data for {accession}: {e}")
            return {}
    
    def save_protein_data(self, accession: str, data: Dict[str, Any]):
        """
        Save protein data to cache.
        
        Args:
            accession: UniProt accession number
            data: Protein data to save
        """
        try:
            cache_file = self.cache_dir / f"{accession}.json"
            
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info(f"Saved protein data for {accession}")
            
        except Exception as e:
            logger.error(f"Error saving protein data: {e}")
    
    def load_protein_data(self, accession: str) -> Optional[Dict[str, Any]]:
        """
        Load protein data from cache.
        
        Args:
            accession: UniProt accession number
            
        Returns:
            Protein data or None if not found
        """
        try:
            cache_file = self.cache_dir / f"{accession}.json"
            
            if not cache_file.exists():
                return None
            
            with open(cache_file, 'r') as f:
                data = json.load(f)
            
            logger.info(f"Loaded protein data for {accession}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading protein data: {e}")
            return None