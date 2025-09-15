"""
Proteomics AI - Advanced AI models for protein analysis and prediction

This module implements state-of-the-art AI models for analyzing protein sequences,
predicting protein structure, function, interactions, and post-translational modifications.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import logging
from typing import Dict, List, Optional, Tuple, Any
import json
import pickle
from pathlib import Path
from Bio import SeqIO
from Bio.SeqUtils import molecular_weight
import networkx as nx

logger = logging.getLogger(__name__)

class ProteinEmbedding(nn.Module):
    """Protein sequence embedding using transformer architecture"""
    
    def __init__(self, vocab_size: int = 21, embed_dim: int = 128, num_heads: int = 8, num_layers: int = 6):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size  # 20 amino acids + padding
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1000, embed_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for protein sequence embedding
        
        Args:
            sequences: Input protein sequences [batch_size, seq_len]
            
        Returns:
            Embedded sequences [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len = sequences.shape
        
        # Embed sequences
        embedded = self.embedding(sequences)  # [batch_size, seq_len, embed_dim]
        
        # Add positional encoding
        pos_enc = self.pos_encoding[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)
        embedded = embedded + pos_enc
        
        # Apply transformer
        output = self.transformer(embedded)
        
        # Project output
        output = self.output_proj(output)
        
        return output

class ProteinStructurePredictor(nn.Module):
    """Predict protein secondary and tertiary structure"""
    
    def __init__(self, embed_dim: int = 128, num_secondary_classes: int = 8, num_tertiary_classes: int = 20):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_secondary_classes = num_secondary_classes
        self.num_tertiary_classes = num_tertiary_classes
        
        # Sequence encoder
        self.sequence_encoder = ProteinEmbedding(embed_dim=embed_dim)
        
        # Secondary structure predictor
        self.secondary_predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, num_secondary_classes)
        )
        
        # Tertiary structure predictor (contact map)
        self.tertiary_predictor = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, sequences: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict protein structure
        
        Args:
            sequences: Protein sequences [batch_size, seq_len]
            
        Returns:
            Secondary structure predictions [batch_size, seq_len, num_secondary_classes]
            Contact map predictions [batch_size, seq_len, seq_len]
        """
        # Encode sequences
        seq_encoded = self.sequence_encoder(sequences)  # [batch_size, seq_len, embed_dim]
        
        # Predict secondary structure
        secondary_pred = self.secondary_predictor(seq_encoded)
        
        # Predict tertiary structure (contact map)
        batch_size, seq_len, embed_dim = seq_encoded.shape
        contact_map = torch.zeros(batch_size, seq_len, seq_len)
        
        for i in range(seq_len):
            for j in range(i + 5, seq_len):  # Skip nearby residues
                # Concatenate embeddings of residue pairs
                pair_embedding = torch.cat([seq_encoded[:, i, :], seq_encoded[:, j, :]], dim=1)
                contact_prob = self.tertiary_predictor(pair_embedding)
                contact_map[:, i, j] = contact_prob.squeeze()
                contact_map[:, j, i] = contact_prob.squeeze()  # Symmetric
        
        return secondary_pred, contact_map

class ProteinFunctionPredictor(nn.Module):
    """Predict protein function from sequence"""
    
    def __init__(self, embed_dim: int = 128, num_functions: int = 1000):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_functions = num_functions
        
        # Sequence encoder
        self.sequence_encoder = ProteinEmbedding(embed_dim=embed_dim)
        
        # Function predictor
        self.function_predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, num_functions),
            nn.Sigmoid()
        )
    
    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Predict protein function
        
        Args:
            sequences: Protein sequences [batch_size, seq_len]
            
        Returns:
            Function predictions [batch_size, num_functions]
        """
        # Encode sequences
        seq_encoded = self.sequence_encoder(sequences)
        
        # Global average pooling
        seq_pooled = torch.mean(seq_encoded, dim=1)
        
        # Predict function
        function_pred = self.function_predictor(seq_pooled)
        
        return function_pred

class ProteinInteractionPredictor(nn.Module):
    """Predict protein-protein interactions"""
    
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Sequence encoder
        self.sequence_encoder = ProteinEmbedding(embed_dim=embed_dim)
        
        # Interaction predictor
        self.interaction_predictor = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, protein1: torch.Tensor, protein2: torch.Tensor) -> torch.Tensor:
        """
        Predict protein-protein interaction
        
        Args:
            protein1: First protein sequence [batch_size, seq_len]
            protein2: Second protein sequence [batch_size, seq_len]
            
        Returns:
            Interaction probability [batch_size, 1]
        """
        # Encode sequences
        seq1_encoded = self.sequence_encoder(protein1)
        seq2_encoded = self.sequence_encoder(protein2)
        
        # Global average pooling
        seq1_pooled = torch.mean(seq1_encoded, dim=1)
        seq2_pooled = torch.mean(seq2_encoded, dim=1)
        
        # Combine features
        combined = torch.cat([seq1_pooled, seq2_pooled], dim=1)
        
        # Predict interaction
        interaction_prob = self.interaction_predictor(combined)
        
        return interaction_prob

class ProteomicsAI:
    """
    Advanced AI system for proteomic analysis and prediction.
    
    This class integrates multiple AI models for comprehensive protein analysis,
    including sequence embedding, structure prediction, function prediction,
    and interaction analysis.
    """
    
    def __init__(self, model_dir: str = "models/proteomics"):
        """
        Initialize Proteomics AI system.
        
        Args:
            model_dir: Directory to save/load models
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize models
        self.protein_embedder = ProteinEmbedding()
        self.structure_predictor = ProteinStructurePredictor()
        self.function_predictor = ProteinFunctionPredictor()
        self.interaction_predictor = ProteinInteractionPredictor()
        
        # Protein sequence tokenizer
        self.protein_tokenizer = self._create_protein_tokenizer()
        
        # Protein database
        self.protein_database = self._load_protein_database()
        
        # Function ontology
        self.function_ontology = self._load_function_ontology()
        
        # Interaction database
        self.interaction_database = self._load_interaction_database()
        
        # Post-translational modifications
        self.ptm_database = self._load_ptm_database()
        
        logger.info("Initialized Proteomics AI system")
    
    def _create_protein_tokenizer(self) -> Dict[str, int]:
        """Create protein sequence tokenizer"""
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        tokenizer = {aa: i for i, aa in enumerate(amino_acids)}
        tokenizer['PAD'] = len(amino_acids)
        return tokenizer
    
    def _load_protein_database(self) -> Dict[str, Any]:
        """Load protein database with annotations"""
        return {
            'P04637': {  # TP53
                'name': 'Cellular tumor antigen p53',
                'sequence': 'MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD',
                'length': 393,
                'molecular_weight': 43653.0,
                'function': 'tumor_suppressor',
                'localization': 'nucleus',
                'structure': 'tetramer'
            },
            'P01106': {  # MYC
                'name': 'Myc proto-oncogene protein',
                'sequence': 'MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD',
                'length': 439,
                'molecular_weight': 49434.0,
                'function': 'transcription_factor',
                'localization': 'nucleus',
                'structure': 'dimer'
            }
        }
    
    def _load_function_ontology(self) -> Dict[str, Any]:
        """Load protein function ontology"""
        return {
            'GO:0003677': {'name': 'DNA binding', 'category': 'molecular_function'},
            'GO:0003700': {'name': 'transcription factor activity', 'category': 'molecular_function'},
            'GO:0005634': {'name': 'nucleus', 'category': 'cellular_component'},
            'GO:0007049': {'name': 'cell cycle', 'category': 'biological_process'},
            'GO:0006915': {'name': 'apoptotic process', 'category': 'biological_process'},
            'GO:0006281': {'name': 'DNA repair', 'category': 'biological_process'}
        }
    
    def _load_interaction_database(self) -> Dict[str, List[str]]:
        """Load protein-protein interaction database"""
        return {
            'P04637': ['P04637', 'P38398', 'P06400'],  # TP53 interactions
            'P01106': ['P01106', 'P15056', 'P01100'],  # MYC interactions
            'P38398': ['P04637', 'P38398'],  # BRCA1 interactions
        }
    
    def _load_ptm_database(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load post-translational modifications database"""
        return {
            'P04637': [
                {'type': 'phosphorylation', 'position': 15, 'residue': 'S', 'effect': 'activation'},
                {'type': 'acetylation', 'position': 120, 'residue': 'K', 'effect': 'stabilization'},
                {'type': 'ubiquitination', 'position': 386, 'residue': 'K', 'effect': 'degradation'}
            ],
            'P01106': [
                {'type': 'phosphorylation', 'position': 58, 'residue': 'T', 'effect': 'stabilization'},
                {'type': 'acetylation', 'position': 143, 'residue': 'K', 'effect': 'activation'}
            ]
        }
    
    def tokenize_protein_sequence(self, sequence: str) -> torch.Tensor:
        """
        Tokenize protein sequence for model input.
        
        Args:
            sequence: Protein sequence string
            
        Returns:
            Tokenized sequence tensor
        """
        tokens = []
        for aa in sequence.upper():
            if aa in self.protein_tokenizer:
                tokens.append(self.protein_tokenizer[aa])
            else:
                tokens.append(self.protein_tokenizer['PAD'])  # Unknown amino acid
        
        return torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    
    def predict_protein_structure(self, sequence: str) -> Dict[str, Any]:
        """
        Predict protein structure from sequence.
        
        Args:
            sequence: Protein sequence
            
        Returns:
            Dictionary containing structure predictions
        """
        # Tokenize sequence
        sequence_tensor = self.tokenize_protein_sequence(sequence)
        
        # Predict structure
        with torch.no_grad():
            secondary_pred, contact_map = self.structure_predictor(sequence_tensor)
        
        # Map secondary structure predictions
        secondary_classes = ['H', 'E', 'T', 'S', 'G', 'I', 'B', 'C']  # Helix, Sheet, Turn, etc.
        secondary_structure = []
        for i in range(secondary_pred.shape[1]):
            pred_class = torch.argmax(secondary_pred[0, i, :]).item()
            secondary_structure.append(secondary_classes[pred_class])
        
        # Analyze contact map
        contact_threshold = 0.5
        contacts = []
        for i in range(contact_map.shape[1]):
            for j in range(i + 5, contact_map.shape[2]):
                if contact_map[0, i, j] > contact_threshold:
                    contacts.append((i, j, float(contact_map[0, i, j])))
        
        return {
            'secondary_structure': ''.join(secondary_structure),
            'contact_map': contact_map[0].numpy(),
            'contacts': contacts,
            'confidence': float(torch.mean(contact_map))
        }
    
    def predict_protein_function(self, sequence: str) -> Dict[str, float]:
        """
        Predict protein function from sequence.
        
        Args:
            sequence: Protein sequence
            
        Returns:
            Dictionary of predicted functions
        """
        # Tokenize sequence
        sequence_tensor = self.tokenize_protein_sequence(sequence)
        
        # Predict function
        with torch.no_grad():
            function_pred = self.function_predictor(sequence_tensor)
        
        # Map to function names
        function_ids = list(self.function_ontology.keys())
        function_dict = {}
        for i, func_id in enumerate(function_ids):
            if i < function_pred.shape[1]:
                function_dict[func_id] = {
                    'name': self.function_ontology[func_id]['name'],
                    'category': self.function_ontology[func_id]['category'],
                    'confidence': float(function_pred[0, i])
                }
        
        return function_dict
    
    def predict_protein_interactions(self, protein1: str, protein2: str) -> Dict[str, Any]:
        """
        Predict protein-protein interaction.
        
        Args:
            protein1: First protein sequence
            protein2: Second protein sequence
            
        Returns:
            Interaction prediction results
        """
        # Tokenize sequences
        seq1_tensor = self.tokenize_protein_sequence(protein1)
        seq2_tensor = self.tokenize_protein_sequence(protein2)
        
        # Predict interaction
        with torch.no_grad():
            interaction_prob = self.interaction_predictor(seq1_tensor, seq2_tensor)
        
        return {
            'interaction_probability': float(interaction_prob[0, 0]),
            'confidence': 'high' if interaction_prob[0, 0] > 0.7 else 'medium' if interaction_prob[0, 0] > 0.4 else 'low'
        }
    
    def analyze_post_translational_modifications(self, sequence: str, protein_id: str = None) -> List[Dict[str, Any]]:
        """
        Analyze post-translational modifications.
        
        Args:
            sequence: Protein sequence
            protein_id: Protein identifier
            
        Returns:
            List of predicted PTMs
        """
        ptms = []
        
        # Check database for known PTMs
        if protein_id and protein_id in self.ptm_database:
            ptms.extend(self.ptm_database[protein_id])
        
        # Predict additional PTMs based on sequence motifs
        # Phosphorylation sites (S/T/Y followed by P)
        for i in range(len(sequence) - 1):
            if sequence[i] in 'STY' and sequence[i + 1] == 'P':
                ptms.append({
                    'type': 'phosphorylation',
                    'position': i + 1,
                    'residue': sequence[i],
                    'effect': 'predicted',
                    'confidence': 0.6
                })
        
        # Acetylation sites (K at N-terminus or in specific contexts)
        for i, aa in enumerate(sequence):
            if aa == 'K' and (i == 0 or sequence[i-1] in 'G'):
                ptms.append({
                    'type': 'acetylation',
                    'position': i + 1,
                    'residue': aa,
                    'effect': 'predicted',
                    'confidence': 0.5
                })
        
        return ptms
    
    def calculate_protein_properties(self, sequence: str) -> Dict[str, float]:
        """
        Calculate basic protein properties.
        
        Args:
            sequence: Protein sequence
            
        Returns:
            Dictionary of protein properties
        """
        # Calculate molecular weight
        aa_weights = {
            'A': 89.09, 'R': 174.20, 'N': 132.12, 'D': 133.10, 'C': 121.16,
            'Q': 146.15, 'E': 147.13, 'G': 75.07, 'H': 155.16, 'I': 131.17,
            'L': 131.17, 'K': 146.19, 'M': 149.21, 'F': 165.19, 'P': 115.13,
            'S': 105.09, 'T': 119.12, 'W': 204.23, 'Y': 181.19, 'V': 117.15
        }
        
        molecular_weight = sum(aa_weights.get(aa, 0) for aa in sequence.upper())
        
        # Calculate isoelectric point (simplified)
        charged_aa = {'K': 1, 'R': 1, 'H': 1, 'D': -1, 'E': -1}
        net_charge = sum(charged_aa.get(aa, 0) for aa in sequence.upper())
        isoelectric_point = 7.0 + net_charge * 0.1
        
        # Calculate hydrophobicity
        hydrophobic_aa = {'A': 1.8, 'V': 4.2, 'I': 4.5, 'L': 3.8, 'F': 2.8, 'W': -0.9, 'M': 1.9}
        hydrophobicity = sum(hydrophobic_aa.get(aa, 0) for aa in sequence.upper()) / len(sequence)
        
        # Calculate secondary structure propensity
        helix_propensity = sum(1 for aa in sequence.upper() if aa in 'AELMQ')
        sheet_propensity = sum(1 for aa in sequence.upper() if aa in 'VITWY')
        
        return {
            'molecular_weight': molecular_weight,
            'isoelectric_point': isoelectric_point,
            'hydrophobicity': hydrophobicity,
            'helix_propensity': helix_propensity / len(sequence),
            'sheet_propensity': sheet_propensity / len(sequence),
            'length': len(sequence)
        }
    
    def generate_protein_report(self, sequence: str, protein_id: str = None) -> Dict[str, Any]:
        """
        Generate comprehensive protein analysis report.
        
        Args:
            sequence: Protein sequence
            protein_id: Protein identifier
            
        Returns:
            Comprehensive protein report
        """
        report = {
            'sequence_info': {
                'length': len(sequence),
                'protein_id': protein_id
            },
            'properties': self.calculate_protein_properties(sequence),
            'structure_prediction': self.predict_protein_structure(sequence),
            'function_prediction': self.predict_protein_function(sequence),
            'post_translational_modifications': self.analyze_post_translational_modifications(sequence, protein_id),
            'interaction_predictions': {}
        }
        
        # Predict interactions with known proteins
        if protein_id and protein_id in self.interaction_database:
            for interactor in self.interaction_database[protein_id]:
                if interactor in self.protein_database:
                    interactor_seq = self.protein_database[interactor]['sequence']
                    interaction_pred = self.predict_protein_interactions(sequence, interactor_seq)
                    report['interaction_predictions'][interactor] = interaction_pred
        
        return report
    
    def save_models(self):
        """Save trained models"""
        torch.save(self.protein_embedder.state_dict(), self.model_dir / 'protein_embedder.pth')
        torch.save(self.structure_predictor.state_dict(), self.model_dir / 'structure_predictor.pth')
        torch.save(self.function_predictor.state_dict(), self.model_dir / 'function_predictor.pth')
        torch.save(self.interaction_predictor.state_dict(), self.model_dir / 'interaction_predictor.pth')
        
        # Save databases
        with open(self.model_dir / 'protein_database.json', 'w') as f:
            json.dump(self.protein_database, f, indent=2)
        
        with open(self.model_dir / 'function_ontology.json', 'w') as f:
            json.dump(self.function_ontology, f, indent=2)
        
        with open(self.model_dir / 'interaction_database.json', 'w') as f:
            json.dump(self.interaction_database, f, indent=2)
        
        with open(self.model_dir / 'ptm_database.json', 'w') as f:
            json.dump(self.ptm_database, f, indent=2)
        
        logger.info("Saved proteomics AI models")
    
    def load_models(self):
        """Load trained models"""
        try:
            self.protein_embedder.load_state_dict(torch.load(self.model_dir / 'protein_embedder.pth'))
            self.structure_predictor.load_state_dict(torch.load(self.model_dir / 'structure_predictor.pth'))
            self.function_predictor.load_state_dict(torch.load(self.model_dir / 'function_predictor.pth'))
            self.interaction_predictor.load_state_dict(torch.load(self.model_dir / 'interaction_predictor.pth'))
            
            # Load databases
            with open(self.model_dir / 'protein_database.json', 'r') as f:
                self.protein_database = json.load(f)
            
            with open(self.model_dir / 'function_ontology.json', 'r') as f:
                self.function_ontology = json.load(f)
            
            with open(self.model_dir / 'interaction_database.json', 'r') as f:
                self.interaction_database = json.load(f)
            
            with open(self.model_dir / 'ptm_database.json', 'r') as f:
                self.ptm_database = json.load(f)
            
            logger.info("Loaded proteomics AI models")
        except FileNotFoundError:
            logger.warning("Model files not found, using default models")