"""
Pathway GNN - Graph Neural Networks for biological pathway analysis

This module implements advanced Graph Neural Networks for analyzing biological
pathways, protein-protein interaction networks, and metabolic networks.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from torch_geometric.nn import GCNConv, GATConv, GraphSAGE, global_mean_pool, global_max_pool
from torch_geometric.data import Data, DataLoader
import logging
from typing import Dict, List, Optional, Tuple, Any
import json
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class PathwayGCN(nn.Module):
    """Graph Convolutional Network for pathway analysis"""
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, output_dim: int = 64, num_layers: int = 3):
        super().__init__()
        self.num_layers = num_layers
        
        # GCN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        self.convs.append(GCNConv(hidden_dim, output_dim))
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
        # Batch normalization
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim if _ < num_layers - 1 else output_dim))
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for pathway GCN
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment for nodes
            
        Returns:
            Node embeddings [num_nodes, output_dim]
        """
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        return x

class PathwayGAT(nn.Module):
    """Graph Attention Network for pathway analysis"""
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, output_dim: int = 64, 
                 num_heads: int = 8, num_layers: int = 3):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # GAT layers
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(input_dim, hidden_dim // num_heads, heads=num_heads, dropout=0.1))
        
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=0.1))
        
        self.convs.append(GATConv(hidden_dim, output_dim, heads=1, dropout=0.1))
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for pathway GAT
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment for nodes
            
        Returns:
            Node embeddings [num_nodes, output_dim]
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        
        return x

class PathwaySAGE(nn.Module):
    """GraphSAGE for pathway analysis"""
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, output_dim: int = 64, num_layers: int = 3):
        super().__init__()
        self.num_layers = num_layers
        
        # GraphSAGE layers
        self.convs = nn.ModuleList()
        self.convs.append(GraphSAGE(input_dim, hidden_dim, num_layers=1))
        
        for _ in range(num_layers - 2):
            self.convs.append(GraphSAGE(hidden_dim, hidden_dim, num_layers=1))
        
        self.convs.append(GraphSAGE(hidden_dim, output_dim, num_layers=1))
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for pathway GraphSAGE
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment for nodes
            
        Returns:
            Node embeddings [num_nodes, output_dim]
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        
        return x

class PathwayClassifier(nn.Module):
    """Pathway classification model"""
    
    def __init__(self, input_dim: int = 64, hidden_dim: int = 128, num_classes: int = 10):
        super().__init__()
        
        # Graph-level pooling
        self.pooling = nn.ModuleList([
            global_mean_pool,
            global_max_pool
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for pathway classification
        
        Args:
            x: Node embeddings [num_nodes, input_dim]
            batch: Batch assignment for nodes
            
        Returns:
            Pathway classifications [batch_size, num_classes]
        """
        # Graph-level pooling
        pooled_features = []
        for pool in self.pooling:
            pooled = pool(x, batch)
            pooled_features.append(pooled)
        
        # Concatenate pooled features
        combined = torch.cat(pooled_features, dim=1)
        
        # Classify
        output = self.classifier(combined)
        
        return output

class PathwayGNN:
    """
    Advanced Graph Neural Network system for biological pathway analysis.
    
    This class integrates multiple GNN architectures for comprehensive pathway
    analysis, including protein-protein interaction networks, metabolic pathways,
    and signaling cascades.
    """
    
    def __init__(self, model_dir: str = "models/pathway_gnn"):
        """
        Initialize Pathway GNN system.
        
        Args:
            model_dir: Directory to save/load models
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize models
        self.gcn_model = PathwayGCN()
        self.gat_model = PathwayGAT()
        self.sage_model = PathwaySAGE()
        self.classifier = PathwayClassifier()
        
        # Pathway databases
        self.pathway_database = self._load_pathway_database()
        self.interaction_database = self._load_interaction_database()
        self.metabolic_database = self._load_metabolic_database()
        
        # Node features
        self.node_features = self._initialize_node_features()
        
        logger.info("Initialized Pathway GNN system")
    
    def _load_pathway_database(self) -> Dict[str, Any]:
        """Load biological pathway database"""
        return {
            'KEGG_00010': {
                'name': 'Glycolysis / Gluconeogenesis',
                'type': 'metabolic',
                'nodes': ['HK1', 'GPI', 'PFK1', 'ALDOA', 'TPI1', 'GAPDH', 'PGK1', 'PGAM1', 'ENO1', 'PKM'],
                'edges': [
                    ('HK1', 'GPI'), ('GPI', 'PFK1'), ('PFK1', 'ALDOA'),
                    ('ALDOA', 'TPI1'), ('TPI1', 'GAPDH'), ('GAPDH', 'PGK1'),
                    ('PGK1', 'PGAM1'), ('PGAM1', 'ENO1'), ('ENO1', 'PKM')
                ],
                'description': 'Central metabolic pathway for glucose breakdown'
            },
            'KEGG_04110': {
                'name': 'Cell cycle',
                'type': 'signaling',
                'nodes': ['CDK1', 'CDK2', 'CDK4', 'CCNA1', 'CCNB1', 'CCND1', 'TP53', 'RB1', 'E2F1'],
                'edges': [
                    ('CDK1', 'CCNB1'), ('CDK2', 'CCNA1'), ('CDK4', 'CCND1'),
                    ('TP53', 'CDK1'), ('TP53', 'CDK2'), ('RB1', 'E2F1'),
                    ('E2F1', 'CDK2'), ('E2F1', 'CDK4')
                ],
                'description': 'Regulation of cell cycle progression'
            },
            'KEGG_04010': {
                'name': 'MAPK signaling pathway',
                'type': 'signaling',
                'nodes': ['MAPK1', 'MAPK3', 'MAP2K1', 'MAP2K2', 'RAF1', 'BRAF', 'KRAS', 'EGFR'],
                'edges': [
                    ('EGFR', 'KRAS'), ('KRAS', 'RAF1'), ('KRAS', 'BRAF'),
                    ('RAF1', 'MAP2K1'), ('BRAF', 'MAP2K1'), ('MAP2K1', 'MAPK1'),
                    ('MAP2K2', 'MAPK3')
                ],
                'description': 'Mitogen-activated protein kinase signaling cascade'
            }
        }
    
    def _load_interaction_database(self) -> Dict[str, List[str]]:
        """Load protein-protein interaction database"""
        return {
            'TP53': ['MDM2', 'ATM', 'CHEK2', 'BRCA1', 'BRCA2'],
            'MYC': ['MAX', 'MAD', 'MXI1', 'MNT'],
            'BRCA1': ['BRCA2', 'TP53', 'ATM', 'CHEK2'],
            'EGFR': ['GRB2', 'SOS1', 'KRAS', 'RAF1'],
            'KRAS': ['RAF1', 'PIK3CA', 'RALGDS'],
            'RAF1': ['MAP2K1', 'MAP2K2'],
            'MAP2K1': ['MAPK1', 'MAPK3'],
            'CDK1': ['CCNB1', 'CCNA1', 'TP53'],
            'CDK2': ['CCNA1', 'CCNE1', 'TP53'],
            'CDK4': ['CCND1', 'CCND2', 'CCND3']
        }
    
    def _load_metabolic_database(self) -> Dict[str, Any]:
        """Load metabolic pathway database"""
        return {
            'glycolysis': {
                'enzymes': ['HK1', 'GPI', 'PFK1', 'ALDOA', 'TPI1', 'GAPDH', 'PGK1', 'PGAM1', 'ENO1', 'PKM'],
                'metabolites': ['glucose', 'glucose6p', 'fructose6p', 'fructose16bp', 'glyceraldehyde3p', 'dihydroxyacetone_phosphate', '13bpg', '3pg', '2pg', 'pep', 'pyruvate'],
                'fluxes': [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]
            },
            'tca_cycle': {
                'enzymes': ['CS', 'ACO1', 'IDH1', 'OGDH', 'SUCLG1', 'SDHA', 'FH', 'MDH1'],
                'metabolites': ['citrate', 'isocitrate', 'alpha_ketoglutarate', 'succinyl_coa', 'succinate', 'fumarate', 'malate', 'oxaloacetate'],
                'fluxes': [0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45]
            }
        }
    
    def _initialize_node_features(self) -> Dict[str, np.ndarray]:
        """Initialize node features for proteins/genes"""
        features = {}
        
        # Create features for each protein in the databases
        all_proteins = set()
        for pathway in self.pathway_database.values():
            all_proteins.update(pathway['nodes'])
        for interactions in self.interaction_database.values():
            all_proteins.update(interactions)
        
        for protein in all_proteins:
            # Create random features (in real implementation, use actual protein features)
            features[protein] = np.random.randn(128).astype(np.float32)
        
        return features
    
    def create_pathway_graph(self, pathway_id: str) -> Data:
        """
        Create PyTorch Geometric Data object for a pathway.
        
        Args:
            pathway_id: Pathway identifier
            
        Returns:
            PyTorch Geometric Data object
        """
        if pathway_id not in self.pathway_database:
            raise ValueError(f"Pathway {pathway_id} not found in database")
        
        pathway = self.pathway_database[pathway_id]
        
        # Create node features
        node_features = []
        node_mapping = {}
        for i, node in enumerate(pathway['nodes']):
            node_mapping[node] = i
            if node in self.node_features:
                node_features.append(self.node_features[node])
            else:
                node_features.append(np.random.randn(128).astype(np.float32))
        
        x = torch.tensor(np.array(node_features), dtype=torch.float)
        
        # Create edge indices
        edge_list = []
        for edge in pathway['edges']:
            if edge[0] in node_mapping and edge[1] in node_mapping:
                edge_list.append([node_mapping[edge[0]], node_mapping[edge[1]]])
                edge_list.append([node_mapping[edge[1]], node_mapping[edge[0]]])  # Undirected
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        # Create edge attributes (optional)
        edge_attr = torch.ones(edge_index.shape[1], 1)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    def create_interaction_network(self, proteins: List[str]) -> Data:
        """
        Create protein-protein interaction network.
        
        Args:
            proteins: List of protein names
            
        Returns:
            PyTorch Geometric Data object
        """
        # Create node features
        node_features = []
        node_mapping = {}
        for i, protein in enumerate(proteins):
            node_mapping[protein] = i
            if protein in self.node_features:
                node_features.append(self.node_features[protein])
            else:
                node_features.append(np.random.randn(128).astype(np.float32))
        
        x = torch.tensor(np.array(node_features), dtype=torch.float)
        
        # Create edges based on interaction database
        edge_list = []
        for protein in proteins:
            if protein in self.interaction_database:
                for interactor in self.interaction_database[protein]:
                    if interactor in node_mapping:
                        edge_list.append([node_mapping[protein], node_mapping[interactor]])
        
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        # Create edge attributes
        edge_attr = torch.ones(edge_index.shape[1], 1)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    def analyze_pathway_structure(self, pathway_id: str) -> Dict[str, Any]:
        """
        Analyze pathway structure using GNN.
        
        Args:
            pathway_id: Pathway identifier
            
        Returns:
            Pathway structure analysis
        """
        # Create pathway graph
        data = self.create_pathway_graph(pathway_id)
        
        # Analyze with different GNN models
        with torch.no_grad():
            gcn_embeddings = self.gcn_model(data.x, data.edge_index)
            gat_embeddings = self.gat_model(data.x, data.edge_index)
            sage_embeddings = self.sage_model(data.x, data.edge_index)
        
        # Calculate graph-level statistics
        num_nodes = data.x.shape[0]
        num_edges = data.edge_index.shape[1]
        density = num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
        
        # Calculate centrality measures
        centrality_measures = self._calculate_centrality_measures(data)
        
        # Calculate clustering coefficient
        clustering_coeff = self._calculate_clustering_coefficient(data)
        
        return {
            'pathway_id': pathway_id,
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'density': density,
            'clustering_coefficient': clustering_coeff,
            'centrality_measures': centrality_measures,
            'gcn_embeddings': gcn_embeddings.numpy(),
            'gat_embeddings': gat_embeddings.numpy(),
            'sage_embeddings': sage_embeddings.numpy()
        }
    
    def _calculate_centrality_measures(self, data: Data) -> Dict[str, List[float]]:
        """Calculate centrality measures for nodes"""
        # Convert to NetworkX graph
        G = nx.Graph()
        G.add_nodes_from(range(data.x.shape[0]))
        G.add_edges_from(data.edge_index.t().numpy())
        
        # Calculate centrality measures
        degree_centrality = list(nx.degree_centrality(G).values())
        betweenness_centrality = list(nx.betweenness_centrality(G).values())
        closeness_centrality = list(nx.closeness_centrality(G).values())
        
        return {
            'degree_centrality': degree_centrality,
            'betweenness_centrality': betweenness_centrality,
            'closeness_centrality': closeness_centrality
        }
    
    def _calculate_clustering_coefficient(self, data: Data) -> float:
        """Calculate clustering coefficient of the graph"""
        # Convert to NetworkX graph
        G = nx.Graph()
        G.add_nodes_from(range(data.x.shape[0]))
        G.add_edges_from(data.edge_index.t().numpy())
        
        return nx.average_clustering(G)
    
    def predict_pathway_function(self, pathway_id: str) -> Dict[str, float]:
        """
        Predict pathway function using GNN.
        
        Args:
            pathway_id: Pathway identifier
            
        Returns:
            Function predictions
        """
        # Create pathway graph
        data = self.create_pathway_graph(pathway_id)
        
        # Create batch (single graph)
        batch = torch.zeros(data.x.shape[0], dtype=torch.long)
        
        # Get node embeddings
        with torch.no_grad():
            node_embeddings = self.gcn_model(data.x, data.edge_index)
        
        # Predict function
        with torch.no_grad():
            function_pred = self.classifier(node_embeddings, batch)
        
        # Map to function types
        function_types = [
            'metabolic', 'signaling', 'transcriptional', 'translational',
            'protein_folding', 'dna_repair', 'cell_cycle', 'apoptosis',
            'immune_response', 'development'
        ]
        
        function_dict = {}
        for i, func_type in enumerate(function_types):
            if i < function_pred.shape[1]:
                function_dict[func_type] = float(function_pred[0, i])
        
        return function_dict
    
    def identify_key_nodes(self, pathway_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Identify key nodes in a pathway.
        
        Args:
            pathway_id: Pathway identifier
            top_k: Number of top nodes to return
            
        Returns:
            List of key nodes with their importance scores
        """
        # Analyze pathway structure
        analysis = self.analyze_pathway_structure(pathway_id)
        
        # Get node names
        pathway = self.pathway_database[pathway_id]
        node_names = pathway['nodes']
        
        # Calculate importance scores
        importance_scores = []
        for i, node in enumerate(node_names):
            score = (
                analysis['centrality_measures']['degree_centrality'][i] * 0.4 +
                analysis['centrality_measures']['betweenness_centrality'][i] * 0.3 +
                analysis['centrality_measures']['closeness_centrality'][i] * 0.3
            )
            importance_scores.append({
                'node': node,
                'importance_score': score,
                'degree_centrality': analysis['centrality_measures']['degree_centrality'][i],
                'betweenness_centrality': analysis['centrality_measures']['betweenness_centrality'][i],
                'closeness_centrality': analysis['centrality_measures']['closeness_centrality'][i]
            })
        
        # Sort by importance score
        importance_scores.sort(key=lambda x: x['importance_score'], reverse=True)
        
        return importance_scores[:top_k]
    
    def simulate_pathway_perturbation(self, pathway_id: str, perturbed_nodes: List[str]) -> Dict[str, Any]:
        """
        Simulate pathway perturbation effects.
        
        Args:
            pathway_id: Pathway identifier
            perturbed_nodes: List of nodes to perturb
            
        Returns:
            Perturbation effects analysis
        """
        # Create original pathway graph
        original_data = self.create_pathway_graph(pathway_id)
        
        # Create perturbed pathway graph
        perturbed_data = original_data.clone()
        
        # Perturb node features (reduce by 50%)
        pathway = self.pathway_database[pathway_id]
        node_mapping = {node: i for i, node in enumerate(pathway['nodes'])}
        
        for node in perturbed_nodes:
            if node in node_mapping:
                node_idx = node_mapping[node]
                perturbed_data.x[node_idx] *= 0.5
        
        # Analyze both graphs
        with torch.no_grad():
            original_embeddings = self.gcn_model(original_data.x, original_data.edge_index)
            perturbed_embeddings = self.gcn_model(perturbed_data.x, perturbed_data.edge_index)
        
        # Calculate perturbation effects
        embedding_diff = torch.norm(original_embeddings - perturbed_embeddings, dim=1)
        
        # Map to node names
        perturbation_effects = []
        for i, node in enumerate(pathway['nodes']):
            perturbation_effects.append({
                'node': node,
                'perturbation_effect': float(embedding_diff[i]),
                'is_perturbed': node in perturbed_nodes
            })
        
        return {
            'pathway_id': pathway_id,
            'perturbed_nodes': perturbed_nodes,
            'perturbation_effects': perturbation_effects,
            'average_effect': float(torch.mean(embedding_diff))
        }
    
    def generate_pathway_report(self, pathway_id: str) -> Dict[str, Any]:
        """
        Generate comprehensive pathway analysis report.
        
        Args:
            pathway_id: Pathway identifier
            
        Returns:
            Comprehensive pathway report
        """
        report = {
            'pathway_info': self.pathway_database[pathway_id],
            'structure_analysis': self.analyze_pathway_structure(pathway_id),
            'function_prediction': self.predict_pathway_function(pathway_id),
            'key_nodes': self.identify_key_nodes(pathway_id),
            'network_properties': {
                'connectivity': 'high' if self.analyze_pathway_structure(pathway_id)['density'] > 0.3 else 'medium' if self.analyze_pathway_structure(pathway_id)['density'] > 0.1 else 'low',
                'robustness': 'high' if self.analyze_pathway_structure(pathway_id)['clustering_coefficient'] > 0.3 else 'medium' if self.analyze_pathway_structure(pathway_id)['clustering_coefficient'] > 0.1 else 'low'
            }
        }
        
        return report
    
    def save_models(self):
        """Save trained models"""
        torch.save(self.gcn_model.state_dict(), self.model_dir / 'gcn_model.pth')
        torch.save(self.gat_model.state_dict(), self.model_dir / 'gat_model.pth')
        torch.save(self.sage_model.state_dict(), self.model_dir / 'sage_model.pth')
        torch.save(self.classifier.state_dict(), self.model_dir / 'classifier.pth')
        
        # Save databases
        with open(self.model_dir / 'pathway_database.json', 'w') as f:
            json.dump(self.pathway_database, f, indent=2)
        
        with open(self.model_dir / 'interaction_database.json', 'w') as f:
            json.dump(self.interaction_database, f, indent=2)
        
        with open(self.model_dir / 'metabolic_database.json', 'w') as f:
            json.dump(self.metabolic_database, f, indent=2)
        
        # Save node features
        with open(self.model_dir / 'node_features.pkl', 'wb') as f:
            pickle.dump(self.node_features, f)
        
        logger.info("Saved pathway GNN models")
    
    def load_models(self):
        """Load trained models"""
        try:
            self.gcn_model.load_state_dict(torch.load(self.model_dir / 'gcn_model.pth'))
            self.gat_model.load_state_dict(torch.load(self.model_dir / 'gat_model.pth'))
            self.sage_model.load_state_dict(torch.load(self.model_dir / 'sage_model.pth'))
            self.classifier.load_state_dict(torch.load(self.model_dir / 'classifier.pth'))
            
            # Load databases
            with open(self.model_dir / 'pathway_database.json', 'r') as f:
                self.pathway_database = json.load(f)
            
            with open(self.model_dir / 'interaction_database.json', 'r') as f:
                self.interaction_database = json.load(f)
            
            with open(self.model_dir / 'metabolic_database.json', 'r') as f:
                self.metabolic_database = json.load(f)
            
            # Load node features
            with open(self.model_dir / 'node_features.pkl', 'rb') as f:
                self.node_features = pickle.load(f)
            
            logger.info("Loaded pathway GNN models")
        except FileNotFoundError:
            logger.warning("Model files not found, using default models")