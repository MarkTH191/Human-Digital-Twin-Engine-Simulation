# Human Digital Twin Engine - Usage Examples

## 🚀 Quick Start Guide

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/human-digital-twin/engine.git
cd human-digital-twin-engine

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run main.py
```

### 2. Basic Usage

```python
from engine.body_simulator import HumanBodySimulator

# Initialize a human body simulator
body = HumanBodySimulator(age=25, sex="male", weight=70, height=175)

# Run a simulation step
metrics = body.simulate_step(dt=1.0)

# Get health summary
health = body.get_health_summary()
print(f"Overall health: {health['overall_health_score']:.2f}")
```

## 🧬 Core Simulation Examples

### 1. Basic Body Simulation

```python
import time
from engine.body_simulator import HumanBodySimulator

# Create a 30-year-old female
body = HumanBodySimulator(age=30, sex="female", weight=65, height=170)

# Simulate for 1 hour (3600 seconds)
for i in range(3600):
    metrics = body.simulate_step(dt=1.0)
    
    # Print vital signs every 10 minutes
    if i % 600 == 0:
        print(f"Time: {i//60} minutes")
        print(f"Heart Rate: {metrics.heart_rate:.1f} bpm")
        print(f"Temperature: {metrics.temperature:.1f}°C")
        print(f"Blood Pressure: {metrics.blood_pressure_systolic:.0f}/{metrics.blood_pressure_diastolic:.0f}")
        print("---")
```

### 2. Drug Intervention Study

```python
# Initialize body
body = HumanBodySimulator(age=45, sex="male", weight=80, height=180)

# Baseline measurements
baseline = body.get_health_summary()
print(f"Baseline heart rate: {baseline['vital_signs']['heart_rate']:.1f} bpm")

# Apply aspirin
body.apply_drug("aspirin", dose=100, duration=3600)

# Simulate for 1 hour
for i in range(3600):
    body.simulate_step(dt=1.0)

# Check effects
final = body.get_health_summary()
print(f"Final heart rate: {final['vital_signs']['heart_rate']:.1f} bpm")
print(f"Heart rate change: {final['vital_signs']['heart_rate'] - baseline['vital_signs']['heart_rate']:.1f} bpm")
```

### 3. Gene Editing Experiment

```python
# Initialize body
body = HumanBodySimulator(age=25, sex="female", weight=60, height=165)

# Baseline gene expression
baseline_health = body.get_health_summary()

# Edit TP53 gene (tumor suppressor)
body.edit_gene("TP53", expression_change=0.5)  # Increase expression by 50%

# Simulate aging for 10 years
body.set_time_acceleration(10.0)  # 10x faster
for i in range(365 * 24 * 3600):  # 1 year in seconds
    body.simulate_step(dt=1.0)

# Check cancer risk (simplified)
final_health = body.get_health_summary()
cancer_risk = 1.0 - final_health['overall_health_score']
print(f"Estimated cancer risk: {cancer_risk:.2f}")
```

## 🤖 AI Model Examples

### 1. Genomics Analysis

```python
from ai_models.genomics_ai import GenomicsAI

# Initialize genomics AI
genomics_ai = GenomicsAI()

# Analyze a DNA sequence
sequence = "ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"
tissue_type = "liver"

# Generate genomic report
report = genomics_ai.generate_genomic_report(sequence, tissue_type)

print("Gene Expression Predictions:")
for gene, expression in report['gene_expression_prediction'].items():
    print(f"{gene}: {expression:.3f}")

print("\nRegulatory Elements:")
for element_type, elements in report['regulatory_elements'].items():
    print(f"{element_type}: {len(elements)} found")
```

### 2. Proteomics Analysis

```python
from ai_models.proteomics_ai import ProteomicsAI

# Initialize proteomics AI
proteomics_ai = ProteomicsAI()

# Analyze a protein sequence
protein_sequence = "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD"

# Generate protein report
report = proteomics_ai.generate_protein_report(protein_sequence, "P04637")

print("Protein Properties:")
for prop, value in report['properties'].items():
    print(f"{prop}: {value:.2f}")

print("\nFunction Predictions:")
for func_id, func_data in report['function_prediction'].items():
    print(f"{func_data['name']}: {func_data['confidence']:.3f}")
```

### 3. Pathway Analysis

```python
from ai_models.pathway_gnn import PathwayGNN

# Initialize pathway GNN
pathway_gnn = PathwayGNN()

# Analyze glycolysis pathway
pathway_id = "KEGG_00010"
report = pathway_gnn.generate_pathway_report(pathway_id)

print("Pathway Structure:")
print(f"Nodes: {report['structure_analysis']['num_nodes']}")
print(f"Edges: {report['structure_analysis']['num_edges']}")
print(f"Density: {report['structure_analysis']['density']:.3f}")

print("\nKey Nodes:")
for node in report['key_nodes'][:5]:
    print(f"{node['node']}: {node['importance_score']:.3f}")

print("\nFunction Predictions:")
for func_type, confidence in report['function_prediction'].items():
    print(f"{func_type}: {confidence:.3f}")
```

### 4. Multi-omics Integration

```python
from ai_models.multiomics_fusion import MultiomicsFusion
import numpy as np

# Initialize multi-omics fusion
multiomics = MultiomicsFusion()

# Create mock multi-omics data
data_sources = {
    'genomics': {
        'variants': np.random.randn(100, 20),
        'expression': np.random.randn(100, 1000)
    },
    'transcriptomics': {
        'rna_seq': np.random.randn(100, 20000),
        'mirna': np.random.randn(100, 500)
    },
    'proteomics': {
        'abundance': np.random.randn(100, 5000),
        'modifications': np.random.randn(100, 100)
    },
    'metabolomics': {
        'metabolites': np.random.randn(100, 1000)
    }
}

# Generate comprehensive report
report = multiomics.generate_multiomics_report(data_sources, "sample_001")

print("Integration Results:")
print(f"Modality Importance: {report['integration_results']['modality_importance']}")

print("\nDisease Risk Predictions:")
for disease, risk in report['disease_risk_predictions'].items():
    print(f"{disease}: {risk:.3f}")

print("\nBiomarkers:")
for modality, biomarkers in report['biomarkers'].items():
    print(f"{modality}: {len(biomarkers)} biomarkers identified")
```

## 📊 Data Integration Examples

### 1. Human Cell Atlas Data

```python
from data.loaders.human_cell_atlas_loader import HumanCellAtlasLoader

# Initialize HCA loader
hca_loader = HumanCellAtlasLoader()

# Get available datasets
datasets = hca_loader.get_available_datasets()
print("Available datasets:")
for dataset in datasets:
    print(f"- {dataset['title']}: {dataset['cell_count']} cells")

# Load a dataset
dataset_id = "HCA_001"
adata = hca_loader.load_dataset(dataset_id)

if adata is not None:
    print(f"\nLoaded dataset: {adata.n_obs} cells, {adata.n_vars} genes")
    
    # Get cell types
    cell_types = hca_loader.get_cell_types(dataset_id)
    print(f"Cell types: {cell_types}")
    
    # Get expression profiles
    profiles = hca_loader.get_tissue_expression_profiles(dataset_id)
    print(f"Expression profiles for {len(profiles)} cell types")
```

### 2. UniProt Protein Data

```python
from data.loaders.uniprot_loader import UniProtLoader

# Initialize UniProt loader
uniprot_loader = UniProtLoader()

# Search for proteins
results = uniprot_loader.search_proteins("p53", limit=5)
print("Search results:")
for result in results:
    print(f"- {result['name']} ({result['accession']})")

# Get detailed protein information
accession = "P04637"
protein_info = uniprot_loader.get_protein_info(accession)

if protein_info:
    print(f"\nProtein: {protein_info['name']}")
    print(f"Length: {protein_info['length']} amino acids")
    print(f"Function: {protein_info['function']}")
    
    # Get protein sequence
    sequence = uniprot_loader.get_protein_sequence(accession)
    print(f"Sequence length: {len(sequence)} amino acids")
    
    # Get interactions
    interactions = uniprot_loader.get_protein_interactions(accession)
    print(f"Interactions: {len(interactions)} partners")
```

### 3. PDB Structure Data

```python
from data.loaders.pdb_loader import PDBLoader

# Initialize PDB loader
pdb_loader = PDBLoader()

# Get structure information
pdb_id = "1TUP"
structure_info = pdb_loader.get_structure_info(pdb_id)

if structure_info:
    print(f"Structure: {structure_info['title']}")
    print(f"Resolution: {structure_info['resolution']} Å")
    print(f"Method: {structure_info['method']}")
    print(f"Molecules: {len(structure_info['molecules'])}")
    
    # Get coordinates
    coordinates = pdb_loader.get_structure_coordinates(pdb_id)
    if coordinates:
        print(f"Atoms: {len(coordinates['atoms'])}")
        print(f"Chains: {coordinates['chains']}")
    
    # Get secondary structure
    secondary_structure = pdb_loader.get_secondary_structure(pdb_id)
    if secondary_structure:
        print(f"Helices: {len(secondary_structure['helices'])}")
        print(f"Sheets: {len(secondary_structure['sheets'])}")
```

## 🌐 API Usage Examples

### 1. REST API Client

```python
import requests
import json

# API base URL
base_url = "http://localhost:8000"

# Initialize simulation
response = requests.post(f"{base_url}/simulation/initialize", 
                        params={"age": 25, "sex": "male", "weight": 70, "height": 175})
print("Simulation initialized:", response.json())

# Run simulation step
response = requests.post(f"{base_url}/simulation/step", params={"dt": 1.0})
metrics = response.json()["metrics"]
print(f"Heart rate: {metrics['heart_rate']:.1f} bpm")

# Apply drug
response = requests.post(f"{base_url}/simulation/drug", 
                        params={"drug_name": "aspirin", "dose": 100})
print("Drug applied:", response.json())

# Get simulation status
response = requests.get(f"{base_url}/simulation/status")
status = response.json()
print(f"Simulation time: {status['simulation_time']:.1f} seconds")
```

### 2. AI Analysis via API

```python
# Genomics analysis
sequence = "ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"
response = requests.post(f"{base_url}/ai/genomics/analyze",
                        params={"sequence": sequence, "tissue_type": "liver"})
genomics_results = response.json()["results"]
print("Genomics analysis completed")

# Proteomics analysis
protein_sequence = "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD"
response = requests.post(f"{base_url}/ai/proteomics/analyze",
                        params={"sequence": protein_sequence})
proteomics_results = response.json()["results"]
print("Proteomics analysis completed")
```

## 🎮 GUI Usage Examples

### 1. Streamlit Dashboard

```python
# Run the Streamlit dashboard
# streamlit run main.py

# The dashboard provides:
# - Real-time health monitoring
# - 3D body visualization
# - Interactive controls
# - AI analysis tools
# - Experiment management
```

### 2. Custom Dashboard Integration

```python
import streamlit as st
from engine.body_simulator import HumanBodySimulator

# Custom dashboard example
st.title("Custom Human Digital Twin Dashboard")

# Initialize simulator
if 'body_simulator' not in st.session_state:
    st.session_state.body_simulator = HumanBodySimulator(age=30, sex="female")

body = st.session_state.body_simulator

# Health metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Heart Rate", f"{body.current_metrics.heart_rate:.1f} bpm")
with col2:
    st.metric("Temperature", f"{body.current_metrics.temperature:.1f}°C")
with col3:
    st.metric("Health Score", f"{body.get_health_summary()['overall_health_score']:.2f}")

# Controls
if st.button("Run Simulation Step"):
    body.simulate_step(1.0)
    st.success("Simulation step completed!")

if st.button("Apply Aspirin"):
    body.apply_drug("aspirin", 100)
    st.success("Aspirin applied!")
```

## 🔬 Advanced Research Examples

### 1. Drug Discovery Pipeline

```python
# Virtual drug screening
def screen_drug_candidates(body, drug_candidates):
    results = []
    
    for drug_name, dose in drug_candidates:
        # Reset body state
        body.simulation_time = 0
        
        # Apply drug
        body.apply_drug(drug_name, dose)
        
        # Simulate for 24 hours
        for i in range(24 * 3600):
            body.simulate_step(dt=1.0)
        
        # Measure efficacy
        final_health = body.get_health_summary()
        efficacy = final_health['overall_health_score']
        
        results.append({
            'drug': drug_name,
            'dose': dose,
            'efficacy': efficacy
        })
    
    return sorted(results, key=lambda x: x['efficacy'], reverse=True)

# Test drug candidates
drug_candidates = [
    ("aspirin", 100),
    ("metformin", 500),
    ("statin", 20),
    ("ace_inhibitor", 10)
]

body = HumanBodySimulator(age=50, sex="male", weight=80, height=175)
results = screen_drug_candidates(body, drug_candidates)

print("Drug screening results:")
for result in results:
    print(f"{result['drug']}: {result['efficacy']:.3f}")
```

### 2. Personalized Medicine Study

```python
# Personalized medicine simulation
def personalized_treatment_study(patient_profile):
    # Create patient-specific body
    body = HumanBodySimulator(
        age=patient_profile['age'],
        sex=patient_profile['sex'],
        weight=patient_profile['weight'],
        height=patient_profile['height']
    )
    
    # Apply patient-specific genetic variants
    for gene, variant in patient_profile['genetics'].items():
        body.edit_gene(gene, variant['effect'])
    
    # Test different treatments
    treatments = patient_profile['treatments']
    results = {}
    
    for treatment_name, treatment_params in treatments.items():
        # Reset body
        body.simulation_time = 0
        
        # Apply treatment
        if treatment_params['type'] == 'drug':
            body.apply_drug(treatment_params['drug'], treatment_params['dose'])
        elif treatment_params['type'] == 'gene_therapy':
            body.edit_gene(treatment_params['gene'], treatment_params['change'])
        
        # Simulate treatment response
        for i in range(treatment_params['duration']):
            body.simulate_step(dt=1.0)
        
        # Measure response
        final_health = body.get_health_summary()
        results[treatment_name] = {
            'efficacy': final_health['overall_health_score'],
            'side_effects': 1.0 - final_health['organ_health'].get('liver', 0.9)
        }
    
    return results

# Patient profile
patient = {
    'age': 45,
    'sex': 'female',
    'weight': 65,
    'height': 170,
    'genetics': {
        'CYP2D6': {'effect': -0.5},  # Poor metabolizer
        'TP53': {'effect': 0.2}      # Increased cancer risk
    },
    'treatments': {
        'standard_dose': {'type': 'drug', 'drug': 'metformin', 'dose': 500, 'duration': 7200},
        'reduced_dose': {'type': 'drug', 'drug': 'metformin', 'dose': 250, 'duration': 7200},
        'gene_therapy': {'type': 'gene_therapy', 'gene': 'TP53', 'change': 0.3, 'duration': 7200}
    }
}

results = personalized_treatment_study(patient)
print("Personalized treatment results:")
for treatment, result in results.items():
    print(f"{treatment}: Efficacy={result['efficacy']:.3f}, Side Effects={result['side_effects']:.3f}")
```

## 📚 Additional Resources

### Documentation
- [API Reference](api_reference.md)
- [Developer Guide](developer_guide.md)
- [Tutorial Series](tutorials/)
- [FAQ](faq.md)

### Community
- [GitHub Repository](https://github.com/human-digital-twin/engine)
- [Discord Community](https://discord.gg/human-digital-twin)
- [Forum](https://forum.human-digital-twin.org)
- [Newsletter](https://newsletter.human-digital-twin.org)

### Support
- [Issue Tracker](https://github.com/human-digital-twin/engine/issues)
- [Documentation](https://docs.human-digital-twin.org)
- [Contact](mailto:support@human-digital-twin.org)

---

*For more examples and advanced usage, visit our [documentation website](https://docs.human-digital-twin.org) or join our [community forum](https://forum.human-digital-twin.org).*