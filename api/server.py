"""
Human Digital Twin API Server

This module provides the main API server for the Human Digital Twin Engine,
enabling remote access and collaboration features.
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
from typing import Dict, List, Optional, Any
import json
import asyncio
from contextlib import asynccontextmanager

from ..engine.body_simulator import HumanBodySimulator
from ..ai_models.genomics_ai import GenomicsAI
from ..ai_models.proteomics_ai import ProteomicsAI
from ..ai_models.pathway_gnn import PathwayGNN
from ..ai_models.multiomics_fusion import MultiomicsFusion
from .endpoints import SimulationEndpoints, AIEndpoints, DataEndpoints

logger = logging.getLogger(__name__)

# Global instances
body_simulator: Optional[HumanBodySimulator] = None
ai_models: Dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Human Digital Twin API Server...")
    
    # Initialize AI models
    global ai_models
    try:
        ai_models['genomics'] = GenomicsAI()
        ai_models['proteomics'] = ProteomicsAI()
        ai_models['pathway_gnn'] = PathwayGNN()
        ai_models['multiomics'] = MultiomicsFusion()
        logger.info("AI models initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing AI models: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Human Digital Twin API Server...")

class HumanDigitalTwinAPI:
    """
    Main API server for the Human Digital Twin Engine.
    
    This class provides RESTful endpoints for accessing and controlling
    the human digital twin simulation, AI models, and data integration.
    """
    
    def __init__(self, title: str = "Human Digital Twin Engine API", version: str = "1.0.0"):
        """
        Initialize the API server.
        
        Args:
            title: API title
            version: API version
        """
        self.app = FastAPI(
            title=title,
            version=version,
            description="The World's First Open Source Human Digital Twin Engine API",
            lifespan=lifespan
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Initialize endpoints
        self.simulation_endpoints = SimulationEndpoints()
        self.ai_endpoints = AIEndpoints()
        self.data_endpoints = DataEndpoints()
        
        # Register routes
        self._register_routes()
        
        logger.info("Human Digital Twin API Server initialized")
    
    def _register_routes(self):
        """Register API routes"""
        
        # Health check
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "message": "Human Digital Twin Engine API is running",
                "version": "1.0.0"
            }
        
        # Simulation endpoints
        @self.app.post("/simulation/initialize")
        async def initialize_simulation(
            age: float = 25.0,
            sex: str = "male",
            weight: float = 70.0,
            height: float = 175.0
        ):
            """Initialize human body simulator"""
            try:
                global body_simulator
                body_simulator = HumanBodySimulator(age, sex, weight, height)
                
                return {
                    "status": "success",
                    "message": "Human body simulator initialized",
                    "body_parameters": {
                        "age": age,
                        "sex": sex,
                        "weight": weight,
                        "height": height
                    }
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/simulation/status")
        async def get_simulation_status():
            """Get current simulation status"""
            global body_simulator
            
            if body_simulator is None:
                raise HTTPException(status_code=404, detail="Simulation not initialized")
            
            health_summary = body_simulator.get_health_summary()
            
            return {
                "status": "running",
                "simulation_time": body_simulator.simulation_time,
                "time_acceleration": body_simulator.time_acceleration,
                "health_summary": health_summary
            }
        
        @self.app.post("/simulation/step")
        async def run_simulation_step(dt: float = 1.0):
            """Run one simulation step"""
            global body_simulator
            
            if body_simulator is None:
                raise HTTPException(status_code=404, detail="Simulation not initialized")
            
            try:
                metrics = body_simulator.simulate_step(dt)
                
                return {
                    "status": "success",
                    "simulation_time": body_simulator.simulation_time,
                    "metrics": {
                        "age": metrics.age,
                        "heart_rate": metrics.heart_rate,
                        "temperature": metrics.temperature,
                        "blood_pressure_systolic": metrics.blood_pressure_systolic,
                        "blood_pressure_diastolic": metrics.blood_pressure_diastolic,
                        "glucose_level": metrics.glucose_level,
                        "cellular_health_score": metrics.cellular_health_score,
                        "epigenetic_age": metrics.epigenetic_age
                    }
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/simulation/drug")
        async def apply_drug(
            drug_name: str,
            dose: float,
            duration: float = 3600.0
        ):
            """Apply drug intervention"""
            global body_simulator
            
            if body_simulator is None:
                raise HTTPException(status_code=404, detail="Simulation not initialized")
            
            try:
                body_simulator.apply_drug(drug_name, dose, duration)
                
                return {
                    "status": "success",
                    "message": f"Applied {dose}mg of {drug_name}",
                    "active_drugs": list(body_simulator.active_drugs.keys())
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/simulation/gene_edit")
        async def edit_gene(
            gene_name: str,
            expression_change: float
        ):
            """Edit gene expression"""
            global body_simulator
            
            if body_simulator is None:
                raise HTTPException(status_code=404, detail="Simulation not initialized")
            
            try:
                body_simulator.edit_gene(gene_name, expression_change)
                
                return {
                    "status": "success",
                    "message": f"Edited {gene_name} expression by {expression_change}"
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/simulation/time_acceleration")
        async def set_time_acceleration(acceleration: float):
            """Set time acceleration"""
            global body_simulator
            
            if body_simulator is None:
                raise HTTPException(status_code=404, detail="Simulation not initialized")
            
            try:
                body_simulator.set_time_acceleration(acceleration)
                
                return {
                    "status": "success",
                    "message": f"Time acceleration set to {acceleration}x",
                    "time_acceleration": body_simulator.time_acceleration
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/simulation/export")
        async def export_simulation():
            """Export simulation state"""
            global body_simulator
            
            if body_simulator is None:
                raise HTTPException(status_code=404, detail="Simulation not initialized")
            
            try:
                state = body_simulator.export_state()
                
                return {
                    "status": "success",
                    "simulation_state": state
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/simulation/import")
        async def import_simulation(simulation_state: dict):
            """Import simulation state"""
            global body_simulator
            
            try:
                if body_simulator is None:
                    body_simulator = HumanBodySimulator()
                
                body_simulator.import_state(simulation_state)
                
                return {
                    "status": "success",
                    "message": "Simulation state imported successfully"
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        # AI endpoints
        @self.app.post("/ai/genomics/analyze")
        async def analyze_genomics(
            sequence: str,
            tissue_type: str = "liver"
        ):
            """Run genomics analysis"""
            try:
                if 'genomics' not in ai_models:
                    raise HTTPException(status_code=500, detail="Genomics AI model not available")
                
                results = ai_models['genomics'].generate_genomic_report(sequence, tissue_type)
                
                return {
                    "status": "success",
                    "results": results
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/ai/proteomics/analyze")
        async def analyze_proteomics(
            sequence: str,
            protein_id: str = None
        ):
            """Run proteomics analysis"""
            try:
                if 'proteomics' not in ai_models:
                    raise HTTPException(status_code=500, detail="Proteomics AI model not available")
                
                results = ai_models['proteomics'].generate_protein_report(sequence, protein_id)
                
                return {
                    "status": "success",
                    "results": results
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/ai/pathway/analyze")
        async def analyze_pathway(pathway_id: str):
            """Run pathway analysis"""
            try:
                if 'pathway_gnn' not in ai_models:
                    raise HTTPException(status_code=500, detail="Pathway GNN model not available")
                
                results = ai_models['pathway_gnn'].generate_pathway_report(pathway_id)
                
                return {
                    "status": "success",
                    "results": results
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/ai/multiomics/analyze")
        async def analyze_multiomics(data_sources: dict):
            """Run multi-omics analysis"""
            try:
                if 'multiomics' not in ai_models:
                    raise HTTPException(status_code=500, detail="Multi-omics fusion model not available")
                
                results = ai_models['multiomics'].generate_multiomics_report(data_sources)
                
                return {
                    "status": "success",
                    "results": results
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        # Data endpoints
        @self.app.get("/data/hca/datasets")
        async def get_hca_datasets():
            """Get available HCA datasets"""
            try:
                from ..data.loaders.human_cell_atlas_loader import HumanCellAtlasLoader
                loader = HumanCellAtlasLoader()
                datasets = loader.get_available_datasets()
                
                return {
                    "status": "success",
                    "datasets": datasets
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/data/uniprot/search")
        async def search_uniprot(query: str, limit: int = 100):
            """Search UniProt database"""
            try:
                from ..data.loaders.uniprot_loader import UniProtLoader
                loader = UniProtLoader()
                results = loader.search_proteins(query, limit)
                
                return {
                    "status": "success",
                    "results": results
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/data/pdb/structure/{pdb_id}")
        async def get_pdb_structure(pdb_id: str):
            """Get PDB structure information"""
            try:
                from ..data.loaders.pdb_loader import PDBLoader
                loader = PDBLoader()
                structure_info = loader.get_structure_info(pdb_id)
                
                if structure_info is None:
                    raise HTTPException(status_code=404, detail=f"Structure {pdb_id} not found")
                
                return {
                    "status": "success",
                    "structure": structure_info
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        # Error handlers
        @self.app.exception_handler(HTTPException)
        async def http_exception_handler(request, exc):
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "status": "error",
                    "message": exc.detail,
                    "status_code": exc.status_code
                }
            )
        
        @self.app.exception_handler(Exception)
        async def general_exception_handler(request, exc):
            logger.error(f"Unhandled exception: {exc}")
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": "Internal server error",
                    "status_code": 500
                }
            )
    
    def run(self, host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
        """
        Run the API server.
        
        Args:
            host: Host address
            port: Port number
            reload: Enable auto-reload for development
        """
        logger.info(f"Starting Human Digital Twin API Server on {host}:{port}")
        
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )

def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    api = HumanDigitalTwinAPI()
    return api.app

if __name__ == "__main__":
    # Run the API server
    api = HumanDigitalTwinAPI()
    api.run(host="0.0.0.0", port=8000, reload=True)