import os
import time
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, Query, Path as FPath
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import pandas as pd

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from api.schemas import (
    TransactionPayload,
    ScoreResponse,
    CaseUpdateRequest,
    SARResponse,
)
from features.feature_store import FeaturePipeline
from models.ensemble import HybridFraudEnsemble
from investigation.case_manager import CaseManager
from investigation.graph_engine import GraphInvestigationEngine
from investigation.sar_generator import SARGenerator


from contextlib import asynccontextmanager

# Global service containers
pipeline: Optional[FeaturePipeline] = None
ensemble: Optional[HybridFraudEnsemble] = None
case_manager: Optional[CaseManager] = None
graph_engine: Optional[GraphInvestigationEngine] = None
sar_generator: Optional[SARGenerator] = None


def load_services():
    """Initializes and caches models and engines in memory."""
    global pipeline, ensemble, case_manager, graph_engine, sar_generator
    print("[API Startup] Loading feature pipeline and model ensemble...")

    try:
        pipeline = FeaturePipeline.load()
    except Exception as e:
        print(f"[API Startup Warning] Could not load feature pipeline: {e}")
        pipeline = FeaturePipeline()

    try:
        ensemble = HybridFraudEnsemble.load()
    except Exception as e:
        print(f"[API Startup Warning] Could not load hybrid ensemble: {e}")
        ensemble = HybridFraudEnsemble()

    case_manager = CaseManager()
    sar_generator = SARGenerator()

    # Load and build graph
    graph_engine = GraphInvestigationEngine()
    enriched_path = config.PROCESSED_DATA_DIR / "enriched_transactions.csv"
    if enriched_path.exists():
        df = pd.read_csv(enriched_path).head(4000)
        graph_engine.build_from_dataframe(df)
        print(f"[API Startup] Graph initialized with {graph_engine.graph.number_of_nodes()} nodes and {graph_engine.graph.number_of_edges()} edges.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_services()
    yield


# Initialize FastAPI Application
app = FastAPI(
    title="AI-Powered Fraud Investigation & Detection Platform API",
    description="Enterprise real-time transaction scoring, explainable AI, graph forensics, and FinCEN SAR compliance API.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Enable CORS for web dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "platform": "AI-Powered Fraud Investigation & Detection Platform",
        "status": "ONLINE",
        "version": "2.0.0",
        "documentation": "/docs",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "pipeline_loaded": pipeline is not None,
        "models_loaded": ensemble is not None,
        "database_connected": case_manager is not None,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/api/v1/score", response_model=ScoreResponse)
def score_transaction(payload: TransactionPayload):
    """
    Real-Time Transaction Scoring Endpoint (<25ms).
    Computes streaming features, evaluates hybrid ensemble, generates SHAP attribution,
    and automatically opens an investigation case if score exceeds threshold.
    """
    start_time = time.time()
    raw_tx = payload.model_dump()
    if not raw_tx.get("transaction_id"):
        raw_tx["transaction_id"] = f"TX_{uuid.uuid4().hex[:12].upper()}"

    # Extract online features
    feature_row = pipeline.extract_single_record(raw_tx)

    # Hybrid score
    score_result = ensemble.score_record(raw_tx, feature_row)
    elapsed_ms = round((time.time() - start_time) * 1000, 2)
    score_result["processing_time_ms"] = elapsed_ms

    # Auto-case creation if HIGH or CRITICAL risk
    created_case_id = None
    if score_result["composite_risk_score"] >= config.THRESHOLD_HIGH:
        created_case_id = case_manager.create_case(
            transaction_id=score_result["transaction_id"],
            user_id=raw_tx["user_id"],
            card_id=raw_tx["card_id"],
            device_id=raw_tx["device_id"],
            ip_address=raw_tx["ip_address"],
            amount=raw_tx["amount"],
            risk_score=score_result["composite_risk_score"],
            risk_tier=score_result["risk_tier"],
            fraud_typology=score_result["rules_triggered"][0]["name"] if score_result["rules_triggered"] else "AI Anomaly Cluster",
            rules_triggered=score_result["rules_triggered"],
            explanation_narrative=score_result["explanation_narrative"],
            assigned_investigator="Real-time Alert Triage Desk",
            initial_notes=f"Real-time streaming alert: {score_result['recommended_action']}",
        )
    score_result["case_id"] = created_case_id

    return score_result


@app.post("/api/v1/batch-score")
def score_batch_transactions(transactions: List[TransactionPayload]):
    """Batch scoring for settlement batches and historical reconciliation."""
    results = []
    for tx in transactions:
        raw_tx = tx.model_dump()
        feat = pipeline.extract_single_record(raw_tx)
        res = ensemble.score_record(raw_tx, feat)
        results.append(res)
    return {
        "total_scored": len(results),
        "results": results,
    }


@app.get("/api/v1/cases")
def list_cases(
    status: Optional[str] = Query(None, description="Filter by status: NEW, ASSIGNED, UNDER_INVESTIGATION, CONFIRMED_FRAUD, FALSE_POSITIVE, CLEARED"),
    risk_tier: Optional[str] = Query(None, description="Filter by risk tier: HIGH, CRITICAL"),
    limit: int = Query(50, ge=1, le=500),
):
    """Retrieves fraud alert cases for investigator triage queue."""
    return case_manager.list_cases(status=status, risk_tier=risk_tier, limit=limit)


@app.get("/api/v1/cases/{case_id}")
def get_case(case_id: str = FPath(..., description="Unique Case ID")):
    """Retrieves full case dossier with audit history."""
    case_data = case_manager.get_case(case_id)
    if not case_data:
        raise HTTPException(status_code=404, detail=f"Case {case_id} not found.")
    return case_data


@app.patch("/api/v1/cases/{case_id}")
def update_case(case_id: str, update: CaseUpdateRequest):
    """Updates case status, assigns investigator, logs notes into immutable audit trail."""
    try:
        success = case_manager.update_case_status(
            case_id=case_id,
            new_status=update.new_status,
            actor=update.actor,
            notes=update.notes,
            assigned_investigator=update.assigned_investigator,
        )
        if not success:
            raise HTTPException(status_code=404, detail="Case not found or update failed.")
        return {"status": "success", "message": f"Case {case_id} updated to {update.new_status}"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/v1/cases/{case_id}/sar", response_model=SARResponse)
def generate_sar(case_id: str):
    """Generates an automated, FinCEN-compliant Suspicious Activity Report (SAR) narrative."""
    case_data = case_manager.get_case(case_id)
    if not case_data:
        raise HTTPException(status_code=404, detail="Case not found.")

    markdown_narrative = sar_generator.generate_narrative_text(case_data)
    pdf_filename = f"SAR_{case_id}.pdf"
    sar_generator.export_pdf(case_data, filename=pdf_filename)

    return {
        "case_id": case_id,
        "narrative_markdown": markdown_narrative,
        "pdf_filename": pdf_filename,
        "generated_at": datetime.utcnow().isoformat(),
    }


@app.get("/api/v1/cases/{case_id}/sar/pdf")
def download_sar_pdf(case_id: str):
    """Direct PDF download endpoint for official regulatory filing."""
    case_data = case_manager.get_case(case_id)
    if not case_data:
        raise HTTPException(status_code=404, detail="Case not found.")

    pdf_filename = f"SAR_{case_id}.pdf"
    pdf_path = sar_generator.output_dir / pdf_filename
    if not pdf_path.exists():
        pdf_path = sar_generator.export_pdf(case_data, filename=pdf_filename)

    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=pdf_filename,
    )


@app.get("/api/v1/graph/{entity_id}")
def get_graph_neighborhood(
    entity_id: str,
    hops: int = Query(2, ge=1, le=3),
    max_nodes: int = Query(40, ge=5, le=100),
):
    """Fetches topological ego-network subgraph around an entity (user, device, card, IP)."""
    if graph_engine is None:
        raise HTTPException(status_code=503, detail="Graph engine is not initialized.")
    return graph_engine.extract_subgraph(center_id=entity_id, hops=hops, max_nodes=max_nodes)


@app.get("/api/v1/graph/rings/collusion")
def get_collusion_rings(min_users: int = Query(3, ge=2, le=10)):
    """Discovers shared-infrastructure collusion rings (multi-account devices or IPs)."""
    if graph_engine is None:
        raise HTTPException(status_code=503, detail="Graph engine is not initialized.")
    return graph_engine.detect_collusion_rings(min_users=min_users)


@app.get("/api/v1/metrics")
def get_operations_metrics():
    """Retrieves operational SOC/FOC monitoring KPIs."""
    return case_manager.get_summary_metrics()
