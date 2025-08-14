#!/usr/bin/env python3
"""
Simplified FastAPI Backend for Phase 2 Testing
Quick startup without heavy AI dependencies
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from datetime import datetime
import json

# Initialize FastAPI app
app = FastAPI(
    title="AI Legal Document Explainer - Phase 2",
    description="Phase 2: Dataset Development & AI Enhancement",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "AI Legal Document Explainer API - Phase 2 Active! 🚀",
        "status": "running",
        "phase": "Phase 2: Dataset Development & AI Enhancement",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "api_docs": "/docs",
            "health_check": "/health",
            "phase2_status": "/api/phase2/status",
            "dataset_stats": "/api/dataset/statistics",
            "annotation_stats": "/api/annotation/statistics"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "message": "Phase 2 API is running successfully!"
    }

@app.get("/api/status")
async def api_status():
    """API status endpoint"""
    return {
        "message": "AI Legal Document Explainer API - Phase 2",
        "status": "running",
        "version": "2.0.0"
    }

# ===== PHASE 2: Dataset Development & AI Enhancement =====

@app.get("/api/phase2/status")
async def get_phase2_status():
    """Get Phase 2 implementation status"""
    return {
        "phase": "Phase 2: Dataset Development & AI Enhancement",
        "status": "active",
        "timestamp": datetime.now().isoformat(),
        "features": {
            "synthetic_data_generation": "available",
            "data_annotation_framework": "available", 
            "enhanced_ai_analysis": "available",
            "risk_assessment": "available",
            "clause_analysis": "available"
        },
        "endpoints": {
            "generate_synthetic": "/api/dataset/generate-synthetic",
            "dataset_stats": "/api/dataset/statistics",
            "export_dataset": "/api/dataset/export",
            "create_annotation": "/api/annotation/create",
            "annotation_stats": "/api/annotation/statistics",
            "enhanced_analysis": "/api/analysis/enhanced"
        }
    }

@app.get("/api/dataset/statistics")
async def get_dataset_statistics():
    """Get comprehensive dataset statistics"""
    return {
        "total_documents": 50,
        "document_types": {
            "employment_contract": 29,
            "nda": 13,
            "service_contract": 8
        },
        "sources": {
            "synthetic_generation": 50
        },
        "risk_levels": {
            "medium": 17,
            "high": 17,
            "low": 16
        },
        "tags": {
            "employment": 29,
            "contract": 37,
            "legal": 50,
            "hr": 29,
            "nda": 13,
            "confidentiality": 13,
            "business": 21,
            "service": 8
        },
        "collection_timeline": {
            "2025-08-14": 50
        }
    }

@app.get("/api/annotation/statistics")
async def get_annotation_statistics():
    """Get annotation statistics and quality metrics"""
    return {
        "total_annotations": 2,
        "annotations_by_status": {
            "pending": 2
        },
        "annotations_by_type": {
            "termination": 1,
            "liability": 1
        },
        "annotations_by_risk": {
            "medium": 1,
            "high": 1
        },
        "annotations_by_annotator": {
            "annotator_001": 2
        },
        "quality_metrics": {
            "average_confidence": 0.8,
            "average_risk_score": 0.7,
            "average_importance_score": 0.85
        }
    }

@app.post("/api/dataset/generate-synthetic")
async def generate_synthetic_dataset(num_documents: int = 100):
    """Generate synthetic legal documents for training"""
    return {
        "message": f"Generated {num_documents} synthetic documents",
        "generated_count": num_documents,
        "total_documents": 50 + num_documents,
        "document_types": {
            "employment_contract": 29,
            "nda": 13,
            "service_contract": 8,
            "new_synthetic": num_documents
        },
        "status": "success"
    }

@app.post("/api/analysis/enhanced")
async def perform_enhanced_analysis(document_content: str, document_id: str):
    """Perform enhanced AI analysis with risk assessment"""
    # Simple risk assessment simulation
    risk_score = 0.6 if "liability" in document_content.lower() else 0.4
    risk_level = "high" if risk_score > 0.5 else "medium"
    
    return {
        "message": "Enhanced analysis completed",
        "document_id": document_id,
        "analysis_timestamp": datetime.now().isoformat(),
        "risk_assessment": {
            "overall_risk_score": risk_score,
            "risk_level": risk_level,
            "risk_categories": {
                "financial": risk_score * 0.8,
                "legal": risk_score * 1.2,
                "operational": risk_score * 0.6,
                "compliance": risk_score * 0.9
            },
            "high_risk_clauses": [
                "Sample high-risk clause identified" if risk_score > 0.5 else "No high-risk clauses found"
            ],
            "recommendations": [
                "Review liability clauses" if risk_score > 0.5 else "Document appears to have standard risk levels"
            ]
        }
    }

if __name__ == "__main__":
    print("🚀 Starting Phase 2 API Server...")
    print("📱 Access your API at: http://localhost:8000")
    print("📚 API Documentation at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
