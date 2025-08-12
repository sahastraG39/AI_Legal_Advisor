#!/usr/bin/env python3
"""
FastAPI Backend for AI Legal Document Explainer
Provides REST API endpoints for document processing and analysis
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import json
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime

# Import our existing modules
from document_parser import DocumentParser
from ocr_processor import OCRProcessor
from llm_integration import LLMIntegration
from vector_store import VectorStore
from main import AILegalDocumentExplainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Legal Document Explainer API",
    description="API for analyzing legal documents using AI",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create upload directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Initialize our AI system
ai_explainer = AILegalDocumentExplainer(vector_store_type='chroma', default_llm='openai')

# Store processing status
processing_status = {}

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "AI Legal Document Explainer API", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test basic functionality
        status = ai_explainer.get_system_status()
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "system_status": status
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"System unhealthy: {str(e)}")

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    analysis_type: str = "general",
    enable_ocr: bool = True,
    store_in_vector_db: bool = True
):
    """Upload and process a legal document"""
    try:
        # Validate file type
        allowed_extensions = {'.pdf', '.docx', '.xlsx', '.txt', '.png', '.jpg', '.jpeg'}
        file_extension = Path(file.filename).suffix.lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Generate unique filename
        file_id = str(uuid.uuid4())
        file_path = UPLOAD_DIR / f"{file_id}_{file.filename}"
        
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Initialize processing status
        processing_status[file_id] = {
            "status": "processing",
            "filename": file.filename,
            "upload_time": datetime.now().isoformat(),
            "progress": 0
        }
        
        # Process document asynchronously
        try:
            results = ai_explainer.process_document(
                str(file_path),
                analysis_type=analysis_type,
                enable_ocr=enable_ocr,
                store_in_vector_db=store_in_vector_db
            )
            
            # Update status
            processing_status[file_id].update({
                "status": "completed",
                "progress": 100,
                "completion_time": datetime.now().isoformat(),
                "results": results
            })
            
            return {
                "file_id": file_id,
                "filename": file.filename,
                "status": "completed",
                "results": results
            }
            
        except Exception as e:
            # Update status with error
            processing_status[file_id].update({
                "status": "error",
                "error": str(e),
                "completion_time": datetime.now().isoformat()
            })
            raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
            
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/status/{file_id}")
async def get_processing_status(file_id: str):
    """Get the processing status of a document"""
    if file_id not in processing_status:
        raise HTTPException(status_code=404, detail="File ID not found")
    
    return processing_status[file_id]

@app.get("/documents")
async def list_documents():
    """List all processed documents"""
    try:
        # Get documents from vector store
        stats = ai_explainer.vector_store.get_statistics()
        return {
            "total_documents": stats.get("total_documents", 0),
            "document_types": stats.get("document_types", {}),
            "processing_status": processing_status
        }
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")

@app.post("/search")
async def search_documents(query: str, top_k: int = 5, threshold: float = 0.5):
    """Search through processed documents"""
    try:
        results = ai_explainer.search_documents(query, top_k=top_k, threshold=threshold)
        return {
            "query": query,
            "results": results,
            "total_found": len(results)
        }
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/document/{document_id}")
async def get_document(document_id: str):
    """Get details of a specific document"""
    try:
        document = ai_explainer.get_document_summary(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        return document
    except Exception as e:
        logger.error(f"Failed to get document: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get document: {str(e)}")

@app.post("/analyze")
async def analyze_text(text: str, analysis_type: str = "general"):
    """Analyze text without uploading a file"""
    try:
        # Use LLM integration directly
        llm = ai_explainer.llm_integration
        analysis = llm.analyze_legal_document(text, analysis_type=analysis_type)
        
        return {
            "analysis_type": analysis_type,
            "text_length": len(text),
            "analysis": analysis
        }
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/system/status")
async def get_system_status():
    """Get overall system status"""
    try:
        status = ai_explainer.get_system_status()
        return {
            "timestamp": datetime.now().isoformat(),
            "system_status": status,
            "processing_queue": len([s for s in processing_status.values() if s["status"] == "processing"])
        }
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system status: {str(e)}")

@app.delete("/document/{file_id}")
async def delete_document(file_id: str):
    """Delete a processed document"""
    try:
        if file_id not in processing_status:
            raise HTTPException(status_code=404, detail="File ID not found")
        
        # Remove from processing status
        del processing_status[file_id]
        
        # TODO: Remove from vector store and clean up files
        
        return {"message": "Document deleted successfully"}
    except Exception as e:
        logger.error(f"Failed to delete document: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
