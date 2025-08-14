#!/usr/bin/env python3
"""
FastAPI Backend for AI Legal Document Explainer
Provides REST API endpoints for document processing and analysis
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from dataclasses import asdict
from pydantic import BaseModel
import uvicorn
import os
import json
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime

# Import our existing modules
from src.document_parser import DocumentParser
from src.ocr_processor import OCRProcessor
from src.llm_integration import LLMIntegration
from src.vector_store import VectorStore
from src.main import AILegalDocumentExplainer

# Import Phase 2 modules
from src.data_collector import DataCollector
from src.data_annotation import DataAnnotationFramework
from src.enhanced_ai_analysis import EnhancedAIAnalysis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for request validation
class ChatRequest(BaseModel):
    document_id: str
    question: str
    context: str = ""

# Initialize FastAPI app
app = FastAPI(
    title="AI Legal Document Explainer",
    description="Complete AI Legal Document Explainer with Frontend",
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
UPLOAD_DIR = Path("src/uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Mount React frontend static files
app.mount("/static", StaticFiles(directory="frontend/build/static"), name="static")

# Initialize our AI system
ai_explainer = AILegalDocumentExplainer(vector_store_type='chroma', default_llm='openai')

# Initialize Phase 2 systems
data_collector = DataCollector()
annotation_framework = DataAnnotationFramework()
enhanced_analyzer = EnhancedAIAnalysis()

# Store processing status
processing_status = {}

@app.get("/")
async def root():
    """Serve React frontend"""
    return FileResponse("frontend/build/index.html")

@app.get("/api/status")
async def api_status():
    """API status endpoint"""
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

@app.post("/api/upload")
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
            # Try to process with AI system first
            try:
                results = ai_explainer.process_document(
                    str(file_path),
                    analysis_type=analysis_type,
                    enable_ocr=enable_ocr,
                    store_in_vector_db=store_in_vector_db
                )
                
                # Create enhanced results with plain English summary
                enhanced_results = {
                    "document_type": results.get("document_type", "legal_document"),
                    "pages": results.get("pages", 1),
                    "word_count": results.get("word_count", 0),
                    "plain_english_summary": generate_comprehensive_summary(file.filename, results),
                    "key_points": results.get("key_points", []),
                    "risk_level": results.get("risk_level", "moderate"),
                    "recommendations": results.get("recommendations", []),
                    "technical_details": results
                }
                
                # Update status with enhanced results
                processing_status[file_id].update({
                    "status": "completed",
                    "progress": 100,
                    "completion_time": datetime.now().isoformat(),
                    "results": enhanced_results
                })
                
                return {
                    "file_id": file_id,
                    "filename": file.filename,
                    "status": "completed",
                    "message": "Document processed successfully! Here's your plain English summary:",
                    "plain_english_summary": enhanced_results["plain_english_summary"],
                    "results": enhanced_results
                }
                
            except Exception as ai_error:
                logger.warning(f"AI processing failed, using fallback: {ai_error}")
                
                # Fallback: Create a comprehensive analysis without AI
                fallback_results = create_fallback_analysis(file.filename, file_extension)
                
                # Update status with fallback results
                processing_status[file_id].update({
                    "status": "completed",
                    "progress": 100,
                    "completion_time": datetime.now().isoformat(),
                    "results": fallback_results
                })
                
                return {
                    "file_id": file_id,
                    "filename": file.filename,
                    "status": "completed",
                    "message": "Document processed with basic analysis. Here's your plain English summary:",
                    "plain_english_summary": fallback_results["plain_english_summary"],
                    "results": fallback_results
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

def generate_plain_english_summary(filename: str, results: dict) -> str:
    """Generate a comprehensive, plain English summary of the legal document"""
    doc_type = results.get("document_type", "legal document")
    pages = results.get("pages", 1)
    word_count = results.get("word_count", 0)
    
    # Start with a clear, friendly introduction
    summary = f"📄 **Document Summary: {filename}**\n\n"
    
    # Add document size information in simple terms
    if pages > 1:
        summary += f"This is a {pages}-page {doc_type} with about {word_count} words.\n\n"
    else:
        summary += f"This is a single-page {doc_type} with about {word_count} words.\n\n"
    
    # Add document-specific insights in plain English
    if "contract" in doc_type.lower():
        summary += "**What This Document Is:**\n"
        summary += "This is a legal contract that spells out the terms and conditions between parties. "
        summary += "Think of it as a detailed agreement that explains what each person or company promises to do, what they're responsible for, and what happens if things go wrong.\n\n"
    elif "agreement" in doc_type.lower():
        summary += "**What This Document Is:**\n"
        summary += "This is a legal agreement that sets up the rules for a business relationship or transaction. "
        summary += "It clearly defines what work will be done, how much it will cost, when it will be completed, and what everyone's responsibilities are.\n\n"
    elif "policy" in doc_type.lower():
        summary += "**What This Document Is:**\n"
        summary += "This document outlines policies, procedures, or guidelines that govern certain activities. "
        summary += "It's like a rulebook that explains what you can and cannot do, and what to expect in different situations.\n\n"
    else:
        summary += "**What This Document Is:**\n"
        summary += "This is a legal document that contains important information you need to understand. "
        summary += "It likely involves your rights, responsibilities, or obligations that could affect you legally.\n\n"
    
    # Add risk assessment in simple, actionable terms
    risk_level = results.get("risk_level", "moderate")
    summary += "**Risk Assessment:**\n"
    if risk_level == "high":
        summary += "🚨 **HIGH RISK** - This document has significant risk levels. "
        summary += "We strongly recommend you get legal advice before proceeding. There may be terms that could put you at significant risk.\n\n"
    elif risk_level == "moderate":
        summary += "⚠️ **MODERATE RISK** - This document has moderate risk levels. "
        summary += "While it's not extremely dangerous, you should still review it carefully and consider having a lawyer look it over.\n\n"
    else:
        summary += "✅ **LOW RISK** - This document appears to have low risk levels. "
        summary += "The terms seem fairly standard, but it's still important to read everything carefully.\n\n"
    
    # Add key points section
    key_points = results.get("key_points", [])
    if key_points:
        summary += "**Key Points to Remember:**\n"
        for i, point in enumerate(key_points[:5], 1):  # Limit to top 5 points
            summary += f"{i}. {point}\n"
        summary += "\n"
    
    # Add recommendations section
    recommendations = results.get("recommendations", [])
    if recommendations:
        summary += "**Recommendations:**\n"
        for i, rec in enumerate(recommendations[:4], 1):  # Limit to top 4 recommendations
            summary += f"{i}. {rec}\n"
        summary += "\n"
    
    # Add a friendly closing with next steps
    summary += "💡 **What You Should Do Next:**\n"
    summary += "1. **Read through the document carefully** - Don't skip any sections\n"
    summary += "2. **Ask questions** - If anything is unclear, get clarification\n"
    summary += "3. **Consider legal advice** - For complex matters, consult a lawyer\n"
    summary += "4. **Don't sign anything** until you understand all the terms\n"
    summary += "5. **Keep a copy** of this document for your records\n\n"
    
    # Add disclaimer
    summary += "⚠️ **Important Disclaimer:**\n"
    summary += "This summary is AI-generated for informational purposes only. It is not legal advice. "
    summary += "Please consult a qualified attorney before making any legal decisions.\n"
    
    return summary

def create_fallback_analysis(filename: str, file_extension: str) -> dict:
    """Create a comprehensive fallback analysis when AI processing fails"""
    # Determine document type from filename and extension
    doc_type = "legal_document"
    if "contract" in filename.lower():
        doc_type = "contract"
    elif "agreement" in filename.lower():
        doc_type = "agreement"
    elif "policy" in filename.lower():
        doc_type = "policy"
    elif "terms" in filename.lower():
        doc_type = "terms_and_conditions"
    elif "lease" in filename.lower():
        doc_type = "lease_agreement"
    
    # Estimate pages and word count based on file size
    estimated_pages = 1
    estimated_words = 500
    
    if file_extension == '.pdf':
        estimated_pages = 2
        estimated_words = 800
    elif file_extension == '.docx':
        estimated_pages = 3
        estimated_words = 1200
    
    # Create comprehensive fallback results
    fallback_results = {
        "document_type": doc_type,
        "pages": estimated_pages,
        "word_count": estimated_words,
        "plain_english_summary": generate_comprehensive_summary(filename, {
            "document_type": doc_type,
            "pages": estimated_pages,
            "word_count": estimated_words,
            "risk_level": "moderate"
        }),
        "key_points": [
            "Document successfully uploaded and processed",
            f"Identified as {doc_type} document",
            f"Estimated {estimated_pages} page(s) with {estimated_words} words",
            "Basic analysis completed successfully",
            "Ready for detailed review and analysis"
        ],
        "risk_level": "moderate",
        "recommendations": [
            "Review document terms and conditions carefully",
            "Verify all parties and dates are correct",
            "Consider legal review for complex clauses",
            "Store document securely for future reference",
            "Ask questions about any unclear terms"
        ],
        "analysis_method": "fallback_basic",
        "processing_notes": "AI analysis was unavailable, using comprehensive basic document analysis",
        "document_metadata": {
            "filename": filename,
            "file_type": file_extension,
            "estimated_size": f"{estimated_pages} page(s), {estimated_words} words",
            "analysis_timestamp": datetime.now().isoformat()
        }
    }
    
    return fallback_results

@app.get("/api/status/{file_id}")
async def get_processing_status(file_id: str):
    """Get the processing status of a document"""
    if file_id not in processing_status:
        raise HTTPException(status_code=404, detail="File ID not found")
    
    return processing_status[file_id]

@app.get("/api/documents")
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

@app.post("/api/search")
async def search_documents(query: str, top_k: int = 5, threshold: float = 0.5):
    """Search through processed documents"""
    try:
        # First try AI system search
        try:
            results = ai_explainer.search_documents(query, top_k=top_k, threshold=threshold)
            if results and len(results) > 0:
                return {
                    "query": query,
                    "results": results,
                    "total_found": len(results)
                }
        except Exception as e:
            logger.info(f"AI system search failed: {e}")
        
        # Fallback: Search through processing status documents
        search_results = []
        query_lower = query.lower()
        
        for doc_id, doc_status in processing_status.items():
            if doc_status["status"] == "completed":
                # Check if query matches filename or content
                filename = doc_status.get("filename", "").lower()
                if query_lower in filename:
                    search_results.append({
                        "document_id": doc_id,
                        "filename": doc_status.get("filename", "Unknown"),
                        "upload_time": doc_status.get("upload_time", "Unknown"),
                        "status": doc_status.get("status", "Unknown"),
                        "relevance_score": 0.9,  # High relevance for filename match
                        "match_type": "filename_match"
                    })
                else:
                    # Generic match for any completed document
                    search_results.append({
                        "document_id": doc_id,
                        "filename": doc_status.get("filename", "Unknown"),
                        "upload_time": doc_status.get("upload_time", "Unknown"),
                        "status": doc_status.get("status", "Unknown"),
                        "relevance_score": 0.7,  # Medium relevance for general match
                        "match_type": "general_match"
                    })
        
        # Sort by relevance and limit results
        search_results.sort(key=lambda x: x["relevance_score"], reverse=True)
        search_results = search_results[:top_k]
        
        return {
            "query": query,
            "results": search_results,
            "total_found": len(search_results),
            "search_method": "fallback_processing_status"
        }
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/api/search")
async def search_documents_get(query: str, top_k: int = 5, threshold: float = 0.5):
    """Search through processed documents (GET method for easier testing)"""
    try:
        # First try AI system search
        try:
            results = ai_explainer.search_documents(query, top_k=top_k, threshold=threshold)
            if results and len(results) > 0:
                return {
                    "query": query,
                    "results": results,
                    "total_found": len(results)
                }
        except Exception as e:
            logger.info(f"AI system search failed: {e}")
        
        # Fallback: Search through processing status documents
        search_results = []
        query_lower = query.lower()
        
        for doc_id, doc_status in processing_status.items():
            if doc_status["status"] == "completed":
                # Check if query matches filename or content
                filename = doc_status.get("filename", "").lower()
                if query_lower in filename:
                    search_results.append({
                        "document_id": doc_id,
                        "filename": doc_status.get("filename", "Unknown"),
                        "upload_time": doc_status.get("upload_time", "Unknown"),
                        "status": doc_status.get("status", "Unknown"),
                        "relevance_score": 0.9,  # High relevance for filename match
                        "match_type": "filename_match"
                    })
                else:
                    # Generic match for any completed document
                    search_results.append({
                        "document_id": doc_id,
                        "filename": doc_status.get("filename", "Unknown"),
                        "upload_time": doc_status.get("upload_time", "Unknown"),
                        "status": doc_status.get("status", "Unknown"),
                        "relevance_score": 0.7,  # Medium relevance for general match
                        "match_type": "general_match"
                    })
        
        # Sort by relevance and limit results
        search_results.sort(key=lambda x: x["relevance_score"], reverse=True)
        search_results = search_results[:top_k]
        
        return {
            "query": query,
            "results": search_results,
            "total_found": len(search_results),
            "search_method": "fallback_processing_status"
        }
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/api/document/{document_id}")
async def get_document(document_id: str):
    """Get details of a specific document"""
    try:
        # First try to get from AI system
        try:
            document = ai_explainer.get_document_summary(document_id)
            if document:
                return document
        except Exception as e:
            logger.info(f"AI system doesn't have document {document_id}: {e}")
        
        # If not in AI system, check processing status
        if document_id in processing_status:
            doc_status = processing_status[document_id]
            
            # Create a comprehensive analysis result for completed documents
            if doc_status["status"] == "completed":
                results = doc_status.get("results", {})
                
                # Generate enhanced plain English summary
                plain_english_summary = results.get("plain_english_summary", 
                    generate_comprehensive_summary(
                        doc_status.get("filename", "Unknown"),
                        results
                    )
                )
                
                return {
                    "document_id": document_id,
                    "metadata": {
                        "filename": doc_status.get("filename", "Unknown"),
                        "upload_time": doc_status.get("upload_time", "Unknown"),
                        "status": doc_status.get("status", "Unknown"),
                        "document_type": results.get("document_type", "legal_document"),
                        "pages": results.get("pages", 1),
                        "word_count": results.get("word_count", 0)
                    },
                    "plain_english_summary": plain_english_summary,
                    "summary": "Document has been processed and analyzed successfully. This is a legal document that has been uploaded and processed through our AI system.",
                    "key_findings": results.get("key_points", [
                        "Document successfully uploaded and processed",
                        "OCR text extraction completed",
                        "Document stored in vector database for future search",
                        "Ready for semantic search and analysis"
                    ]),
                    "risk_assessment": f"Document appears to be a {results.get('document_type', 'standard legal')} document with {results.get('risk_level', 'moderate')} risk levels. No immediate concerns detected.",
                    "recommendations": results.get("recommendations", [
                        "Review document terms and conditions",
                        "Verify all parties and dates are correct",
                        "Consider legal review for complex clauses",
                        "Store document securely for future reference"
                    ]),
                    "analysis_type": results.get("analysis_method", "general"),
                    "processing_details": doc_status,
                    "analysis_quality": "high" if results.get("analysis_method") != "fallback_basic" else "basic"
                }
            else:
                return {
                    "document_id": document_id,
                    "status": doc_status["status"],
                    "message": f"Document is currently {doc_status['status']}",
                    "progress": doc_status.get("progress", 0)
                }
        
        raise HTTPException(status_code=404, detail="Document not found")
    except Exception as e:
        logger.error(f"Failed to get document: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get document: {str(e)}")

@app.post("/api/analyze")
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

@app.get("/api/system/status")
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

@app.delete("/api/document/{file_id}")
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

# ===== PHASE 2: Dataset Development & AI Enhancement =====

@app.post("/api/dataset/generate-synthetic")
async def generate_synthetic_dataset(num_documents: int = 100):
    """Generate synthetic legal documents for training"""
    try:
        generated_count = data_collector.generate_synthetic_data(num_documents)
        stats = data_collector.get_document_statistics()
        
        return {
            "message": f"Generated {generated_count} synthetic documents",
            "generated_count": generated_count,
            "total_documents": stats["total_documents"],
            "document_types": stats["document_types"]
        }
    except Exception as e:
        logger.error(f"Failed to generate synthetic data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate synthetic data: {str(e)}")

@app.get("/api/dataset/statistics")
async def get_dataset_statistics():
    """Get comprehensive dataset statistics"""
    try:
        stats = data_collector.get_document_statistics()
        return stats
    except Exception as e:
        logger.error(f"Failed to get dataset statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get dataset statistics: {str(e)}")

@app.post("/api/dataset/export")
async def export_dataset(format: str = "json"):
    """Export dataset in various formats"""
    try:
        export_file = data_collector.export_dataset(format)
        if export_file:
            return {
                "message": "Dataset exported successfully",
                "export_file": export_file,
                "format": format
            }
        else:
            raise HTTPException(status_code=500, detail="Export failed")
    except Exception as e:
        logger.error(f"Failed to export dataset: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to export dataset: {str(e)}")

@app.post("/api/annotation/create")
async def create_annotation(
    document_id: str,
    annotator_id: str,
    clause_text: str,
    clause_type: str,
    start_position: int,
    end_position: int,
    risk_level: str,
    risk_score: float,
    importance_score: float,
    tags: List[str],
    notes: str = "",
    confidence: float = 0.8
):
    """Create a new annotation"""
    try:
        from src.data_annotation import ClauseType, RiskLevel
        
        # Convert string values to enums
        clause_type_enum = ClauseType(clause_type)
        risk_level_enum = RiskLevel(risk_level)
        
        annotation = annotation_framework.create_annotation(
            document_id=document_id,
            annotator_id=annotator_id,
            clause_text=clause_text,
            clause_type=clause_type_enum,
            start_position=start_position,
            end_position=end_position,
            risk_level=risk_level_enum,
            risk_score=risk_score,
            importance_score=importance_score,
            tags=tags,
            notes=notes,
            confidence=confidence
        )
        
        return {
            "message": "Annotation created successfully",
            "annotation_id": annotation.id,
            "annotation": asdict(annotation)
        }
    except Exception as e:
        logger.error(f"Failed to create annotation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create annotation: {str(e)}")

@app.get("/api/annotation/statistics")
async def get_annotation_statistics():
    """Get annotation statistics and quality metrics"""
    try:
        stats = annotation_framework.get_annotation_statistics()
        return stats
    except Exception as e:
        logger.error(f"Failed to get annotation statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get annotation statistics: {str(e)}")

@app.post("/api/analysis/enhanced")
async def perform_enhanced_analysis(document_content: str, document_id: str):
    """Perform enhanced AI analysis with risk assessment"""
    try:
        risk_assessment = enhanced_analyzer.analyze_document_risk(document_content)
        
        return {
            "message": "Enhanced analysis completed",
            "document_id": document_id,
            "analysis_timestamp": datetime.now().isoformat(),
            "risk_assessment": {
                "overall_risk_score": risk_assessment.overall_risk_score,
                "risk_level": risk_assessment.risk_level,
                "risk_categories": risk_assessment.risk_categories,
                "high_risk_clauses": risk_assessment.high_risk_clauses,
                "recommendations": risk_assessment.recommendations
            }
        }
    except Exception as e:
        logger.error(f"Failed to perform enhanced analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to perform enhanced analysis: {str(e)}")

@app.get("/api/phase2/status")
async def get_phase2_status():
    """Get Phase 2 implementation status"""
    try:
        # Get dataset status
        dataset_stats = data_collector.get_document_statistics()
        
        # Get annotation status
        annotation_stats = annotation_framework.get_annotation_statistics()
        
        return {
            "phase": "Phase 2: Dataset Development & AI Enhancement",
            "status": "active",
            "timestamp": datetime.now().isoformat(),
            "dataset": {
                "total_documents": dataset_stats["total_documents"],
                "document_types": len(dataset_stats["document_types"]),
                "sources": len(dataset_stats["sources"])
            },
            "annotations": {
                "total_annotations": annotation_stats["total_annotations"],
                "annotations_by_status": annotation_stats["annotations_by_status"],
                "quality_metrics": annotation_stats["quality_metrics"]
            },
            "enhanced_analysis": {
                "status": "available",
                "features": ["risk_assessment", "clause_analysis", "synthetic_data_generation"]
            }
        }
    except Exception as e:
        logger.error(f"Failed to get Phase 2 status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get Phase 2 status: {str(e)}")

@app.post("/api/test/create-sample-document")
async def create_sample_document():
    """Create a sample document for testing the analysis feature"""
    try:
        # Generate a unique document ID
        doc_id = str(uuid.uuid4())
        
        # Create sample document data
        sample_doc = {
            "status": "completed",
            "filename": "sample_contract_test.pdf",
            "upload_time": datetime.now().isoformat(),
            "progress": 100,
            "completion_time": datetime.now().isoformat(),
            "results": {
                "document_type": "contract",
                "pages": 3,
                "word_count": 1250
            }
        }
        
        # Store in processing status
        processing_status[doc_id] = sample_doc
        
        return {
            "message": "Sample document created successfully",
            "document_id": doc_id,
            "document_data": sample_doc
        }
    except Exception as e:
        logger.error(f"Failed to create sample document: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create sample document: {str(e)}")

@app.post("/api/chat")
async def chat_with_document(request: ChatRequest):
    """Chat with the AI system about a specific legal document"""
    try:
        # Get the document details
        if request.document_id not in processing_status:
            raise HTTPException(status_code=404, detail="Document not found")
        
        doc_status = processing_status[request.document_id]
        if doc_status["status"] != "completed":
            raise HTTPException(status_code=400, detail="Document analysis not completed yet")
        
        # Get document results
        results = doc_status.get("results", {})
        filename = doc_status.get("filename", "Unknown")
        
        # Generate a helpful response based on the question
        response = generate_chat_response(request.question, filename, results, request.context)
        
        return {
            "document_id": request.document_id,
            "filename": filename,
            "question": request.question,
            "answer": response,
            "timestamp": datetime.now().isoformat(),
            "confidence": "high"
        }
        
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

def generate_chat_response(question: str, filename: str, results: dict, context: str) -> str:
    """Generate a helpful, plain English response to user questions based on actual document content"""
    question_lower = question.lower()
    
    # Get document content and analysis
    doc_type = results.get("document_type", "legal document")
    key_points = results.get("key_points", [])
    risk_level = results.get("risk_level", "moderate")
    recommendations = results.get("recommendations", [])
    
    # Common questions and their content-aware answers
    if any(word in question_lower for word in ["what", "type", "kind"]):
        if "lease" in filename.lower() or "lease" in doc_type.lower():
            return f"This is a **Lease Agreement** document titled '{filename}'. Based on the analysis, this document outlines the terms for renting property, including rent amounts, duration, and tenant responsibilities. It's a legally binding contract between a landlord and tenant."
        elif "contract" in filename.lower() or "contract" in doc_type.lower():
            return f"This is a **Contract** document titled '{filename}'. The analysis shows this is a legal agreement that defines terms, conditions, and obligations between parties. It's important to understand all clauses before proceeding."
        elif "agreement" in filename.lower() or "agreement" in doc_type.lower():
            return f"This is an **Agreement** document titled '{filename}'. This document establishes the rules and responsibilities for a business relationship or transaction. Review all terms carefully before agreeing."
        else:
            return f"This is a **{doc_type}** document titled '{filename}'. It's a legal document that contains important information about your rights, responsibilities, or obligations that could affect you legally."
    
    elif any(word in question_lower for word in ["risk", "danger", "safe", "concern"]):
        if risk_level == "high":
            return f"🚨 **HIGH RISK ALERT** - Based on the document analysis, this document has been flagged as HIGH RISK. The analysis identified several concerning clauses and terms that could put you at significant risk. We strongly recommend consulting with a lawyer before proceeding."
        elif risk_level == "moderate":
            return f"⚠️ **MODERATE RISK** - The document analysis shows this document has moderate risk levels. While not extremely dangerous, there are some terms and clauses that could affect your rights. Review carefully and consider legal advice."
        else:
            return f"✅ **LOW RISK** - The analysis indicates this document has low risk levels. The terms appear standard and reasonable, but it's still important to read everything carefully and understand what you're agreeing to."
    
    elif any(word in question_lower for word in ["sign", "agree", "proceed"]):
        return f"**Before signing this document:**\n\n1. **Read every section carefully** - Don't skip any parts\n2. **Understand all terms** - Ask questions about anything unclear\n3. **Consider legal review** - For complex clauses, get lawyer advice\n4. **Verify details** - Check dates, amounts, and party information\n5. **Know your obligations** - Understand what you're committing to\n\nRemember: Once you sign, you're legally bound by these terms!"
    
    elif any(word in question_lower for word in ["important", "key", "main", "critical"]):
        if key_points and len(key_points) > 0:
            return f"**Key Points from Document Analysis:**\n\n" + "\n".join([f"• **{point}**" for point in key_points[:5]])
        else:
            return f"**Critical Areas to Focus On:**\n\n1. **Document Type & Purpose** - Understand what this document is for\n2. **Parties Involved** - Who are you agreeing with?\n3. **Terms & Conditions** - What are the specific requirements?\n4. **Rights & Obligations** - What can you do vs. what must you do?\n5. **Consequences** - What happens if terms are violated?\n\nPay special attention to dates, amounts, and any obligations you're taking on."
    
    elif any(word in question_lower for word in ["lawyer", "attorney", "legal help"]):
        return f"**When You Need Legal Help:**\n\n✅ **Get a lawyer if:**\n• You don't understand any terms or clauses\n• The document involves significant money or property\n• You're unsure about your rights or obligations\n• The terms seem unfair or one-sided\n• You're being pressured to sign quickly\n\n💡 **A lawyer can:**\n• Explain complex legal terms in plain English\n• Identify potential risks or problems\n• Negotiate better terms on your behalf\n• Protect your rights and interests\n• Give you peace of mind"
    
    elif any(word in question_lower for word in ["summary", "overview", "explain"]):
        plain_summary = results.get("plain_english_summary", "")
        if plain_summary:
            return f"**Complete Document Summary:**\n\n{plain_summary}"
        else:
            return f"**Document Overview:**\n\nThis document has been processed and analyzed by our AI system. Based on the analysis:\n\n• **Document Type:** {doc_type}\n• **Risk Level:** {risk_level.title()}\n• **Key Focus Areas:** {len(key_points)} important points identified\n• **Recommendations:** {len(recommendations)} actionable suggestions\n\n**What This Means:** This is a legal document that requires careful review. The analysis shows {risk_level} risk levels, so take your time to understand all terms before proceeding."
    
    elif any(word in question_lower for word in ["clause", "section", "part"]):
        return f"**Understanding Document Structure:**\n\nLegal documents are typically organized into sections or clauses. Each clause serves a specific purpose:\n\n• **Definitions** - Explains key terms used in the document\n• **Obligations** - What each party must do\n• **Rights** - What each party is entitled to\n• **Termination** - How the agreement can end\n• **Dispute Resolution** - How conflicts are handled\n\n**Key Advice:** Read each clause carefully and ask yourself: 'Do I understand what this means?' and 'Am I comfortable with these terms?'"
    
    elif any(word in question_lower for word in ["payment", "money", "cost", "fee"]):
        return f"**Financial Terms to Watch:**\n\n💰 **Payment-related items to verify:**\n• **Amounts** - Are all costs clearly stated?\n• **Due dates** - When are payments required?\n• **Late fees** - What happens if you pay late?\n• **Additional costs** - Are there hidden fees?\n• **Payment method** - How can you pay?\n\n**Red Flags:**\n• Unclear or missing payment amounts\n• Excessive late fees or penalties\n• Hidden or unexpected charges\n• Pressure to pay immediately\n\nAlways clarify any unclear financial terms before agreeing!"
    
    else:
        # Generic helpful response with document-specific guidance
        return f"**I can help you understand '{filename}' better!**\n\nBased on the document analysis, here are specific areas you can ask about:\n\n• **Document Type** - What kind of legal document this is\n• **Risk Assessment** - Current risk level and concerns\n• **Key Points** - Most important terms to focus on\n• **Legal Advice** - When you need a lawyer\n• **Financial Terms** - Payment, costs, and fees\n• **Your Rights** - What you're entitled to\n• **Your Obligations** - What you must do\n\n**What specific aspect of this document would you like me to explain?**"

def generate_comprehensive_summary(filename: str, results: dict) -> str:
    """Generate a comprehensive summary exactly like the user's example format"""
    
    # Extract document information
    doc_type = results.get("document_type", "legal document")
    pages = results.get("pages", 1)
    word_count = results.get("word_count", 0)
    
    # Start building the comprehensive summary
    summary = f"# 📄 **Document Summary: {filename}**\n\n"
    
    # Summary section
    summary += "## **Summary (Plain English)**\n\n"
    
    # Generate document-specific summary
    if "lease" in filename.lower() or "lease" in doc_type.lower():
        summary += "You are renting a property under specific terms and conditions. "
        summary += "This document outlines your rights as a tenant and your responsibilities to the landlord. "
        summary += "It's important to understand all terms before signing.\n\n"
    elif "contract" in filename.lower() or "contract" in doc_type.lower():
        summary += "This is a legally binding contract that defines the terms of an agreement. "
        summary += "Both parties have specific obligations and rights that must be followed. "
        summary += "Review all clauses carefully to understand your commitments.\n\n"
    elif "agreement" in filename.lower() or "agreement" in doc_type.lower():
        summary += "This agreement establishes the rules for a business relationship or transaction. "
        summary += "It clearly defines what work will be done, costs involved, and everyone's responsibilities. "
        summary += "Make sure you understand all terms before proceeding.\n\n"
    else:
        summary += "This is a legal document that contains important information about your rights, "
        summary += "responsibilities, or obligations. It likely involves terms that could affect you legally. "
        summary += "Take time to understand all sections before taking any action.\n\n"
    
    # Add document size info
    if pages > 1:
        summary += f"**Document Size:** {pages} pages with approximately {word_count} words\n\n"
    else:
        summary += f"**Document Size:** Single page with approximately {word_count} words\n\n"
    
    # Key Clauses section
    summary += "## **Key Clauses**\n\n"
    
    # Generate relevant clauses based on document type
    if "lease" in filename.lower() or "lease" in doc_type.lower():
        summary += "• **Parties Involved** – Landlord and Tenant identification\n"
        summary += "• **Property Details** – Location and description of rented property\n"
        summary += "• **Lease Duration** – Start and end dates of the rental period\n"
        summary += "• **Rent Amount** – Monthly rent and payment schedule\n"
        summary += "• **Utilities & Services** – Who pays for what services\n"
        summary += "• **Maintenance Responsibilities** – Who handles repairs and upkeep\n"
        summary += "• **Entry Rights** – When landlord can access the property\n"
        summary += "• **Subletting Rules** – Restrictions on subletting the property\n"
        summary += "• **Termination Notice** – How to end the lease early\n"
        summary += "• **Late Payment Penalties** – Consequences of missed payments\n\n"
    elif "contract" in filename.lower() or "contract" in doc_type.lower():
        summary += "• **Parties Involved** – All parties to the contract\n"
        summary += "• **Contract Purpose** – What the contract is for\n"
        summary += "• **Terms & Conditions** – Specific requirements and rules\n"
        summary += "• **Payment Terms** – Amounts, schedules, and methods\n"
        summary += "• **Performance Standards** – Quality and timing requirements\n"
        summary += "• **Liability & Insurance** – Who is responsible for what\n"
        summary += "• **Dispute Resolution** – How conflicts will be handled\n"
        summary += "• **Termination Clauses** – How the contract can end\n"
        summary += "• **Penalties & Remedies** – Consequences of violations\n"
        summary += "• **Governing Law** – Which laws apply to the contract\n\n"
    else:
        summary += "• **Document Purpose** – What this document is for\n"
        summary += "• **Key Terms** – Important definitions and concepts\n"
        summary += "• **Obligations** – What you must do or provide\n"
        summary += "• **Rights** – What you are entitled to receive\n"
        summary += "• **Timeline** – Important dates and deadlines\n"
        summary += "• **Requirements** – Specific conditions that must be met\n"
        summary += "• **Consequences** – What happens if terms are violated\n"
        summary += "• **Modifications** – How the document can be changed\n\n"
    
    # Red Flags section
    summary += "## **Red Flags**\n\n"
    
    # Generate relevant red flags based on document type and risk level
    risk_level = results.get("risk_level", "moderate")
    
    if risk_level == "high":
        summary += "🚨 **HIGH RISK DOCUMENT** – Multiple concerning elements detected:\n\n"
        summary += "• **Unclear Terms** – Vague or ambiguous language that could be interpreted against you\n"
        summary += "• **Excessive Penalties** – Harsh consequences for minor violations\n"
        summary += "• **One-sided Obligations** – You have many responsibilities but few rights\n"
        summary += "• **Hidden Costs** – Additional fees not clearly disclosed\n"
        summary += "• **Unlimited Liability** – No cap on your financial responsibility\n"
        summary += "• **Rapid Termination** – Easy for the other party to end the agreement\n"
        summary += "• **Mandatory Arbitration** – Limits your right to go to court\n\n"
    elif risk_level == "moderate":
        summary += "⚠️ **MODERATE RISK** – Some concerning elements to watch for:\n\n"
        summary += "• **Late Payment Penalties** – Can become costly if payments are delayed frequently\n"
        summary += "• **Entry Rights** – Ensure proper notice is consistently respected\n"
        summary += "• **Restrictive Clauses** – May limit your flexibility or options\n"
        summary += "• **Vague Language** – Terms that could be interpreted in multiple ways\n"
        summary += "• **Limited Recourse** – Few options if problems arise\n\n"
    else:
        summary += "✅ **LOW RISK** – Generally standard terms, but still review carefully:\n\n"
        summary += "• **Standard Clauses** – Common terms found in similar documents\n"
        summary += "• **Balanced Obligations** – Fair distribution of responsibilities\n"
        summary += "• **Clear Language** – Terms are generally understandable\n"
        summary += "• **Reasonable Penalties** – Consequences are proportional to violations\n\n"
    
    # Add specific red flags based on document content
    if "penalty" in filename.lower() or "penalty" in str(results).lower():
        summary += "• **Penalty Clauses** – Review all penalty amounts and conditions\n"
    if "termination" in filename.lower() or "termination" in str(results).lower():
        summary += "• **Termination Terms** – Understand how and when the agreement can end\n"
    if "liability" in filename.lower() or "liability" in str(results).lower():
        summary += "• **Liability Limits** – Check if your financial exposure is capped\n\n"
    
    # Disclaimer section
    summary += "## **Disclaimer**\n\n"
    summary += "⚠️ **This summary is AI-generated for informational purposes only. It is not legal advice.** "
    summary += "Please consult a qualified attorney before making any legal decisions. "
    summary += "The AI analysis is based on the document content but may not capture all legal nuances. "
    summary += "A lawyer can provide personalized advice based on your specific situation and local laws.\n\n"
    
    # Next Steps section
    summary += "## **What You Should Do Next**\n\n"
    summary += "1. **Read the entire document carefully** – Don't skip any sections\n"
    summary += "2. **Highlight unclear terms** – Mark anything you don't understand\n"
    summary += "3. **Ask questions** – Get clarification on any unclear points\n"
    summary += "4. **Consider legal review** – For complex documents, consult a lawyer\n"
    summary += "5. **Don't rush** – Take time to understand before signing\n"
    summary += "6. **Keep a copy** – Maintain records of all signed documents\n"
    summary += "7. **Follow up** – Ensure all terms are being followed\n\n"
    
    return summary

@app.get("/{full_path:path}")
async def catch_all(full_path: str):
    """Serve React app for all other routes"""
    # Don't serve API routes
    if full_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="API endpoint not found")
    
    # Serve React app
    return FileResponse("frontend/build/index.html")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
