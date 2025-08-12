"""
Main Application for AI Legal Document Explainer

This is the main entry point that orchestrates all the components:
- Document parsing
- OCR processing
- LLM integration
- Vector storage and search
"""

import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import time

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Import our modules
from document_parser import DocumentParser
from ocr_processor import OCRProcessor
from llm_integration import LLMIntegration
from vector_store import VectorStore

# Environment variables
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AILegalDocumentExplainer:
    """Main class that orchestrates the AI Legal Document Explainer system."""
    
    def __init__(self, 
                 vector_store_type: str = 'chroma',
                 default_llm: str = 'openai'):
        """
        Initialize the AI Legal Document Explainer.
        
        Args:
            vector_store_type: Type of vector store to use
            default_llm: Default LLM provider to use
        """
        logger.info("Initializing AI Legal Document Explainer...")
        
        # Initialize components
        self.document_parser = DocumentParser()
        self.ocr_processor = OCRProcessor()
        self.llm_integration = LLMIntegration(default_model=default_llm)
        self.vector_store = VectorStore(store_type=vector_store_type)
        
        logger.info("AI Legal Document Explainer initialized successfully")
    
    def process_document(self, 
                        file_path: str,
                        analysis_type: str = 'general',
                        enable_ocr: bool = True,
                        store_in_vector_db: bool = True) -> Dict[str, Any]:
        """
        Process a legal document end-to-end.
        
        Args:
            file_path: Path to the document file
            analysis_type: Type of legal analysis to perform
            enable_ocr: Whether to enable OCR processing
            store_in_vector_db: Whether to store in vector database
            
        Returns:
            Dictionary containing processing results
        """
        start_time = time.time()
        results = {
            'file_path': file_path,
            'processing_time': 0,
            'success': False,
            'error': None,
            'parsing_results': None,
            'ocr_results': None,
            'llm_analysis': None,
            'vector_store_id': None
        }
        
        try:
            logger.info(f"Processing document: {file_path}")
            
            # Step 1: Parse document
            logger.info("Step 1: Parsing document...")
            parsing_results = self.document_parser.parse_document(file_path)
            results['parsing_results'] = parsing_results
            
            # Step 2: OCR processing (if enabled and needed)
            if enable_ocr and self._needs_ocr(parsing_results):
                logger.info("Step 2: Running OCR processing...")
                ocr_results = self._process_with_ocr(parsing_results)
                results['ocr_results'] = ocr_results
                
                # Update text content with OCR results
                if ocr_results:
                    parsing_results['content'] = self._combine_ocr_results(ocr_results)
            
            # Step 3: LLM analysis
            logger.info("Step 3: Running LLM analysis...")
            if parsing_results.get('content'):
                llm_analysis = self.llm_integration.analyze_legal_document(
                    parsing_results['content'],
                    analysis_type=analysis_type
                )
                results['llm_analysis'] = llm_analysis
            else:
                logger.warning("No text content available for LLM analysis")
            
            # Step 4: Store in vector database (if requested)
            if store_in_vector_db and parsing_results.get('content'):
                logger.info("Step 4: Storing in vector database...")
                vector_id = self._store_document_in_vector_db(
                    parsing_results['content'],
                    parsing_results,
                    llm_analysis if 'llm_analysis' in results else None
                )
                results['vector_store_id'] = vector_id
            
            results['success'] = True
            results['processing_time'] = time.time() - start_time
            
            logger.info(f"Document processing completed successfully in {results['processing_time']:.2f} seconds")
            
        except Exception as e:
            results['error'] = str(e)
            results['processing_time'] = time.time() - start_time
            logger.error(f"Error processing document: {e}")
        
        return results
    
    def _needs_ocr(self, parsing_results: Dict[str, Any]) -> bool:
        """Determine if OCR processing is needed."""
        # Check if the document has images or if text extraction was minimal
        content_length = len(parsing_results.get('content', ''))
        has_images = parsing_results.get('images') or parsing_results.get('text_blocks')
        
        # If content is very short or there are images, OCR might be needed
        return content_length < 100 or has_images
    
    def _process_with_ocr(self, parsing_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process document with OCR."""
        ocr_results = []
        
        try:
            # Process PDF images if available
            if parsing_results.get('images'):
                pdf_images = [img['image'] for img in parsing_results['images']]
                ocr_results = self.ocr_processor.process_pdf_images(pdf_images)
            
            # Process individual images if available
            elif parsing_results.get('text_blocks'):
                # This would need to be implemented based on the specific structure
                # of text blocks from the document parser
                pass
            
        except Exception as e:
            logger.error(f"Error in OCR processing: {e}")
        
        return ocr_results
    
    def _combine_ocr_results(self, ocr_results: List[Dict[str, Any]]) -> str:
        """Combine OCR results into a single text string."""
        combined_text = ""
        
        for result in ocr_results:
            if result.get('text'):
                combined_text += result['text'] + "\n"
        
        return combined_text.strip()
    
    def _store_document_in_vector_db(self, 
                                    content: str,
                                    parsing_results: Dict[str, Any],
                                    llm_analysis: Optional[Dict[str, Any]]) -> str:
        """Store document in vector database."""
        try:
            # Create metadata for vector storage
            metadata = {
                'file_path': parsing_results.get('file_path', ''),
                'file_size': parsing_results.get('file_size', 0),
                'file_extension': parsing_results.get('file_extension', ''),
                'total_pages': len(parsing_results.get('pages', [])),
                'total_paragraphs': len(parsing_results.get('paragraphs', [])),
                'total_tables': len(parsing_results.get('tables', [])),
                'processing_timestamp': time.time()
            }
            
            # Add LLM analysis metadata if available
            if llm_analysis:
                metadata.update({
                    'analysis_type': llm_analysis.get('analysis_type', ''),
                    'model_used': llm_analysis.get('model_used', ''),
                    'llm_provider': llm_analysis.get('provider', '')
                })
            
            # Store in vector database
            document_id = self.vector_store.add_document(
                text=content,
                metadata=metadata
            )
            
            logger.info(f"Document stored in vector database with ID: {document_id}")
            return document_id
            
        except Exception as e:
            logger.error(f"Error storing document in vector database: {e}")
            return None
    
    def search_documents(self, 
                        query: str,
                        top_k: int = 5,
                        threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            top_k: Number of top results
            threshold: Similarity threshold
            
        Returns:
            List of similar documents
        """
        try:
            logger.info(f"Searching for documents with query: {query}")
            results = self.vector_store.search_similar(query, top_k, threshold)
            logger.info(f"Found {len(results)} similar documents")
            return results
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    def get_document_summary(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get a summary of a specific document."""
        try:
            return self.vector_store.get_document(document_id)
        except Exception as e:
            logger.error(f"Error getting document summary: {e}")
            return None
    
    def batch_process_documents(self, 
                               file_paths: List[str],
                               analysis_type: str = 'general',
                               enable_ocr: bool = True) -> List[Dict[str, Any]]:
        """
        Process multiple documents in batch.
        
        Args:
            file_paths: List of file paths to process
            analysis_type: Type of legal analysis
            enable_ocr: Whether to enable OCR
            
        Returns:
            List of processing results
        """
        results = []
        
        logger.info(f"Starting batch processing of {len(file_paths)} documents")
        
        for i, file_path in enumerate(file_paths, 1):
            logger.info(f"Processing document {i}/{len(file_paths)}: {file_path}")
            
            try:
                result = self.process_document(
                    file_path=file_path,
                    analysis_type=analysis_type,
                    enable_ocr=enable_ocr,
                    store_in_vector_db=True
                )
                results.append(result)
                
                # Add progress information
                result['batch_progress'] = {
                    'current': i,
                    'total': len(file_paths),
                    'percentage': (i / len(file_paths)) * 100
                }
                
            except Exception as e:
                logger.error(f"Error processing document {file_path}: {e}")
                results.append({
                    'file_path': file_path,
                    'success': False,
                    'error': str(e),
                    'batch_progress': {
                        'current': i,
                        'total': len(file_paths),
                        'percentage': (i / len(file_paths)) * 100
                    }
                })
        
        logger.info(f"Batch processing completed. {len([r for r in results if r['success']])}/{len(results)} successful")
        return results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get the current status of the system."""
        try:
            status = {
                'system_status': 'operational',
                'components': {
                    'document_parser': 'initialized',
                    'ocr_processor': 'initialized',
                    'llm_integration': 'initialized',
                    'vector_store': 'initialized'
                },
                'vector_store_stats': self.vector_store.get_statistics(),
                'environment': {
                    'openai_available': bool(os.getenv('OPENAI_API_KEY')),
                    'anthropic_available': bool(os.getenv('ANTHROPIC_API_KEY')),
                    'pinecone_available': bool(os.getenv('PINECONE_API_KEY'))
                }
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {
                'system_status': 'error',
                'error': str(e)
            }
    
    def export_results(self, results: Dict[str, Any], format: str = 'json') -> str:
        """
        Export processing results.
        
        Args:
            results: Processing results to export
            format: Export format ('json', 'txt')
            
        Returns:
            Exported results as string
        """
        try:
            if format == 'json':
                return json.dumps(results, indent=2, default=str)
            elif format == 'txt':
                return self._format_results_as_text(results)
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Error exporting results: {e}")
            return f"Error exporting results: {e}"
    
    def _format_results_as_text(self, results: Dict[str, Any]) -> str:
        """Format results as human-readable text."""
        text = f"AI Legal Document Explainer - Results\n"
        text += "=" * 50 + "\n\n"
        
        text += f"File: {results.get('file_path', 'Unknown')}\n"
        text += f"Processing Time: {results.get('processing_time', 0):.2f} seconds\n"
        text += f"Status: {'Success' if results.get('success') else 'Failed'}\n\n"
        
        if results.get('error'):
            text += f"Error: {results['error']}\n\n"
        
        if results.get('llm_analysis'):
            text += "LLM Analysis:\n"
            text += "-" * 20 + "\n"
            text += results['llm_analysis'].get('analysis', 'No analysis available') + "\n\n"
        
        if results.get('vector_store_id'):
            text += f"Vector Store ID: {results['vector_store_id']}\n"
        
        return text


def main():
    """Main function for testing the AI Legal Document Explainer."""
    print("AI Legal Document Explainer - Main Application")
    print("=" * 60)
    
    try:
        # Initialize the system
        explainer = AILegalDocumentExplainer()
        
        # Get system status
        status = explainer.get_system_status()
        print("System Status:")
        print(json.dumps(status, indent=2, default=str))
        print()
        
        # Check for sample documents
        sample_files = [
            'sample.pdf',
            'sample.docx',
            'sample.txt'
        ]
        
        available_files = [f for f in sample_files if Path(f).exists()]
        
        if available_files:
            print(f"Found sample files: {available_files}")
            print("Would you like to process one? (y/n): ", end="")
            
            # For testing purposes, we'll process the first available file
            if available_files:
                sample_file = available_files[0]
                print(f"\nProcessing sample file: {sample_file}")
                
                results = explainer.process_document(
                    file_path=sample_file,
                    analysis_type='general',
                    enable_ocr=True,
                    store_in_vector_db=True
                )
                
                print("\nProcessing Results:")
                print(json.dumps(results, indent=2, default=str))
                
                # Export results
                export_text = explainer.export_results(results, 'txt')
                print(f"\nExported Results (Text):\n{export_text}")
        
        else:
            print("No sample files found. The system is ready to process documents.")
            print("\nTo test the system, place a document file in the project directory and run:")
            print("python src/main.py")
        
    except Exception as e:
        print(f"Error in main application: {e}")
        logger.error(f"Main application error: {e}")


if __name__ == "__main__":
    main()
