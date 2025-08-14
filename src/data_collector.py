#!/usr/bin/env python3
"""
Data Collection and Management System for Phase 2
Handles legal document dataset collection, synthetic data generation, and data management
"""

import os
import json
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from dataclasses import dataclass, asdict
import requests
from bs4 import BeautifulSoup
import random

logger = logging.getLogger(__name__)

@dataclass
class LegalDocument:
    """Legal document data structure"""
    id: str
    title: str
    content: str
    document_type: str
    source: str
    collection_date: str
    metadata: Dict[str, Any]
    tags: List[str]
    risk_level: Optional[str] = None
    confidence_score: Optional[float] = None

class DataCollector:
    """Main data collection and management system"""
    
    def __init__(self, data_dir: str = "data/legal_documents"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.documents_file = self.data_dir / "documents.jsonl"
        self.metadata_file = self.data_dir / "metadata.json"
        self.documents = []
        self.load_existing_data()
    
    def load_existing_data(self):
        """Load existing documents from storage"""
        try:
            if self.documents_file.exists():
                with open(self.documents_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            doc_data = json.loads(line.strip())
                            self.documents.append(LegalDocument(**doc_data))
                logger.info(f"Loaded {len(self.documents)} existing documents")
        except Exception as e:
            logger.error(f"Error loading existing data: {e}")
    
    def save_document(self, document: LegalDocument):
        """Save a document to storage"""
        try:
            with open(self.documents_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(asdict(document), ensure_ascii=False) + '\n')
            self.documents.append(document)
            logger.info(f"Saved document: {document.title}")
        except Exception as e:
            logger.error(f"Error saving document: {e}")
    
    def collect_from_public_sources(self):
        """Collect legal documents from public sources"""
        sources = [
            "https://www.contracts.gov/",
            "https://www.law.cornell.edu/",
            "https://www.findlaw.com/"
        ]
        
        collected_count = 0
        for source in sources:
            try:
                logger.info(f"Collecting from: {source}")
                # This is a simplified example - in production you'd implement proper web scraping
                # with rate limiting, robots.txt compliance, etc.
                collected_count += self._scrape_source(source)
            except Exception as e:
                logger.error(f"Error collecting from {source}: {e}")
        
        return collected_count
    
    def _scrape_source(self, source_url: str) -> int:
        """Scrape documents from a source (simplified implementation)"""
        try:
            response = requests.get(source_url, timeout=10)
            if response.status_code == 200:
                # Parse and extract relevant information
                # This is a placeholder - implement actual scraping logic
                return 0
        except Exception as e:
            logger.error(f"Error scraping {source_url}: {e}")
        return 0
    
    def generate_synthetic_data(self, num_documents: int = 100):
        """Generate synthetic legal documents for training"""
        contract_templates = [
            "employment_agreement",
            "non_disclosure_agreement", 
            "service_contract",
            "lease_agreement",
            "purchase_agreement"
        ]
        
        generated_count = 0
        for i in range(num_documents):
            try:
                doc = self._create_synthetic_document(
                    template=random.choice(contract_templates),
                    doc_id=f"synthetic_{i+1}"
                )
                self.save_document(doc)
                generated_count += 1
            except Exception as e:
                logger.error(f"Error generating synthetic document {i+1}: {e}")
        
        logger.info(f"Generated {generated_count} synthetic documents")
        return generated_count
    
    def _create_synthetic_document(self, template: str, doc_id: str) -> LegalDocument:
        """Create a synthetic legal document based on template"""
        templates = {
            "employment_agreement": {
                "title": "Employment Agreement",
                "content": self._generate_employment_agreement(),
                "document_type": "employment_contract",
                "tags": ["employment", "contract", "legal", "hr"]
            },
            "non_disclosure_agreement": {
                "title": "Non-Disclosure Agreement",
                "content": self._generate_nda(),
                "document_type": "nda",
                "tags": ["nda", "confidentiality", "legal", "business"]
            },
            "service_contract": {
                "title": "Service Contract",
                "content": self._generate_service_contract(),
                "document_type": "service_contract",
                "tags": ["service", "contract", "legal", "business"]
            }
        }
        
        template_data = templates.get(template, templates["employment_agreement"])
        
        return LegalDocument(
            id=doc_id,
            title=template_data["title"],
            content=template_data["content"],
            document_type=template_data["document_type"],
            source="synthetic_generation",
            collection_date=datetime.now().isoformat(),
            metadata={
                "template": template,
                "generation_method": "synthetic",
                "version": "1.0"
            },
            tags=template_data["tags"],
            risk_level=random.choice(["low", "medium", "high"]),
            confidence_score=random.uniform(0.7, 0.95)
        )
    
    def _generate_employment_agreement(self) -> str:
        """Generate synthetic employment agreement content"""
        return """
        EMPLOYMENT AGREEMENT
        
        This Employment Agreement (the "Agreement") is entered into on [DATE] between [COMPANY NAME] (the "Company") and [EMPLOYEE NAME] (the "Employee").
        
        ARTICLE 1: EMPLOYMENT
        The Company hereby employs the Employee as [POSITION] and the Employee accepts such employment on the terms and conditions set forth in this Agreement.
        
        ARTICLE 2: COMPENSATION
        The Employee shall receive an annual salary of [SALARY] payable in accordance with the Company's standard payroll practices.
        
        ARTICLE 3: TERM
        This Agreement shall commence on [START DATE] and shall continue until terminated by either party in accordance with the terms herein.
        
        ARTICLE 4: DUTIES
        The Employee shall perform all duties and responsibilities associated with the position of [POSITION] and such other duties as may be assigned by the Company.
        
        ARTICLE 5: TERMINATION
        Either party may terminate this Agreement with [NOTICE PERIOD] written notice to the other party.
        """
    
    def _generate_nda(self) -> str:
        """Generate synthetic NDA content"""
        return """
        NON-DISCLOSURE AGREEMENT
        
        This Non-Disclosure Agreement (the "Agreement") is entered into on [DATE] between [COMPANY NAME] (the "Disclosing Party") and [RECIPIENT NAME] (the "Receiving Party").
        
        ARTICLE 1: CONFIDENTIAL INFORMATION
        The Receiving Party acknowledges that it may receive confidential and proprietary information from the Disclosing Party.
        
        ARTICLE 2: NON-DISCLOSURE
        The Receiving Party agrees to maintain the confidentiality of all confidential information and not to disclose such information to any third party.
        
        ARTICLE 3: USE RESTRICTIONS
        The Receiving Party shall use the confidential information solely for the purpose of [PURPOSE] and shall not use it for any other purpose.
        
        ARTICLE 4: TERM
        This Agreement shall remain in effect for [DURATION] from the date of execution.
        """
    
    def _generate_service_contract(self) -> str:
        """Generate synthetic service contract content"""
        return """
        SERVICE CONTRACT
        
        This Service Contract (the "Contract") is entered into on [DATE] between [CLIENT NAME] (the "Client") and [SERVICE PROVIDER NAME] (the "Provider").
        
        ARTICLE 1: SERVICES
        The Provider shall provide the following services to the Client: [SERVICE DESCRIPTION]
        
        ARTICLE 2: COMPENSATION
        The Client shall pay the Provider [AMOUNT] for the services rendered under this Contract.
        
        ARTICLE 3: TERM
        This Contract shall commence on [START DATE] and shall continue until [END DATE] unless terminated earlier in accordance with the terms herein.
        
        ARTICLE 4: TERMINATION
        Either party may terminate this Contract with [NOTICE PERIOD] written notice to the other party.
        """
    
    def get_document_statistics(self) -> Dict[str, Any]:
        """Get statistics about collected documents"""
        stats = {
            "total_documents": len(self.documents),
            "document_types": {},
            "sources": {},
            "risk_levels": {},
            "tags": {},
            "collection_timeline": {}
        }
        
        for doc in self.documents:
            # Document types
            doc_type = doc.document_type
            stats["document_types"][doc_type] = stats["document_types"].get(doc_type, 0) + 1
            
            # Sources
            source = doc.source
            stats["sources"][source] = stats["sources"].get(source, 0) + 1
            
            # Risk levels
            if doc.risk_level:
                risk = doc.risk_level
                stats["risk_levels"][risk] = stats["risk_levels"].get(risk, 0) + 1
            
            # Tags
            for tag in doc.tags:
                stats["tags"][tag] = stats["tags"].get(tag, 0) + 1
            
            # Collection timeline
            date = doc.collection_date[:10]  # YYYY-MM-DD
            stats["collection_timeline"][date] = stats["collection_timeline"].get(date, 0) + 1
        
        return stats
    
    def export_dataset(self, format: str = "jsonl", output_file: Optional[str] = None):
        """Export the dataset in various formats"""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.data_dir / f"legal_documents_export_{timestamp}.{format}"
        
        try:
            if format == "jsonl":
                with open(output_file, 'w', encoding='utf-8') as f:
                    for doc in self.documents:
                        f.write(json.dumps(asdict(doc), ensure_ascii=False) + '\n')
            elif format == "json":
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump([asdict(doc) for doc in self.documents], f, indent=2, ensure_ascii=False)
            elif format == "csv":
                import csv
                with open(output_file, 'w', newline='', encoding='utf-8') as f:
                    if self.documents:
                        writer = csv.DictWriter(f, fieldnames=asdict(self.documents[0]).keys())
                        writer.writeheader()
                        for doc in self.documents:
                            writer.writerow(asdict(doc))
            
            logger.info(f"Dataset exported to: {output_file}")
            return str(output_file)
        except Exception as e:
            logger.error(f"Error exporting dataset: {e}")
            return None

if __name__ == "__main__":
    # Example usage
    collector = DataCollector()
    
    # Generate synthetic data
    print("Generating synthetic legal documents...")
    collector.generate_synthetic_data(50)
    
    # Get statistics
    stats = collector.get_document_statistics()
    print(f"Dataset statistics: {json.dumps(stats, indent=2)}")
    
    # Export dataset
    print("Exporting dataset...")
    export_file = collector.export_dataset("json")
    print(f"Dataset exported to: {export_file}")
