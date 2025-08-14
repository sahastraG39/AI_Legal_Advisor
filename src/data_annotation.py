#!/usr/bin/env python3
"""
Data Annotation Framework for Phase 2
Handles legal document annotation, quality control, and annotation management
"""

import json
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import random

logger = logging.getLogger(__name__)

class AnnotationStatus(Enum):
    """Annotation status enumeration"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    REVIEWED = "reviewed"
    REJECTED = "rejected"

class RiskLevel(Enum):
    """Risk level enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ClauseType(Enum):
    """Legal clause type enumeration"""
    TERMINATION = "termination"
    LIABILITY = "liability"
    CONFIDENTIALITY = "confidentiality"
    PAYMENT = "payment"
    INTELLECTUAL_PROPERTY = "intellectual_property"
    DISPUTE_RESOLUTION = "dispute_resolution"
    FORCE_MAJEURE = "force_majeure"
    AMENDMENT = "amendment"
    ASSIGNMENT = "assignment"
    GOVERNING_LAW = "governing_law"

@dataclass
class Annotation:
    """Legal document annotation data structure"""
    id: str
    document_id: str
    annotator_id: str
    clause_text: str
    clause_type: ClauseType
    start_position: int
    end_position: int
    risk_level: RiskLevel
    risk_score: float
    importance_score: float
    tags: List[str]
    notes: str
    confidence: float
    status: AnnotationStatus
    created_at: str
    updated_at: str
    reviewed_by: Optional[str] = None
    review_notes: Optional[str] = None

@dataclass
class AnnotationSchema:
    """Annotation schema definition"""
    name: str
    version: str
    clause_types: List[Dict[str, Any]]
    risk_levels: List[Dict[str, Any]]
    tags: List[str]
    validation_rules: Dict[str, Any]
    created_at: str
    updated_at: str

class DataAnnotationFramework:
    """Main data annotation framework"""
    
    def __init__(self, annotation_dir: str = "data/annotations"):
        self.annotation_dir = Path(annotation_dir)
        self.annotation_dir.mkdir(parents=True, exist_ok=True)
        self.annotations_file = self.annotation_dir / "annotations.jsonl"
        self.schema_file = self.annotation_dir / "annotation_schema.json"
        self.annotations = []
        self.schema = self._load_or_create_schema()
        self.load_existing_annotations()
    
    def _load_or_create_schema(self) -> AnnotationSchema:
        """Load existing schema or create default one"""
        if self.schema_file.exists():
            try:
                with open(self.schema_file, 'r', encoding='utf-8') as f:
                    schema_data = json.load(f)
                    return AnnotationSchema(**schema_data)
            except Exception as e:
                logger.error(f"Error loading schema: {e}")
        
        # Create default schema
        default_schema = AnnotationSchema(
            name="Legal Document Annotation Schema v1.0",
            version="1.0.0",
            clause_types=[
                {
                    "type": "termination",
                    "description": "Clauses related to contract termination",
                    "examples": ["termination for cause", "termination without cause"],
                    "required_fields": ["risk_level", "importance_score"]
                },
                {
                    "type": "liability",
                    "description": "Clauses defining liability and responsibility",
                    "examples": ["limitation of liability", "indemnification"],
                    "required_fields": ["risk_level", "importance_score"]
                },
                {
                    "type": "confidentiality",
                    "description": "Clauses related to information confidentiality",
                    "examples": ["non-disclosure", "confidential information"],
                    "required_fields": ["risk_level", "importance_score"]
                }
            ],
            risk_levels=[
                {
                    "level": "low",
                    "description": "Minimal legal or financial risk",
                    "score_range": [0.0, 0.3]
                },
                {
                    "level": "medium",
                    "description": "Moderate legal or financial risk",
                    "score_range": [0.3, 0.6]
                },
                {
                    "level": "high",
                    "description": "Significant legal or financial risk",
                    "score_range": [0.6, 0.8]
                },
                {
                    "level": "critical",
                    "description": "Severe legal or financial risk",
                    "score_range": [0.8, 1.0]
                }
            ],
            tags=[
                "contract_formation", "performance", "breach", "remedies",
                "governing_law", "jurisdiction", "arbitration", "mediation"
            ],
            validation_rules={
                "min_confidence": 0.7,
                "required_fields": ["clause_type", "risk_level", "importance_score"],
                "risk_score_range": [0.0, 1.0],
                "importance_score_range": [0.0, 1.0]
            },
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )
        
        self._save_schema(default_schema)
        return default_schema
    
    def _save_schema(self, schema: AnnotationSchema):
        """Save annotation schema to file"""
        try:
            with open(self.schema_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(schema), f, indent=2, ensure_ascii=False)
            logger.info("Schema saved successfully")
        except Exception as e:
            logger.error(f"Error saving schema: {e}")
    
    def load_existing_annotations(self):
        """Load existing annotations from storage"""
        try:
            if self.annotations_file.exists():
                with open(self.annotations_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            ann_data = json.loads(line.strip())
                            # Convert enum values back
                            ann_data['clause_type'] = ClauseType(ann_data['clause_type'])
                            ann_data['risk_level'] = RiskLevel(ann_data['risk_level'])
                            ann_data['status'] = AnnotationStatus(ann_data['status'])
                            self.annotations.append(Annotation(**ann_data))
                logger.info(f"Loaded {len(self.annotations)} existing annotations")
        except Exception as e:
            logger.error(f"Error loading existing annotations: {e}")
    
    def create_annotation(self, document_id: str, annotator_id: str, 
                         clause_text: str, clause_type: ClauseType,
                         start_position: int, end_position: int,
                         risk_level: RiskLevel, risk_score: float,
                         importance_score: float, tags: List[str],
                         notes: str = "", confidence: float = 0.8) -> Annotation:
        """Create a new annotation"""
        annotation = Annotation(
            id=str(uuid.uuid4()),
            document_id=document_id,
            annotator_id=annotator_id,
            clause_text=clause_text,
            clause_type=clause_type,
            start_position=start_position,
            end_position=end_position,
            risk_level=risk_level,
            risk_score=risk_score,
            importance_score=importance_score,
            tags=tags,
            notes=notes,
            confidence=confidence,
            status=AnnotationStatus.PENDING,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )
        
        self.annotations.append(annotation)
        self._save_annotation(annotation)
        logger.info(f"Created annotation: {annotation.id}")
        return annotation
    
    def _save_annotation(self, annotation: Annotation):
        """Save annotation to storage"""
        try:
            # Convert enum values to strings for JSON serialization
            ann_dict = asdict(annotation)
            ann_dict['clause_type'] = annotation.clause_type.value
            ann_dict['risk_level'] = annotation.risk_level.value
            ann_dict['status'] = annotation.status.value
            
            with open(self.annotations_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(ann_dict, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(f"Error saving annotation: {e}")
    
    def update_annotation(self, annotation_id: str, updates: Dict[str, Any]) -> Optional[Annotation]:
        """Update an existing annotation"""
        annotation = self.get_annotation(annotation_id)
        if not annotation:
            return None
        
        # Update fields
        for key, value in updates.items():
            if hasattr(annotation, key):
                if key in ['clause_type', 'risk_level', 'status']:
                    # Handle enum conversions
                    if key == 'clause_type':
                        value = ClauseType(value)
                    elif key == 'risk_level':
                        value = RiskLevel(value)
                    elif key == 'status':
                        value = AnnotationStatus(value)
                setattr(annotation, key, value)
        
        annotation.updated_at = datetime.now().isoformat()
        
        # Re-save the annotation
        self._save_annotation(annotation)
        logger.info(f"Updated annotation: {annotation_id}")
        return annotation
    
    def get_annotation(self, annotation_id: str) -> Optional[Annotation]:
        """Get annotation by ID"""
        for annotation in self.annotations:
            if annotation.id == annotation_id:
                return annotation
        return None
    
    def get_document_annotations(self, document_id: str) -> List[Annotation]:
        """Get all annotations for a specific document"""
        return [ann for ann in self.annotations if ann.document_id == document_id]
    
    def get_annotator_annotations(self, annotator_id: str) -> List[Annotation]:
        """Get all annotations by a specific annotator"""
        return [ann for ann in self.annotations if ann.annotator_id == annotator_id]
    
    def review_annotation(self, annotation_id: str, reviewer_id: str, 
                         approved: bool, review_notes: str = "") -> bool:
        """Review an annotation"""
        annotation = self.get_annotation(annotation_id)
        if not annotation:
            return False
        
        if approved:
            annotation.status = AnnotationStatus.REVIEWED
        else:
            annotation.status = AnnotationStatus.REJECTED
        
        annotation.reviewed_by = reviewer_id
        annotation.review_notes = review_notes
        annotation.updated_at = datetime.now().isoformat()
        
        self._save_annotation(annotation)
        logger.info(f"Annotation {annotation_id} reviewed by {reviewer_id}")
        return True
    
    def get_annotation_statistics(self) -> Dict[str, Any]:
        """Get statistics about annotations"""
        stats = {
            "total_annotations": len(self.annotations),
            "annotations_by_status": {},
            "annotations_by_type": {},
            "annotations_by_risk": {},
            "annotations_by_annotator": {},
            "quality_metrics": {
                "average_confidence": 0.0,
                "average_risk_score": 0.0,
                "average_importance_score": 0.0
            }
        }
        
        if not self.annotations:
            return stats
        
        # Calculate statistics
        total_confidence = sum(ann.confidence for ann in self.annotations)
        total_risk_score = sum(ann.risk_score for ann in self.annotations)
        total_importance_score = sum(ann.importance_score for ann in self.annotations)
        
        stats["quality_metrics"]["average_confidence"] = total_confidence / len(self.annotations)
        stats["quality_metrics"]["average_risk_score"] = total_risk_score / len(self.annotations)
        stats["quality_metrics"]["average_importance_score"] = total_importance_score / len(self.annotations)
        
        for annotation in self.annotations:
            # Status counts
            status = annotation.status.value
            stats["annotations_by_status"][status] = stats["annotations_by_status"].get(status, 0) + 1
            
            # Type counts
            clause_type = annotation.clause_type.value
            stats["annotations_by_type"][clause_type] = stats["annotations_by_type"].get(clause_type, 0) + 1
            
            # Risk level counts
            risk_level = annotation.risk_level.value
            stats["annotations_by_risk"][risk_level] = stats["annotations_by_risk"].get(risk_level, 0) + 1
            
            # Annotator counts
            annotator = annotation.annotator_id
            stats["annotations_by_annotator"][annotator] = stats["annotations_by_annotator"].get(annotator, 0) + 1
        
        return stats
    
    def export_annotations(self, format: str = "json", output_file: Optional[str] = None):
        """Export annotations in various formats"""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.annotation_dir / f"annotations_export_{timestamp}.{format}"
        
        try:
            if format == "jsonl":
                with open(output_file, 'w', encoding='utf-8') as f:
                    for ann in self.annotations:
                        ann_dict = asdict(ann)
                        ann_dict['clause_type'] = ann.clause_type.value
                        ann_dict['risk_level'] = ann.risk_level.value
                        ann_dict['status'] = ann.status.value
                        f.write(json.dumps(ann_dict, ensure_ascii=False) + '\n')
            elif format == "json":
                with open(output_file, 'w', encoding='utf-8') as f:
                    annotations_list = []
                    for ann in self.annotations:
                        ann_dict = asdict(ann)
                        ann_dict['clause_type'] = ann.clause_type.value
                        ann_dict['risk_level'] = ann.risk_level.value
                        ann_dict['status'] = ann.status.value
                        annotations_list.append(ann_dict)
                    json.dump(annotations_list, f, indent=2, ensure_ascii=False)
            elif format == "csv":
                import csv
                with open(output_file, 'w', newline='', encoding='utf-8') as f:
                    if self.annotations:
                        # Get fieldnames from first annotation
                        first_ann = asdict(self.annotations[0])
                        first_ann['clause_type'] = self.annotations[0].clause_type.value
                        first_ann['risk_level'] = self.annotations[0].risk_level.value
                        first_ann['status'] = self.annotations[0].status.value
                        fieldnames = first_ann.keys()
                        
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        for ann in self.annotations:
                            ann_dict = asdict(ann)
                            ann_dict['clause_type'] = ann.clause_type.value
                            ann_dict['risk_level'] = ann.risk_level.value
                            ann_dict['status'] = ann.status.value
                            writer.writerow(ann_dict)
            
            logger.info(f"Annotations exported to: {output_file}")
            return str(output_file)
        except Exception as e:
            logger.error(f"Error exporting annotations: {e}")
            return None
    
    def validate_annotation(self, annotation: Annotation) -> Tuple[bool, List[str]]:
        """Validate annotation against schema rules"""
        errors = []
        
        # Check required fields
        for field in self.schema.validation_rules.get("required_fields", []):
            if not getattr(annotation, field, None):
                errors.append(f"Missing required field: {field}")
        
        # Check confidence threshold
        min_confidence = self.schema.validation_rules.get("min_confidence", 0.7)
        if annotation.confidence < min_confidence:
            errors.append(f"Confidence {annotation.confidence} below threshold {min_confidence}")
        
        # Check score ranges
        risk_range = self.schema.validation_rules.get("risk_score_range", [0.0, 1.0])
        if not (risk_range[0] <= annotation.risk_score <= risk_range[1]):
            errors.append(f"Risk score {annotation.risk_score} outside valid range {risk_range}")
        
        importance_range = self.schema.validation_rules.get("importance_score_range", [0.0, 1.0])
        if not (importance_range[0] <= annotation.importance_score <= importance_range[1]):
            errors.append(f"Importance score {annotation.importance_score} outside valid range {importance_range}")
        
        return len(errors) == 0, errors

if __name__ == "__main__":
    # Example usage
    framework = DataAnnotationFramework()
    
    # Create sample annotations
    print("Creating sample annotations...")
    
    # Sample annotation 1
    ann1 = framework.create_annotation(
        document_id="doc_001",
        annotator_id="annotator_001",
        clause_text="Either party may terminate this agreement with 30 days written notice.",
        clause_type=ClauseType.TERMINATION,
        start_position=100,
        end_position=150,
        risk_level=RiskLevel.MEDIUM,
        risk_score=0.6,
        importance_score=0.8,
        tags=["termination", "notice", "agreement"],
        notes="Standard termination clause with reasonable notice period"
    )
    
    # Sample annotation 2
    ann2 = framework.create_annotation(
        document_id="doc_001",
        annotator_id="annotator_001",
        clause_text="The Company shall not be liable for any indirect or consequential damages.",
        clause_type=ClauseType.LIABILITY,
        start_position=200,
        end_position=250,
        risk_level=RiskLevel.HIGH,
        risk_score=0.8,
        importance_score=0.9,
        tags=["liability", "damages", "limitation"],
        notes="Limitation of liability clause - high importance for risk assessment"
    )
    
    # Get statistics
    stats = framework.get_annotation_statistics()
    print(f"Annotation statistics: {json.dumps(stats, indent=2)}")
    
    # Export annotations
    print("Exporting annotations...")
    export_file = framework.export_annotations("json")
    print(f"Annotations exported to: {export_file}")
