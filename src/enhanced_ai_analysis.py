#!/usr/bin/env python3
"""
Enhanced AI Analysis System for Phase 2
Advanced features including risk scoring and clause importance ranking
"""

import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import re

logger = logging.getLogger(__name__)

@dataclass
class RiskAssessment:
    """Risk assessment result"""
    overall_risk_score: float
    risk_level: str
    risk_categories: Dict[str, float]
    high_risk_clauses: List[str]
    recommendations: List[str]

@dataclass
class ClauseAnalysis:
    """Individual clause analysis"""
    clause_text: str
    importance_score: float
    risk_score: float
    key_terms: List[str]
    recommendations: List[str]

class EnhancedAIAnalysis:
    """Enhanced AI analysis system with advanced features"""
    
    def __init__(self):
        self.risk_keywords = {
            "financial": ["damages", "penalties", "fines", "compensation", "payment"],
            "legal": ["termination", "breach", "default", "enforcement", "jurisdiction"],
            "operational": ["performance", "delivery", "timeline", "obligations"],
            "compliance": ["regulatory", "compliance", "standards", "requirements"]
        }
    
    def analyze_document_risk(self, document_content: str) -> RiskAssessment:
        """Analyze document risk comprehensively"""
        try:
            clauses = self._extract_clauses(document_content)
            
            # Calculate risk scores for each category
            risk_scores = {}
            high_risk_clauses = []
            
            for category, keywords in self.risk_keywords.items():
                category_score = self._calculate_category_risk(document_content, keywords)
                risk_scores[category] = category_score
                
                if category_score > 0.7:
                    high_risk_clauses.extend(self._identify_high_risk_clauses(clauses, category))
            
            # Calculate overall risk score
            overall_score = sum(risk_scores.values()) / len(risk_scores)
            risk_level = self._determine_risk_level(overall_score)
            
            # Generate recommendations
            recommendations = self._generate_risk_recommendations(risk_scores, high_risk_clauses)
            
            return RiskAssessment(
                overall_risk_score=overall_score,
                risk_level=risk_level,
                risk_categories=risk_scores,
                high_risk_clauses=high_risk_clauses,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error in risk analysis: {e}")
            return self._create_default_risk_assessment()
    
    def _extract_clauses(self, content: str) -> List[str]:
        """Extract individual clauses from document content"""
        clauses = []
        
        # Split by common clause indicators
        clause_patterns = [
            r"ARTICLE \d+[:\s]+([^A-Z]+?)(?=ARTICLE \d+|$)",
            r"Section \d+[:\s]+([^A-Z]+?)(?=Section \d+|$)",
            r"(\d+\.\s+[^0-9]+?)(?=\d+\.|$)"
        ]
        
        for pattern in clause_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
            clauses.extend([match.strip() for match in matches if match.strip()])
        
        # If no structured clauses found, split by paragraphs
        if not clauses:
            paragraphs = content.split('\n\n')
            clauses = [p.strip() for p in paragraphs if len(p.strip()) > 50]
        
        return clauses
    
    def _calculate_category_risk(self, content: str, keywords: List[str]) -> float:
        """Calculate risk score for a specific category"""
        if not keywords:
            return 0.0
        
        # Count keyword occurrences
        keyword_count = 0
        total_words = len(content.split())
        
        for keyword in keywords:
            pattern = re.compile(rf'\b{re.escape(keyword)}\b', re.IGNORECASE)
            matches = len(pattern.findall(content))
            keyword_count += matches
        
        # Calculate risk score based on keyword density
        base_score = min(keyword_count / max(total_words / 1000, 1), 1.0)
        
        # Adjust based on context
        if "damages" in content.lower():
            base_score *= 1.2
        if "termination" in content.lower():
            base_score *= 1.1
        
        return min(base_score, 1.0)
    
    def _identify_high_risk_clauses(self, clauses: List[str], category: str) -> List[str]:
        """Identify clauses with high risk for a category"""
        high_risk = []
        keywords = self.risk_keywords.get(category, [])
        
        for clause in clauses:
            clause_lower = clause.lower()
            risk_keywords_found = [kw for kw in keywords if kw.lower() in clause_lower]
            
            if len(risk_keywords_found) >= 2:
                high_risk.append(clause[:100] + "..." if len(clause) > 100 else clause)
        
        return high_risk[:5]
    
    def _determine_risk_level(self, score: float) -> str:
        """Determine risk level based on score"""
        if score >= 0.8:
            return "critical"
        elif score >= 0.6:
            return "high"
        elif score >= 0.4:
            return "medium"
        elif score >= 0.2:
            return "low"
        else:
            return "minimal"
    
    def _generate_risk_recommendations(self, risk_scores: Dict[str, float], 
                                     high_risk_clauses: List[str]) -> List[str]:
        """Generate risk mitigation recommendations"""
        recommendations = []
        
        if risk_scores.get("financial", 0) > 0.7:
            recommendations.append("Review and potentially limit liability clauses")
        
        if risk_scores.get("legal", 0) > 0.7:
            recommendations.append("Strengthen termination and breach provisions")
        
        if risk_scores.get("operational", 0) > 0.7:
            recommendations.append("Define clear performance metrics and timelines")
        
        if high_risk_clauses:
            recommendations.append(f"Review {len(high_risk_clauses)} high-risk clauses")
        
        return recommendations
    
    def _create_default_risk_assessment(self) -> RiskAssessment:
        """Create default risk assessment when analysis fails"""
        return RiskAssessment(
            overall_risk_score=0.5,
            risk_level="medium",
            risk_categories={},
            high_risk_clauses=[],
            recommendations=["Review document manually", "Re-run analysis"]
        )

if __name__ == "__main__":
    # Example usage
    analyzer = EnhancedAIAnalysis()
    
    sample_content = """
    EMPLOYMENT AGREEMENT
    
    ARTICLE 1: EMPLOYMENT
    The Company hereby employs the Employee as Software Engineer.
    
    ARTICLE 2: COMPENSATION
    The Employee shall receive an annual salary of $100,000.
    
    ARTICLE 3: TERMINATION
    Either party may terminate this Agreement with 30 days written notice.
    
    ARTICLE 4: LIABILITY
    The Company shall not be liable for any indirect damages.
    """
    
    # Perform enhanced analysis
    print("Performing enhanced analysis...")
    analysis = analyzer.analyze_document_risk(sample_content)
    
    print(f"Risk Level: {analysis.risk_level}")
    print(f"Overall Risk Score: {analysis.overall_risk_score:.2f}")
    print(f"Recommendations: {analysis.recommendations}")
