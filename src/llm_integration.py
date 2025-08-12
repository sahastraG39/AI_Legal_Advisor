"""
LLM Integration Module for AI Legal Document Explainer

This module handles integration with various Large Language Models:
- OpenAI GPT models
- Anthropic Claude models
- LangChain for orchestration
- Text embeddings and vector operations
"""

import logging
import os
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import json

# Environment variables
from dotenv import load_dotenv

# OpenAI integration
import openai

# Anthropic integration
import anthropic

# LangChain integration
from langchain.llms import OpenAI, Anthropic
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, SentenceTransformerEmbeddings
from langchain.schema import Document
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain, ConversationChain
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMIntegration:
    """Main LLM integration class for handling various AI models."""
    
    def __init__(self, 
                 openai_api_key: Optional[str] = None,
                 anthropic_api_key: Optional[str] = None,
                 default_model: str = 'openai'):
        """
        Initialize LLM integration.
        
        Args:
            openai_api_key: OpenAI API key (from env if not provided)
            anthropic_api_key: Anthropic API key (from env if not provided)
            default_model: Default model to use ('openai' or 'anthropic')
        """
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        self.anthropic_api_key = anthropic_api_key or os.getenv('ANTHROPIC_API_KEY')
        self.default_model = default_model
        
        # Initialize clients
        self._initialize_clients()
        
        # Initialize LangChain components
        self._initialize_langchain()
        
        # Text processing
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def _initialize_clients(self):
        """Initialize API clients."""
        try:
            if self.openai_api_key:
                openai.api_key = self.openai_api_key
                logger.info("OpenAI client initialized")
            else:
                logger.warning("OpenAI API key not found")
            
            if self.anthropic_api_key:
                self.anthropic_client = anthropic.Anthropic(api_key=self.anthropic_api_key)
                logger.info("Anthropic client initialized")
            else:
                logger.warning("Anthropic API key not found")
                
        except Exception as e:
            logger.error(f"Error initializing clients: {e}")
    
    def _initialize_langchain(self):
        """Initialize LangChain components."""
        try:
            # OpenAI models
            if self.openai_api_key:
                self.openai_llm = OpenAI(temperature=0.1, openai_api_key=self.openai_api_key)
                self.openai_chat = ChatOpenAI(temperature=0.1, openai_api_key=self.openai_api_key)
                self.openai_embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
            
            # Anthropic models
            if self.anthropic_api_key:
                self.anthropic_chat = ChatAnthropic(
                    temperature=0.1, 
                    anthropic_api_key=self.anthropic_api_key,
                    model="claude-3-sonnet-20240229"
                )
            
            # Local embeddings (fallback)
            self.local_embeddings = SentenceTransformerEmbeddings(
                model_name="all-MiniLM-L6-v2"
            )
            
            logger.info("LangChain components initialized")
            
        except Exception as e:
            logger.error(f"Error initializing LangChain: {e}")
    
    def analyze_legal_document(self, 
                              document_text: str,
                              analysis_type: str = 'general',
                              model: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze legal document using LLM.
        
        Args:
            document_text: Text content of the document
            analysis_type: Type of analysis ('general', 'contract', 'case_law', 'statute')
            model: Model to use ('openai', 'anthropic', or None for default)
            
        Returns:
            Dictionary containing analysis results
        """
        model = model or self.default_model
        
        # Create analysis prompt
        prompt = self._create_legal_analysis_prompt(analysis_type)
        
        try:
            if model == 'openai' and self.openai_api_key:
                result = self._analyze_with_openai(document_text, prompt)
            elif model == 'anthropic' and self.anthropic_api_key:
                result = self._analyze_with_anthropic(document_text, prompt)
            else:
                # Fallback to available model
                if self.openai_api_key:
                    result = self._analyze_with_openai(document_text, prompt)
                elif self.anthropic_api_key:
                    result = self._analyze_with_anthropic(document_text, prompt)
                else:
                    raise ValueError("No available LLM models")
            
            # Add metadata
            result['analysis_type'] = analysis_type
            result['model_used'] = model
            result['document_length'] = len(document_text)
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing document: {e}")
            return {
                'error': str(e),
                'analysis_type': analysis_type,
                'model_used': model
            }
    
    def _create_legal_analysis_prompt(self, analysis_type: str) -> str:
        """Create appropriate prompt for legal document analysis."""
        base_prompt = """
        You are an expert legal analyst. Analyze the following legal document and provide:
        
        1. Document Type and Purpose
        2. Key Legal Terms and Definitions
        3. Rights and Obligations
        4. Potential Legal Issues or Risks
        5. Summary and Recommendations
        
        Document Text:
        {document_text}
        
        Please provide a comprehensive analysis in a structured format.
        """
        
        if analysis_type == 'contract':
            base_prompt += "\n\nFocus on contract terms, conditions, and legal implications."
        elif analysis_type == 'case_law':
            base_prompt += "\n\nFocus on legal precedent, reasoning, and implications for future cases."
        elif analysis_type == 'statute':
            base_prompt += "\n\nFocus on statutory interpretation, scope, and application."
        
        return base_prompt
    
    def _analyze_with_openai(self, document_text: str, prompt: str) -> Dict[str, Any]:
        """Analyze document using OpenAI."""
        try:
            # Split text if too long
            if len(document_text) > 4000:
                chunks = self.text_splitter.split_text(document_text)
                analysis_parts = []
                
                for chunk in chunks:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are an expert legal analyst."},
                            {"role": "user", "content": prompt.format(document_text=chunk)}
                        ],
                        max_tokens=1000,
                        temperature=0.1
                    )
                    analysis_parts.append(response.choices[0].message.content)
                
                # Combine analyses
                combined_analysis = "\n\n".join(analysis_parts)
                
                # Get final summary
                summary_response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are an expert legal analyst."},
                        {"role": "user", "content": f"Provide a comprehensive summary of this legal analysis:\n\n{combined_analysis}"}
                    ],
                    max_tokens=1500,
                    temperature=0.1
                )
                
                return {
                    'analysis': combined_analysis,
                    'summary': summary_response.choices[0].message.content,
                    'provider': 'openai'
                }
            else:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are an expert legal analyst."},
                        {"role": "user", "content": prompt.format(document_text=document_text)}
                    ],
                    max_tokens=2000,
                    temperature=0.1
                )
                
                return {
                    'analysis': response.choices[0].message.content,
                    'provider': 'openai'
                }
                
        except Exception as e:
            logger.error(f"OpenAI analysis error: {e}")
            raise
    
    def _analyze_with_anthropic(self, document_text: str, prompt: str) -> Dict[str, Any]:
        """Analyze document using Anthropic Claude."""
        try:
            # Split text if too long
            if len(document_text) > 4000:
                chunks = self.text_splitter.split_text(document_text)
                analysis_parts = []
                
                for chunk in chunks:
                    response = self.anthropic_client.messages.create(
                        model="claude-3-sonnet-20240229",
                        max_tokens=1000,
                        temperature=0.1,
                        messages=[
                            {"role": "user", "content": prompt.format(document_text=chunk)}
                        ]
                    )
                    analysis_parts.append(response.content[0].text)
                
                # Combine analyses
                combined_analysis = "\n\n".join(analysis_parts)
                
                # Get final summary
                summary_response = self.anthropic_client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=1500,
                    temperature=0.1,
                    messages=[
                        {"role": "user", "content": f"Provide a comprehensive summary of this legal analysis:\n\n{combined_analysis}"}
                    ]
                )
                
                return {
                    'analysis': combined_analysis,
                    'summary': summary_response.content[0].text,
                    'provider': 'anthropic'
                }
            else:
                response = self.anthropic_client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=2000,
                    temperature=0.1,
                    messages=[
                        {"role": "user", "content": prompt.format(document_text=document_text)}
                    ]
                )
                
                return {
                    'analysis': response.content[0].text,
                    'provider': 'anthropic'
                }
                
        except Exception as e:
            logger.error(f"Anthropic analysis error: {e}")
            raise
    
    def generate_embeddings(self, text: str, model: str = 'local') -> List[float]:
        """
        Generate text embeddings.
        
        Args:
            text: Text to embed
            model: Embedding model ('openai', 'local')
            
        Returns:
            List of embedding values
        """
        try:
            if model == 'openai' and self.openai_api_key:
                embeddings = self.openai_embeddings.embed_query(text)
            else:
                embeddings = self.local_embeddings.embed_query(text)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def batch_generate_embeddings(self, texts: List[str], model: str = 'local') -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        try:
            if model == 'openai' and self.openai_api_key:
                embeddings = self.openai_embeddings.embed_documents(texts)
            else:
                embeddings = self.local_embeddings.embed_documents(texts)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            raise
    
    def create_conversation_chain(self, model: str = 'openai') -> ConversationChain:
        """Create a conversation chain for interactive legal analysis."""
        try:
            if model == 'openai' and self.openai_api_key:
                llm = self.openai_llm
            elif model == 'anthropic' and self.anthropic_api_key:
                llm = self.anthropic_chat
            else:
                raise ValueError(f"Model {model} not available")
            
            memory = ConversationBufferMemory()
            conversation = ConversationChain(llm=llm, memory=memory)
            
            return conversation
            
        except Exception as e:
            logger.error(f"Error creating conversation chain: {e}")
            raise
    
    def extract_legal_entities(self, text: str, model: str = 'openai') -> Dict[str, List[str]]:
        """
        Extract legal entities from text.
        
        Args:
            text: Text to analyze
            model: Model to use
            
        Returns:
            Dictionary of entity types and their values
        """
        prompt = """
        Extract legal entities from the following text. Identify and categorize:
        
        1. Parties (individuals, companies, organizations)
        2. Dates and deadlines
        3. Monetary amounts
        4. Legal references (statutes, case law, regulations)
        5. Jurisdictions
        6. Legal terms and definitions
        
        Text: {text}
        
        Return the results in JSON format.
        """
        
        try:
            if model == 'openai' and self.openai_api_key:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a legal entity extraction specialist. Return results in valid JSON format only."},
                        {"role": "user", "content": prompt.format(text=text)}
                    ],
                    max_tokens=1000,
                    temperature=0.1
                )
                
                result_text = response.choices[0].message.content
                
            elif model == 'anthropic' and self.anthropic_api_key:
                response = self.anthropic_client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=1000,
                    temperature=0.1,
                    messages=[
                        {"role": "user", "content": prompt.format(text=text)}
                    ]
                )
                
                result_text = response.content[0].text
            else:
                raise ValueError(f"Model {model} not available")
            
            # Try to parse JSON response
            try:
                # Clean the response to extract JSON
                if '{' in result_text and '}' in result_text:
                    start = result_text.find('{')
                    end = result_text.rfind('}') + 1
                    json_str = result_text[start:end]
                    entities = json.loads(json_str)
                else:
                    entities = {'error': 'Could not parse JSON response'}
            except json.JSONDecodeError:
                entities = {'error': 'Invalid JSON response', 'raw_response': result_text}
            
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting legal entities: {e}")
            return {'error': str(e)}


def main():
    """Test function for the LLM integration."""
    # Check if API keys are available
    openai_key = os.getenv('OPENAI_API_KEY')
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    
    if not openai_key and not anthropic_key:
        print("No API keys found. Please set OPENAI_API_KEY or ANTHROPIC_API_KEY in your .env file.")
        return
    
    try:
        llm = LLMIntegration()
        
        # Test with sample legal text
        sample_text = """
        This Agreement is made and entered into as of January 1, 2024, by and between 
        ABC Corporation, a Delaware corporation ("Company"), and John Doe ("Employee").
        
        The Company hereby employs the Employee as Chief Technology Officer, and the 
        Employee accepts such employment, subject to the terms and conditions set forth herein.
        
        The Employee shall be entitled to an annual salary of $150,000, payable in 
        accordance with the Company's standard payroll practices.
        """
        
        print("Testing LLM Integration...")
        print("=" * 50)
        
        # Test legal analysis
        if openai_key or anthropic_key:
            analysis = llm.analyze_legal_document(sample_text, 'contract')
            print(f"Legal Analysis Result:")
            print(f"Provider: {analysis.get('provider', 'Unknown')}")
            print(f"Analysis: {analysis.get('analysis', '')[:200]}...")
        
        # Test entity extraction
        entities = llm.extract_legal_entities(sample_text)
        print(f"\nLegal Entities Extracted:")
        print(json.dumps(entities, indent=2))
        
        # Test embeddings
        embeddings = llm.generate_embeddings(sample_text[:100])
        print(f"\nEmbeddings generated: {len(embeddings)} dimensions")
        
    except Exception as e:
        print(f"Error testing LLM integration: {e}")


if __name__ == "__main__":
    main()
