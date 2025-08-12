#!/usr/bin/env python3
"""
Test script to verify the AI Legal Document Explainer environment setup.
Run this script to ensure all packages are properly installed and working.
"""

import sys
import importlib

def test_imports():
    """Test importing all required packages."""
    packages = [
        # PDF and Document Processing
        'fitz',  # PyMuPDF
        'pdfplumber',
        'pdf2image',
        'pytesseract',
        'PIL',  # Pillow
        'cv2',  # OpenCV
        'docx',  # python-docx
        'openpyxl',
        'pandas',
        
        # AI and LLM Integration
        'openai',
        'anthropic',
        'langchain',
        'langchain_openai',
        'langchain_anthropic',
        'langchain_community',
        'sentence_transformers',
        'transformers',
        
        # Vector Database and Search
        'pinecone',
        'faiss',
        'chromadb',
        'qdrant_client',
        
        # Additional Utilities
        'dotenv',
        'requests',
        'numpy',
        'sklearn',
        'fastapi',
        'uvicorn',
        'pytest',
    ]
    
    failed_imports = []
    successful_imports = []
    
    print("Testing package imports...")
    print("=" * 50)
    
    for package in packages:
        try:
            importlib.import_module(package)
            successful_imports.append(package)
            print(f"✅ {package}")
        except ImportError as e:
            failed_imports.append(package)
            print(f"❌ {package}: {e}")
    
    print("=" * 50)
    print(f"Successful imports: {len(successful_imports)}")
    print(f"Failed imports: {len(failed_imports)}")
    
    if failed_imports:
        print(f"\nFailed packages: {', '.join(failed_imports)}")
        return False
    else:
        print("\n🎉 All packages imported successfully!")
        return True

def test_basic_functionality():
    """Test basic functionality of key packages."""
    print("\nTesting basic functionality...")
    print("=" * 50)
    
    try:
        # Test PyMuPDF
        import fitz
        print("✅ PyMuPDF: Basic import successful")
        
        # Test pandas
        import pandas as pd
        df = pd.DataFrame({'test': [1, 2, 3]})
        print("✅ Pandas: DataFrame creation successful")
        
        # Test numpy
        import numpy as np
        arr = np.array([1, 2, 3])
        print("✅ NumPy: Array creation successful")
        
        # Test OpenAI client
        import openai
        print("✅ OpenAI: Client import successful")
        
        # Test LangChain
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        print("✅ LangChain: Text splitter import successful")
        
        # Test sentence transformers
        from sentence_transformers import SentenceTransformer
        print("✅ Sentence Transformers: Model import successful")
        
        print("\n🎉 Basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def main():
    """Main test function."""
    print("AI Legal Document Explainer - Environment Test")
    print("=" * 60)
    
    # Test Python version
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print()
    
    # Run tests
    imports_ok = test_imports()
    functionality_ok = test_basic_functionality()
    
    print("\n" + "=" * 60)
    if imports_ok and functionality_ok:
        print("🎉 ENVIRONMENT SETUP COMPLETE AND WORKING!")
        print("\nYou can now start building your AI Legal Document Explainer!")
    else:
        print("⚠️  Some issues were found. Please check the errors above.")
        print("\nYou may need to reinstall some packages or check dependencies.")
    
    return imports_ok and functionality_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
