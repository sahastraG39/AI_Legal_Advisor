# Python Virtual Environment Setup Guide for AI Legal Document Explainer

This guide provides step-by-step instructions for creating and managing a Python virtual environment specifically designed for the AI Legal Document Explainer project.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Creating the Virtual Environment](#creating-the-virtual-environment)
3. [Activating the Virtual Environment](#activating-the-virtual-environment)
4. [Installing Essential Packages](#installing-essential-packages)
5. [Managing Dependencies](#managing-dependencies)
6. [Deactivating and Removing the Environment](#deactivating-and-removing-the-environment)
7. [Troubleshooting](#troubleshooting)

## Prerequisites

Before starting, ensure you have:
- Python 3.8+ installed on your system
- pip (Python package installer) available
- Access to command line/terminal

To check your Python version:
```bash
python --version
# or
python3 --version
```

## Creating the Virtual Environment

### Method 1: Using venv (Recommended)
Navigate to your project directory and create the virtual environment:

```bash
# Navigate to your project directory
cd /path/to/your/ai-legal-document-explainer

# Create virtual environment
python -m venv venv
# or
python3 -m venv venv
```

### Method 2: Using virtualenv (Alternative)
If you prefer virtualenv or venv is not available:

```bash
# Install virtualenv if not already installed
pip install virtualenv

# Create virtual environment
virtualenv venv
```

## Activating the Virtual Environment

### Windows (PowerShell/Command Prompt)
```powershell
# PowerShell
.\venv\Scripts\Activate.ps1

# Command Prompt
.\venv\Scripts\activate.bat
```

### macOS and Linux
```bash
source venv/bin/activate
```

### Verification
After activation, you should see `(venv)` at the beginning of your command prompt:

```bash
(venv) C:\Users\username\project>
# or
(venv) username@machine:~/project$
```

## Installing Essential Packages

With your virtual environment activated, install the required packages:

### Core PDF and Document Processing
```bash
# PDF parsing and manipulation
pip install PyMuPDF
pip install pdfplumber
pip install pdf2image

# OCR capabilities
pip install pytesseract
pip install Pillow
pip install opencv-python

# Document text extraction
pip install python-docx
pip install openpyxl
pip install pandas
```

### AI and LLM Integration
```bash
# OpenAI integration
pip install openai

# Anthropic integration
pip install anthropic

# LangChain for LLM orchestration
pip install langchain
pip install langchain-openai
pip install langchain-anthropic
pip install langchain-community

# Vector embeddings
pip install sentence-transformers
pip install transformers
```

### Vector Database and Search
```bash
# Pinecone vector database
pip install pinecone-client

# FAISS for local vector search
pip install faiss-cpu
# or for GPU support
# pip install faiss-gpu

# Alternative vector databases
pip install chromadb
pip install qdrant-client
```

### Additional Utilities
```bash
# Environment management
pip install python-dotenv

# HTTP requests
pip install requests

# Data processing
pip install numpy
pip install scikit-learn

# Web framework (if building API)
pip install fastapi
pip install uvicorn

# Testing
pip install pytest
pip install pytest-asyncio
```

### Install All at Once
You can also install multiple packages in a single command:

```bash
pip install PyMuPDF pdfplumber pytesseract Pillow openai anthropic langchain langchain-openai langchain-anthropic sentence-transformers pinecone-client faiss-cpu chromadb python-dotenv requests numpy fastapi uvicorn pytest
```

## Managing Dependencies

### Creating requirements.txt
After installing all necessary packages, create a requirements file:

```bash
pip freeze > requirements.txt
```

### Installing from requirements.txt
To install all dependencies in a new environment:

```bash
pip install -r requirements.txt
```

### Updating Dependencies
To update all packages to their latest versions:

```bash
pip install --upgrade -r requirements.txt
```

### Best Practices
1. **Pin versions**: Use specific versions for production stability
2. **Regular updates**: Update requirements.txt after adding new packages
3. **Clean environment**: Start with a fresh environment for each major change
4. **Documentation**: Keep notes on why specific packages were chosen

## Deactivating and Removing the Environment

### Deactivating
To exit the virtual environment:

```bash
deactivate
```

### Removing the Environment
To completely remove the virtual environment:

```bash
# Windows
rmdir /s venv

# macOS/Linux
rm -rf venv
```

**Note**: This will permanently delete the environment and all installed packages. You can always recreate it using the steps above.

## Project Structure Recommendation

After setting up your environment, consider organizing your project like this:

```
ai-legal-document-explainer/
├── venv/                          # Virtual environment (don't commit)
├── src/                           # Source code
│   ├── __init__.py
│   ├── document_parser.py        # PDF/document parsing
│   ├── ocr_processor.py          # OCR functionality
│   ├── llm_integration.py        # AI/LLM integration
│   ├── vector_store.py           # Vector database operations
│   └── main.py                   # Main application
├── tests/                         # Test files
├── data/                          # Sample documents
├── requirements.txt               # Dependencies
├── .env                          # Environment variables (don't commit)
├── .gitignore                    # Git ignore file
└── README.md                     # Project documentation
```

## Environment Variables Setup

Create a `.env` file in your project root:

```bash
# .env file
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=your_pinecone_environment
```

Load these in your Python code:

```python
from dotenv import load_dotenv
import os

load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Permission Errors (Windows)
```bash
# Run PowerShell as Administrator
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### 2. Virtual Environment Not Activating
```bash
# Check if activation script exists
ls venv/Scripts/  # Windows
ls venv/bin/      # macOS/Linux
```

#### 3. Package Installation Failures
```bash
# Upgrade pip first
pip install --upgrade pip

# Clear pip cache
pip cache purge

# Try installing with --user flag
pip install --user package_name
```

#### 4. Tesseract OCR Issues
- **Windows**: Download and install Tesseract from GitHub releases
- **macOS**: `brew install tesseract`
- **Linux**: `sudo apt-get install tesseract-ocr`

#### 5. PyMuPDF Installation Issues
```bash
# Alternative installation method
pip install --upgrade pip setuptools wheel
pip install PyMuPDF
```

## Next Steps

After completing this setup:

1. **Test your environment**: Create a simple test script to verify all packages work
2. **Configure your IDE**: Set your IDE to use the virtual environment's Python interpreter
3. **Start development**: Begin building your AI Legal Document Explainer features
4. **Version control**: Initialize git and commit your requirements.txt and project structure

## Additional Resources

- [Python venv documentation](https://docs.python.org/3/library/venv.html)
- [pip user guide](https://pip.pypa.io/en/stable/user_guide/)
- [LangChain documentation](https://python.langchain.com/)
- [PyMuPDF documentation](https://pymupdf.readthedocs.io/)

---

**Remember**: Always activate your virtual environment before working on your project, and keep your requirements.txt updated as you add new dependencies.
