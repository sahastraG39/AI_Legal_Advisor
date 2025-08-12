# AI Legal Document Explainer

An intelligent system for parsing, analyzing, and explaining legal documents using AI and machine learning technologies.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Virtual environment (already set up)

### Environment Setup

1. **Activate the virtual environment:**
   ```bash
   # Windows (PowerShell)
   .\venv\Scripts\Activate.ps1
   
   # Windows (Command Prompt)
   .\venv\Scripts\activate.bat
   
   # macOS/Linux
   source venv/bin/activate
   ```

2. **Verify the environment:**
   ```bash
   python test_environment.py
   ```

3. **Set up environment variables:**
   - Copy `.env.example` to `.env`
   - Add your API keys for OpenAI, Anthropic, and Pinecone

## 📦 Installed Packages

### Core Document Processing
- **PyMuPDF** - PDF parsing and manipulation
- **pdfplumber** - PDF text extraction
- **pdf2image** - PDF to image conversion
- **pytesseract** - OCR capabilities
- **Pillow** - Image processing
- **OpenCV** - Computer vision and image analysis
- **python-docx** - Word document processing
- **openpyxl** - Excel file processing
- **pandas** - Data manipulation and analysis

### AI and LLM Integration
- **OpenAI** - GPT models integration
- **Anthropic** - Claude models integration
- **LangChain** - LLM orchestration framework
- **Sentence Transformers** - Text embeddings
- **Transformers** - Hugging Face models

### Vector Database and Search
- **Pinecone** - Cloud vector database
- **FAISS** - Local vector similarity search
- **ChromaDB** - Local vector database
- **Qdrant** - Vector similarity search engine

### Additional Utilities
- **FastAPI** - Web framework for APIs
- **Uvicorn** - ASGI server
- **pytest** - Testing framework
- **scikit-learn** - Machine learning utilities

## 🏗️ Project Structure

```
ai-legal-document-explainer/
├── venv/                          # Virtual environment
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
├── .env.example                   # Environment variables template
├── .gitignore                     # Git ignore file
├── test_environment.py            # Environment test script
└── README.md                      # This file
```

## 🔧 Development

### Adding New Dependencies
```bash
pip install new_package
pip freeze > requirements.txt
```

### Installing from Requirements
```bash
pip install -r requirements.txt
```

### Running Tests
```bash
pytest tests/
```

## 📚 Next Steps

1. **Configure API Keys**: Set up your OpenAI, Anthropic, and Pinecone API keys
2. **Design Prompts**: Create effective prompts for legal document analysis
3. **Build Core Modules**: Implement document parsing, OCR, and LLM integration
4. **Create Vector Store**: Set up document embedding and storage
5. **Build API**: Create FastAPI endpoints for document processing
6. **Add Testing**: Write comprehensive tests for all functionality

## 🆘 Troubleshooting

### Common Issues
- **Virtual environment not activating**: Check if activation scripts exist in `venv/Scripts/`
- **Package import errors**: Ensure virtual environment is activated
- **API key errors**: Verify `.env` file contains correct API keys

### Getting Help
- Check the `Python_Virtual_Environment_Setup_Guide.md` for detailed setup instructions
- Run `python test_environment.py` to verify your environment
- Ensure all packages are installed: `pip list`

## 📄 License

This project is for educational and development purposes.

---

**Happy coding! 🎯**
