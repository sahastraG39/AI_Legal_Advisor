import React, { useState, useEffect } from 'react';
import { BarChart3, FileText, TrendingUp, AlertTriangle, CheckCircle, Clock, Download, MessageCircle, Send } from 'lucide-react';
import axios from 'axios';
import toast from 'react-hot-toast';
import { useParams } from 'react-router-dom';

const DocumentAnalysis = () => {
  const [documents, setDocuments] = useState([]);
  const [selectedDocument, setSelectedDocument] = useState(null);
  const [loading, setLoading] = useState(true);
  const [analyzing, setAnalyzing] = useState(false);
  const [chatMessage, setChatMessage] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [chatLoading, setChatLoading] = useState(false);
  const { fileId } = useParams(); // Get fileId from URL parameters

  useEffect(() => {
    fetchDocuments();
  }, []);

  useEffect(() => {
    // If fileId is provided in URL, automatically load that document
    if (fileId) {
      analyzeDocument(fileId);
    }
  }, [fileId]);

  const fetchDocuments = async () => {
    try {
      const response = await axios.get('/api/documents');
      const docs = response.data.processing_status || {};
      const completedDocs = Object.entries(docs)
        .filter(([_, doc]) => doc.status === 'completed')
        .map(([id, doc]) => ({ id, ...doc }));
      
      setDocuments(completedDocs);
      setLoading(false);
    } catch (error) {
      console.error('Failed to fetch documents:', error);
      toast.error('Failed to load documents');
      setLoading(false);
    }
  };

  const analyzeDocument = async (documentId) => {
    setAnalyzing(true);
    try {
      const response = await axios.get(`/api/document/${documentId}`);
      setSelectedDocument(response.data);
      toast.success('Document analysis loaded');
      // Clear chat history when switching documents
      setChatHistory([]);
    } catch (error) {
      console.error('Failed to analyze document:', error);
      toast.error('Failed to load document analysis');
    } finally {
      setAnalyzing(false);
    }
  };

  const sendChatMessage = async (e) => {
    e.preventDefault();
    if (!chatMessage.trim() || !selectedDocument) return;

    const userMessage = chatMessage.trim();
    setChatMessage('');
    
    // Add user message to chat history
    const newUserMessage = {
      id: Date.now(),
      type: 'user',
      content: userMessage,
      timestamp: new Date().toLocaleTimeString()
    };
    
    setChatHistory(prev => [...prev, newUserMessage]);
    setChatLoading(true);

    try {
      const response = await axios.post('/api/chat', {
        document_id: selectedDocument.document_id,
        question: userMessage,
        context: ''
      });

      // Add AI response to chat history
      const aiMessage = {
        id: Date.now() + 1,
        type: 'ai',
        content: response.data.answer,
        timestamp: new Date().toLocaleTimeString()
      };
      
      setChatHistory(prev => [...prev, aiMessage]);
      toast.success('AI response received');
    } catch (error) {
      console.error('Chat failed:', error);
      toast.error('Failed to get AI response');
      
      // Add error message to chat history
      const errorMessage = {
        id: Date.now() + 1,
        type: 'error',
        content: 'Sorry, I encountered an error. Please try again.',
        timestamp: new Date().toLocaleTimeString()
      };
      
      setChatHistory(prev => [...prev, errorMessage]);
    } finally {
      setChatLoading(false);
    }
  };

  const exportAnalysis = (format = 'json') => {
    if (!selectedDocument) return;
    
    let content, filename, mimeType;
    
    if (format === 'json') {
      content = JSON.stringify(selectedDocument, null, 2);
      filename = `analysis_${selectedDocument.metadata?.filename || 'document'}.json`;
      mimeType = 'application/json';
    } else {
      content = formatAnalysisAsText(selectedDocument);
      filename = `analysis_${selectedDocument.metadata?.filename || 'document'}.txt`;
      mimeType = 'text/plain';
    }
    
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    toast.success(`Analysis exported as ${format.toUpperCase()}`);
  };

  const formatAnalysisAsText = (doc) => {
    let text = `Document Analysis Report\n`;
    text += `========================\n\n`;
    text += `Filename: ${doc.metadata?.filename || 'Unknown'}\n`;
    text += `Upload Time: ${doc.metadata?.upload_time || 'Unknown'}\n`;
    text += `Analysis Type: ${doc.analysis_type || 'General'}\n\n`;
    
    if (doc.summary) {
      text += `Summary:\n${doc.summary}\n\n`;
    }
    
    if (doc.key_findings) {
      text += `Key Findings:\n`;
      doc.key_findings.forEach((finding, index) => {
        text += `${index + 1}. ${finding}\n`;
      });
      text += '\n';
    }
    
    if (doc.risk_assessment) {
      text += `Risk Assessment:\n${doc.risk_assessment}\n\n`;
    }
    
    if (doc.recommendations) {
      text += `Recommendations:\n`;
      doc.recommendations.forEach((rec, index) => {
        text += `${index + 1}. ${rec}\n`;
      });
    }
    
    return text;
  };

  const getRiskLevel = (riskScore) => {
    if (riskScore >= 0.8) return { level: 'High', color: 'bg-red-100 text-red-800' };
    if (riskScore >= 0.5) return { level: 'Medium', color: 'bg-yellow-100 text-yellow-800' };
    return { level: 'Low', color: 'bg-green-100 text-green-800' };
  };

  const getDocumentTypeIcon = (filename) => {
    const ext = filename.split('.').pop()?.toLowerCase();
    switch (ext) {
      case 'pdf':
        return <FileText className="h-6 w-6 text-red-600" />;
      case 'docx':
        return <FileText className="h-6 w-6 text-blue-600" />;
      case 'xlsx':
        return <FileText className="h-6 w-6 text-green-600" />;
      case 'txt':
        return <FileText className="h-6 w-6 text-gray-600" />;
      default:
        return <FileText className="h-6 w-6 text-gray-600" />;
    }
  };

  return (
    <div className="max-w-7xl mx-auto fade-in">
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-4">
          Document Analysis
        </h1>
        <p className="text-gray-600">
          Review AI-powered analysis results and insights from your legal documents
        </p>
      </div>

      <div className="grid lg:grid-cols-3 gap-8">
        {/* Document List */}
        <div className="lg:col-span-1">
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">
              Analyzed Documents ({documents.length})
            </h2>
            
            {loading ? (
              <div className="text-center py-8">
                <div className="spinner mx-auto mb-4"></div>
                <p className="text-gray-600">Loading documents...</p>
              </div>
            ) : documents.length === 0 ? (
              <div className="text-center py-8">
                <BarChart3 className="h-16 w-16 text-gray-400 mx-auto mb-4" />
                <p className="text-gray-600">No analyzed documents</p>
                <p className="text-sm text-gray-500">Upload and process documents to see analysis</p>
              </div>
            ) : (
              <div className="space-y-3">
                {documents.map((doc) => (
                  <div
                    key={doc.id}
                    onClick={() => analyzeDocument(doc.id)}
                    className={`p-4 border rounded-lg cursor-pointer transition-all duration-200 ${
                      selectedDocument?.id === doc.id
                        ? 'border-blue-500 bg-blue-50'
                        : 'border-gray-200 hover:border-blue-300 hover:bg-gray-50'
                    }`}
                  >
                    <div className="flex items-center space-x-3">
                      {getDocumentTypeIcon(doc.filename)}
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium text-gray-900 truncate">
                          {doc.filename}
                        </p>
                        <p className="text-xs text-gray-500">
                          {new Date(doc.upload_time).toLocaleDateString()}
                        </p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Analysis Results */}
        <div className="lg:col-span-2">
          {selectedDocument ? (
            <div className="space-y-6">
              {/* Document Header */}
              <div className="bg-white rounded-lg shadow-md p-6">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center space-x-3">
                    {getDocumentTypeIcon(selectedDocument.metadata?.filename)}
                    <div>
                      <h2 className="text-xl font-semibold text-gray-900">
                        {selectedDocument.metadata?.filename}
                      </h2>
                      <p className="text-sm text-gray-500">
                        Analyzed on {new Date(selectedDocument.metadata?.upload_time).toLocaleString()}
                      </p>
                    </div>
                  </div>
                  
                  <div className="flex space-x-2">
                    <button
                      onClick={() => exportAnalysis('json')}
                      className="px-3 py-2 text-sm bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors duration-200"
                    >
                      <Download className="h-4 w-4 mr-1 inline" />
                      JSON
                    </button>
                    <button
                      onClick={() => exportAnalysis('txt')}
                      className="px-3 py-2 text-sm bg-green-600 text-white rounded-md hover:bg-green-700 transition-colors duration-200"
                    >
                      <Download className="h-4 w-4 mr-1 inline" />
                      TXT
                    </button>
                  </div>
                </div>
              </div>

              {/* Analysis Summary */}
              {selectedDocument.summary && (
                <div className="bg-white rounded-lg shadow-md p-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-3 flex items-center">
                    <BarChart3 className="h-5 w-5 text-blue-600 mr-2" />
                    Analysis Summary
                  </h3>
                  <p className="text-gray-700 leading-relaxed">
                    {selectedDocument.summary}
                  </p>
                </div>
              )}

              {/* Key Findings */}
              {selectedDocument.key_findings && selectedDocument.key_findings.length > 0 && (
                <div className="bg-white rounded-lg shadow-md p-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-3 flex items-center">
                    <TrendingUp className="h-5 w-5 text-green-600 mr-2" />
                    Key Findings
                  </h3>
                  <div className="space-y-3">
                    {selectedDocument.key_findings.map((finding, index) => (
                      <div key={index} className="flex items-start space-x-3">
                        <div className="flex-shrink-0 w-6 h-6 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center text-sm font-medium">
                          {index + 1}
                        </div>
                        <p className="text-gray-700">{finding}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Risk Assessment */}
              {selectedDocument.risk_assessment && (
                <div className="bg-white rounded-lg shadow-md p-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-3 flex items-center">
                    <AlertTriangle className="h-5 w-5 text-yellow-600 mr-2" />
                    Risk Assessment
                  </h3>
                  <div className="bg-gray-50 rounded-lg p-4">
                    <p className="text-gray-700 mb-3">
                      {selectedDocument.risk_assessment}
                    </p>
                    {selectedDocument.risk_score && (
                      <div className="flex items-center space-x-2">
                        <span className="text-sm text-gray-600">Risk Level:</span>
                        <span className={`px-2 py-1 text-xs font-medium rounded-full ${getRiskLevel(selectedDocument.risk_score).color}`}>
                          {getRiskLevel(selectedDocument.risk_score).level}
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Recommendations */}
              {selectedDocument.recommendations && selectedDocument.recommendations.length > 0 && (
                <div className="bg-white rounded-lg shadow-md p-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-3 flex items-center">
                    <CheckCircle className="h-5 w-5 text-green-600 mr-2" />
                    Recommendations
                  </h3>
                  <div className="space-y-3">
                    {selectedDocument.recommendations.map((rec, index) => (
                      <div key={index} className="flex items-start space-x-3">
                        <div className="flex-shrink-0 w-6 h-6 bg-green-100 text-green-600 rounded-full flex items-center justify-center text-sm font-medium">
                          {index + 1}
                        </div>
                        <p className="text-gray-700">{rec}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Raw Results */}
              {selectedDocument.results && (
                <div className="bg-white rounded-lg shadow-md p-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-3">Raw Analysis Results</h3>
                  <div className="bg-gray-50 rounded-lg p-4">
                    <pre className="text-sm text-gray-700 whitespace-pre-wrap overflow-x-auto">
                      {JSON.stringify(selectedDocument.results, null, 2)}
                    </pre>
                  </div>
                </div>
              )}

              {/* Chat Interface */}
              <div className="bg-white rounded-lg shadow-md p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-3 flex items-center">
                  <MessageCircle className="h-5 w-5 text-purple-600 mr-2" />
                  Chat with AI
                </h3>
                
                {/* Suggested Questions */}
                <div className="mb-4">
                  <p className="text-sm text-gray-600 mb-2">💡 Try asking:</p>
                  <div className="flex flex-wrap gap-2">
                    {[
                      "Is this document safe to sign?",
                      "What are the key points?",
                      "Do I need a lawyer?",
                      "What type of document is this?",
                      "Explain this in simple terms"
                    ].map((question, index) => (
                      <button
                        key={index}
                        onClick={() => setChatMessage(question)}
                        className="px-3 py-1 text-xs bg-blue-100 text-blue-700 rounded-full hover:bg-blue-200 transition-colors duration-200"
                      >
                        {question}
                      </button>
                    ))}
                  </div>
                </div>
                
                <div className="space-y-4">
                  {chatHistory.map((msg) => (
                    <div
                      key={msg.id}
                      className={`flex ${msg.type === 'user' ? 'justify-end' : 'justify-start'}`}
                    >
                      <div
                        className={`p-3 rounded-lg max-w-[80%] ${
                          msg.type === 'user'
                            ? 'bg-blue-500 text-white'
                            : msg.type === 'ai'
                            ? 'bg-gray-100 text-gray-800'
                            : 'bg-red-100 text-red-800'
                        }`}
                      >
                        <p className="text-sm">{msg.content}</p>
                        <p className="text-xs text-gray-500 text-right mt-1">
                          {msg.timestamp}
                        </p>
                      </div>
                    </div>
                  ))}
                  {chatLoading && (
                    <div className="flex items-center space-x-2">
                      <div className="spinner mx-auto"></div>
                      <span className="text-sm text-gray-600">Thinking...</span>
                    </div>
                  )}
                </div>
                <form onSubmit={sendChatMessage} className="mt-4 flex space-x-2">
                  <input
                    type="text"
                    value={chatMessage}
                    onChange={(e) => setChatMessage(e.target.value)}
                    placeholder="Ask a question about this document..."
                    className="flex-1 p-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    disabled={chatLoading || !selectedDocument}
                  />
                  <button
                    type="submit"
                    className="p-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors duration-200"
                    disabled={chatLoading || !selectedDocument}
                  >
                    <Send className="h-5 w-5" />
                  </button>
                </form>
              </div>
            </div>
          ) : (
            <div className="bg-white rounded-lg shadow-md p-12 text-center">
              <BarChart3 className="h-16 w-16 text-gray-400 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">No Document Selected</h3>
              <p className="text-gray-600">
                Select a document from the list to view its analysis results
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default DocumentAnalysis;
