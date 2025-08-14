import React, { useState, useEffect } from 'react';
import { Search, FileText, Calendar, User, ArrowRight } from 'lucide-react';
import axios from 'axios';
import toast from 'react-hot-toast';

const DocumentSearch = () => {
  const [query, setQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [searching, setSearching] = useState(false);
  const [topK, setTopK] = useState(5);
  const [threshold, setThreshold] = useState(0.5);
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchDocuments();
  }, []);

  const fetchDocuments = async () => {
    try {
      const response = await axios.get('/api/documents');
      setDocuments(response.data.processing_status || {});
      setLoading(false);
    } catch (error) {
      console.error('Failed to fetch documents:', error);
      toast.error('Failed to load documents');
      setLoading(false);
    }
  };

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!query.trim()) {
      toast.error('Please enter a search query');
      return;
    }

    setSearching(true);
    try {
      const response = await axios.post('/api/search', {
        query: query.trim(),
        top_k: topK,
        threshold: threshold
      });
      
      setSearchResults(response.data.results || []);
      toast.success(`Found ${response.data.total_found} results`);
    } catch (error) {
      console.error('Search failed:', error);
      toast.error('Search failed. Please try again.');
      setSearchResults([]);
    } finally {
      setSearching(false);
    }
  };

  const formatDate = (dateString) => {
    if (!dateString) return 'N/A';
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const getDocumentType = (filename) => {
    const ext = filename.split('.').pop()?.toLowerCase();
    switch (ext) {
      case 'pdf':
        return { type: 'PDF', color: 'bg-red-100 text-red-800' };
      case 'docx':
        return { type: 'Word', color: 'bg-blue-100 text-blue-800' };
      case 'xlsx':
        return { type: 'Excel', color: 'bg-green-100 text-green-800' };
      case 'txt':
        return { type: 'Text', color: 'bg-gray-100 text-gray-800' };
      case 'png':
      case 'jpg':
      case 'jpeg':
        return { type: 'Image', color: 'bg-purple-100 text-purple-800' };
      default:
        return { type: 'Unknown', color: 'bg-gray-100 text-gray-800' };
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'completed':
        return 'bg-green-100 text-green-800';
      case 'processing':
        return 'bg-blue-100 text-blue-800';
      case 'error':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className="max-w-6xl mx-auto fade-in">
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-4">
          Search Documents
        </h1>
        <p className="text-gray-600">
          Search through your processed legal documents using semantic search
        </p>
      </div>

      {/* Search Form */}
      <div className="bg-white rounded-lg shadow-md p-6 mb-8">
        <form onSubmit={handleSearch} className="space-y-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Search Query
            </label>
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Enter your search query (e.g., 'contract terms', 'liability clause', 'payment schedule')"
                className="w-full pl-10 pr-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
          </div>

          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Number of Results
              </label>
              <select
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value={3}>3 results</option>
                <option value={5}>5 results</option>
                <option value={10}>10 results</option>
                <option value={20}>20 results</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Similarity Threshold
              </label>
              <select
                value={threshold}
                onChange={(e) => setThreshold(Number(e.target.value))}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value={0.3}>Low (0.3)</option>
                <option value={0.5}>Medium (0.5)</option>
                <option value={0.7}>High (0.7)</option>
                <option value={0.9}>Very High (0.9)</option>
              </select>
            </div>
          </div>

          <button
            type="submit"
            disabled={searching || !query.trim()}
            className="w-full md:w-auto px-8 py-3 bg-blue-600 text-white font-semibold rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors duration-200"
          >
            {searching ? (
              <div className="flex items-center justify-center">
                <div className="spinner mr-2"></div>
                Searching...
              </div>
            ) : (
              <div className="flex items-center justify-center">
                <Search className="h-5 w-5 mr-2" />
                Search Documents
              </div>
            )}
          </button>
        </form>
      </div>

      {/* Search Results */}
      {searchResults.length > 0 && (
        <div className="bg-white rounded-lg shadow-md p-6 mb-8">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">
            Search Results ({searchResults.length})
          </h2>
          <div className="space-y-4">
            {searchResults.map((result, index) => (
              <div key={index} className="border border-gray-200 rounded-lg p-4 hover:bg-gray-50 transition-colors duration-200">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-3 mb-2">
                      <FileText className="h-5 w-5 text-blue-600" />
                      <h3 className="text-lg font-medium text-gray-900">
                        {result.metadata?.filename || `Document ${index + 1}`}
                      </h3>
                      <span className={`px-2 py-1 text-xs font-medium rounded-full ${getDocumentType(result.metadata?.filename || '').color}`}>
                        {getDocumentType(result.metadata?.filename || '').type}
                      </span>
                    </div>
                    
                    {result.metadata?.similarity && (
                      <div className="mb-2">
                        <span className="text-sm text-gray-600">Similarity Score: </span>
                        <span className="text-sm font-medium text-blue-600">
                          {(result.metadata.similarity * 100).toFixed(1)}%
                        </span>
                      </div>
                    )}
                    
                    <p className="text-gray-700 mb-3">
                      {result.text || result.content || 'No content available'}
                    </p>
                    
                    <div className="flex items-center space-x-4 text-sm text-gray-500">
                      <div className="flex items-center space-x-1">
                        <Calendar className="h-4 w-4" />
                        <span>{formatDate(result.metadata?.upload_time)}</span>
                      </div>
                      {result.metadata?.file_size && (
                        <div className="flex items-center space-x-1">
                          <FileText className="h-4 w-4" />
                          <span>{(result.metadata.file_size / 1024 / 1024).toFixed(2)} MB</span>
                        </div>
                      )}
                    </div>
                  </div>
                  
                  <button className="ml-4 p-2 text-gray-400 hover:text-blue-600 transition-colors duration-200">
                    <ArrowRight className="h-5 w-5" />
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Document Library */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">
          Document Library ({Object.keys(documents).length})
        </h2>
        
        {loading ? (
          <div className="text-center py-8">
            <div className="spinner mx-auto mb-4"></div>
            <p className="text-gray-600">Loading documents...</p>
          </div>
        ) : Object.keys(documents).length === 0 ? (
          <div className="text-center py-8">
            <FileText className="h-16 w-16 text-gray-400 mx-auto mb-4" />
            <p className="text-gray-600">No documents found</p>
            <p className="text-sm text-gray-500">Upload some documents to get started</p>
          </div>
        ) : (
          <div className="space-y-4">
            {Object.entries(documents).map(([fileId, doc]) => (
              <div key={fileId} className="border border-gray-200 rounded-lg p-4 hover:bg-gray-50 transition-colors duration-200">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3 flex-1">
                    <FileText className="h-5 w-5 text-blue-600" />
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-gray-900 truncate">
                        {doc.filename}
                      </p>
                      <div className="flex items-center space-x-4 text-sm text-gray-500">
                        <span>{formatDate(doc.upload_time)}</span>
                        <span className={`px-2 py-1 text-xs font-medium rounded-full ${getStatusColor(doc.status)}`}>
                          {doc.status}
                        </span>
                        <span className={`px-2 py-1 text-xs font-medium rounded-full ${getDocumentType(doc.filename).color}`}>
                          {getDocumentType(doc.filename).type}
                        </span>
                      </div>
                    </div>
                  </div>
                  
                  {doc.status === 'completed' && (
                    <button className="px-3 py-1 text-sm bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors duration-200">
                      View Analysis
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default DocumentSearch;
