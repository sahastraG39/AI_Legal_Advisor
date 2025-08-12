import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, FileText, X, CheckCircle, AlertCircle, Loader } from 'lucide-react';
import toast from 'react-hot-toast';
import axios from 'axios';

const DocumentUpload = () => {
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [processing, setProcessing] = useState(false);
  const [analysisType, setAnalysisType] = useState('general');
  const [enableOCR, setEnableOCR] = useState(true);
  const [storeInVectorDB, setStoreInVectorDB] = useState(true);

  const onDrop = useCallback((acceptedFiles) => {
    const newFiles = acceptedFiles.map(file => ({
      file,
      id: Math.random().toString(36).substr(2, 9),
      status: 'pending',
      progress: 0,
      error: null,
      results: null
    }));
    
    setUploadedFiles(prev => [...prev, ...newFiles]);
    toast.success(`${acceptedFiles.length} file(s) added to queue`);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'text/plain': ['.txt'],
      'image/png': ['.png'],
      'image/jpeg': ['.jpg', '.jpeg']
    },
    multiple: true
  });

  const uploadFile = async (fileData) => {
    const formData = new FormData();
    formData.append('file', fileData.file);
    formData.append('analysis_type', analysisType);
    formData.append('enable_ocr', enableOCR);
    formData.append('store_in_vector_db', storeInVectorDB);

    try {
      setUploadedFiles(prev => 
        prev.map(f => 
          f.id === fileData.id 
            ? { ...f, status: 'processing', progress: 10 }
            : f
        )
      );

      const response = await axios.post('/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setUploadedFiles(prev => 
            prev.map(f => 
              f.id === fileData.id 
                ? { ...f, progress: Math.min(90, progress) }
                : f
            )
          );
        }
      });

      setUploadedFiles(prev => 
        prev.map(f => 
          f.id === fileData.id 
            ? { 
                ...f, 
                status: 'completed', 
                progress: 100, 
                results: response.data.results 
              }
            : f
        )
      );

      toast.success(`${fileData.file.name} processed successfully!`);
      
    } catch (error) {
      console.error('Upload error:', error);
      setUploadedFiles(prev => 
        prev.map(f => 
          f.id === fileData.id 
            ? { 
                ...f, 
                status: 'error', 
                error: error.response?.data?.detail || 'Upload failed' 
              }
            : f
        )
      );
      toast.error(`Failed to process ${fileData.file.name}`);
    }
  };

  const processAllFiles = async () => {
    if (uploadedFiles.length === 0) {
      toast.error('No files to process');
      return;
    }

    setProcessing(true);
    const pendingFiles = uploadedFiles.filter(f => f.status === 'pending');
    
    for (const fileData of pendingFiles) {
      await uploadFile(fileData);
    }
    
    setProcessing(false);
    toast.success('All files processed!');
  };

  const removeFile = (fileId) => {
    setUploadedFiles(prev => prev.filter(f => f.id !== fileId));
  };

  const clearCompleted = () => {
    setUploadedFiles(prev => prev.filter(f => f.status !== 'completed'));
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'pending':
        return <FileText className="h-5 w-5 text-gray-400" />;
      case 'processing':
        return <Loader className="h-5 w-5 text-blue-500 animate-spin" />;
      case 'completed':
        return <CheckCircle className="h-5 w-5 text-green-500" />;
      case 'error':
        return <AlertCircle className="h-5 w-5 text-red-500" />;
      default:
        return <FileText className="h-5 w-5 text-gray-400" />;
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'pending':
        return 'bg-gray-100 text-gray-800';
      case 'processing':
        return 'bg-blue-100 text-blue-800';
      case 'completed':
        return 'bg-green-100 text-green-800';
      case 'error':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className="max-w-4xl mx-auto fade-in">
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-4">
          Upload Legal Documents
        </h1>
        <p className="text-gray-600">
          Upload your legal documents for AI-powered analysis and insights
        </p>
      </div>

      {/* Upload Configuration */}
      <div className="bg-white rounded-lg shadow-md p-6 mb-8">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">Analysis Configuration</h2>
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Analysis Type
            </label>
            <select
              value={analysisType}
              onChange={(e) => setAnalysisType(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="general">General Analysis</option>
              <option value="contract">Contract Review</option>
              <option value="risk">Risk Assessment</option>
              <option value="compliance">Compliance Check</option>
              <option value="summary">Document Summary</option>
            </select>
          </div>
          
          <div className="space-y-4">
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={enableOCR}
                onChange={(e) => setEnableOCR(e.target.checked)}
                className="mr-2 h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
              />
              <span className="text-sm text-gray-700">Enable OCR for scanned documents</span>
            </label>
            
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={storeInVectorDB}
                onChange={(e) => setStoreInVectorDB(e.target.checked)}
                className="mr-2 h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
              />
              <span className="text-sm text-gray-700">Store in vector database for search</span>
            </label>
          </div>
        </div>
      </div>

      {/* Drop Zone */}
      <div className="bg-white rounded-lg shadow-md p-8 mb-8">
        <div
          {...getRootProps()}
          className={`upload-area p-12 text-center cursor-pointer transition-all duration-200 ${
            isDragActive ? 'dragover' : ''
          }`}
        >
          <input {...getInputProps()} />
          <Upload className="h-16 w-16 text-gray-400 mx-auto mb-4" />
          <p className="text-xl text-gray-600 mb-2">
            {isDragActive ? 'Drop files here' : 'Drag & drop files here, or click to select'}
          </p>
          <p className="text-sm text-gray-500">
            Supports PDF, DOCX, XLSX, TXT, PNG, JPG files
          </p>
        </div>
      </div>

      {/* File List */}
      {uploadedFiles.length > 0 && (
        <div className="bg-white rounded-lg shadow-md p-6 mb-8">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-xl font-semibold text-gray-900">
              Uploaded Files ({uploadedFiles.length})
            </h2>
            <div className="space-x-2">
              <button
                onClick={clearCompleted}
                className="px-4 py-2 text-sm text-gray-600 hover:text-gray-800 border border-gray-300 rounded-md hover:bg-gray-50"
              >
                Clear Completed
              </button>
              <button
                onClick={processAllFiles}
                disabled={processing || uploadedFiles.every(f => f.status !== 'pending')}
                className="px-4 py-2 bg-blue-600 text-white text-sm font-medium rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {processing ? 'Processing...' : 'Process All Files'}
              </button>
            </div>
          </div>

          <div className="space-y-4">
            {uploadedFiles.map((fileData) => (
              <div key={fileData.id} className="border border-gray-200 rounded-lg p-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3 flex-1">
                    {getStatusIcon(fileData.status)}
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-gray-900 truncate">
                        {fileData.file.name}
                      </p>
                      <p className="text-sm text-gray-500">
                        {(fileData.file.size / 1024 / 1024).toFixed(2)} MB
                      </p>
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-3">
                    <span className={`px-2 py-1 text-xs font-medium rounded-full ${getStatusColor(fileData.status)}`}>
                      {fileData.status}
                    </span>
                    
                    {fileData.status === 'processing' && (
                      <div className="w-20 bg-gray-200 rounded-full h-2">
                        <div 
                          className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                          style={{ width: `${fileData.progress}%` }}
                        ></div>
                      </div>
                    )}
                    
                    {fileData.status === 'pending' && (
                      <button
                        onClick={() => uploadFile(fileData)}
                        className="px-3 py-1 text-sm bg-blue-600 text-white rounded-md hover:bg-blue-700"
                      >
                        Process
                      </button>
                    )}
                    
                    <button
                      onClick={() => removeFile(fileData.id)}
                      className="text-gray-400 hover:text-red-500"
                    >
                      <X className="h-5 w-5" />
                    </button>
                  </div>
                </div>
                
                {fileData.error && (
                  <div className="mt-3 p-3 bg-red-50 border border-red-200 rounded-md">
                    <p className="text-sm text-red-800">{fileData.error}</p>
                  </div>
                )}
                
                {fileData.results && (
                  <div className="mt-3 p-3 bg-green-50 border border-green-200 rounded-md">
                    <p className="text-sm text-green-800 font-medium">Analysis Complete!</p>
                    <p className="text-sm text-green-700">
                      Document processed successfully. View results in the Analysis section.
                    </p>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Processing Status */}
      {processing && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 text-center">
          <Loader className="h-6 w-6 text-blue-600 mx-auto mb-2 animate-spin" />
          <p className="text-blue-800">Processing documents... Please wait.</p>
        </div>
      )}
    </div>
  );
};

export default DocumentUpload;
