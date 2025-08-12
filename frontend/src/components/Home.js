import React from 'react';
import { Link } from 'react-router-dom';
import { FileText, Upload, Search, BarChart3, Shield, Zap, Users } from 'lucide-react';

const Home = () => {
  const features = [
    {
      icon: FileText,
      title: 'Document Processing',
      description: 'Upload and analyze legal documents in multiple formats including PDF, DOCX, and scanned images.',
      color: 'text-blue-600',
      bgColor: 'bg-blue-50'
    },
    {
      icon: Shield,
      title: 'AI-Powered Analysis',
      description: 'Get intelligent insights, risk assessments, and clause identification using advanced AI models.',
      color: 'text-green-600',
      bgColor: 'bg-green-50'
    },
    {
      icon: Search,
      title: 'Smart Search',
      description: 'Search through your document library with semantic understanding and context-aware results.',
      color: 'text-purple-600',
      bgColor: 'bg-purple-50'
    },
    {
      icon: BarChart3,
      title: 'Comprehensive Reports',
      description: 'Generate detailed analysis reports with actionable insights and recommendations.',
      color: 'text-orange-600',
      bgColor: 'bg-orange-50'
    }
  ];

  const quickActions = [
    {
      title: 'Upload Document',
      description: 'Start analyzing a new legal document',
      path: '/upload',
      icon: Upload,
      color: 'bg-blue-600 hover:bg-blue-700'
    },
    {
      title: 'Search Documents',
      description: 'Find specific information across your documents',
      path: '/search',
      icon: Search,
      color: 'bg-green-600 hover:bg-green-700'
    },
    {
      title: 'View Analysis',
      description: 'Review analysis results and reports',
      path: '/analysis',
      icon: BarChart3,
      color: 'bg-purple-600 hover:bg-purple-700'
    }
  ];

  return (
    <div className="fade-in">
      {/* Hero Section */}
      <div className="text-center mb-16">
        <h1 className="text-4xl md:text-6xl font-bold text-gray-900 mb-6">
          AI Legal Document
          <span className="text-blue-600 block">Explainer</span>
        </h1>
        <p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto">
          Transform complex legal documents into clear, actionable insights using advanced AI technology. 
          Upload, analyze, and understand legal contracts, agreements, and documents in minutes.
        </p>
        <div className="flex flex-col sm:flex-row gap-4 justify-center">
          <Link
            to="/upload"
            className="inline-flex items-center px-8 py-4 bg-blue-600 text-white font-semibold rounded-lg hover:bg-blue-700 transition-colors duration-200 shadow-lg hover:shadow-xl"
          >
            <Upload className="h-5 w-5 mr-2" />
            Get Started
          </Link>
          <Link
            to="/search"
            className="inline-flex items-center px-8 py-4 bg-white text-blue-600 font-semibold rounded-lg border-2 border-blue-600 hover:bg-blue-50 transition-colors duration-200"
          >
            <Search className="h-5 w-5 mr-2" />
            Explore Documents
          </Link>
        </div>
      </div>

      {/* Features Section */}
      <div className="mb-16">
        <h2 className="text-3xl font-bold text-gray-900 text-center mb-12">
          Powerful Features for Legal Professionals
        </h2>
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
          {features.map((feature, index) => {
            const Icon = feature.icon;
            return (
              <div key={index} className="text-center p-6 rounded-lg bg-white shadow-md hover:shadow-lg transition-shadow duration-200">
                <div className={`inline-flex items-center justify-center w-16 h-16 rounded-full ${feature.bgColor} mb-4`}>
                  <Icon className={`h-8 w-8 ${feature.color}`} />
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mb-2">{feature.title}</h3>
                <p className="text-gray-600">{feature.description}</p>
              </div>
            );
          })}
        </div>
      </div>

      {/* Quick Actions */}
      <div className="mb-16">
        <h2 className="text-3xl font-bold text-gray-900 text-center mb-12">
          Quick Actions
        </h2>
        <div className="grid md:grid-cols-3 gap-8">
          {quickActions.map((action, index) => {
            const Icon = action.icon;
            return (
              <Link
                key={index}
                to={action.path}
                className={`${action.color} text-white p-8 rounded-lg shadow-md hover:shadow-lg transition-all duration-200 transform hover:-translate-y-1`}
              >
                <div className="text-center">
                  <Icon className="h-12 w-12 mx-auto mb-4" />
                  <h3 className="text-xl font-semibold mb-2">{action.title}</h3>
                  <p className="text-blue-100">{action.description}</p>
                </div>
              </Link>
            );
          })}
        </div>
      </div>

      {/* Stats Section */}
      <div className="bg-white rounded-lg shadow-md p-8 mb-16">
        <h2 className="text-3xl font-bold text-gray-900 text-center mb-8">
          System Status
        </h2>
        <div className="grid md:grid-cols-3 gap-8">
          <div className="text-center">
            <div className="text-3xl font-bold text-blue-600 mb-2">Ready</div>
            <div className="text-gray-600">AI Models Loaded</div>
          </div>
          <div className="text-center">
            <div className="text-3xl font-bold text-green-600 mb-2">Active</div>
            <div className="text-gray-600">Processing Engine</div>
          </div>
          <div className="text-center">
            <div className="text-3xl font-bold text-purple-600 mb-2">Secure</div>
            <div className="text-gray-600">Data Protection</div>
          </div>
        </div>
      </div>

      {/* CTA Section */}
      <div className="text-center bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg p-12 text-white">
        <h2 className="text-3xl font-bold mb-4">
          Ready to Transform Your Legal Document Analysis?
        </h2>
        <p className="text-xl mb-8 text-blue-100">
          Join thousands of legal professionals who are already using AI to save time and improve accuracy.
        </p>
        <Link
          to="/upload"
          className="inline-flex items-center px-8 py-4 bg-white text-blue-600 font-semibold rounded-lg hover:bg-gray-100 transition-colors duration-200 shadow-lg"
        >
          <Zap className="h-5 w-5 mr-2" />
          Start Analyzing Now
        </Link>
      </div>
    </div>
  );
};

export default Home;
