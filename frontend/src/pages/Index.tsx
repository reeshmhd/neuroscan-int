import React from 'react'
import { useNavigate } from 'react-router-dom'
import { Brain, Upload, Shield, Zap, CheckCircle } from 'lucide-react'

export default function Index() {
  const nav = useNavigate()
  return (
    <div className="min-h-screen">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-md shadow-lg border-b border-gray-200/50 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-6 flex flex-wrap items-center justify-between">
          <div className="flex items-center space-x-4 group">
            <div className="p-3 gradient-medical rounded-xl shadow-lg">
              <Brain className="h-8 w-8 text-white animate-pulse-slow" />
            </div>
            <div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-gray-900 to-gray-700 bg-clip-text text-transparent">
                NeuroScan AI
              </h1>
              <p className="text-sm text-gray-600 font-medium">Advanced MRI-Only Analysis</p>
            </div>
          </div>
          <div className="flex items-center space-x-2 px-4 py-2 bg-amber-50 border border-amber-200 rounded-full hover:scale-105 transition-all">
            <Shield className="h-4 w-4 text-amber-600" />
            <span className="text-sm font-medium text-amber-800">Research Tool</span>
          </div>
        </div>
      </header>
      {/* Hero Section */}
      <main className="max-w-7xl mx-auto px-6 py-16">
        <div className="text-center mb-16 animate-fade-in px-4 md:px-0">
          <h2 className="text-5xl font-bold mb-6 bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent leading-tight max-w-[700px] mx-auto">
            Early Detection of Alzheimer&apos;s Disease
          </h2>
          <p className="text-xl text-gray-700 mb-8 max-w-3xl mx-auto leading-relaxed">
            Leverage cutting-edge AI technology to analyze brain MRI scans with 93% accuracy.
            Our model uses Transfer Learning and ResNet50 to provides rapid, reliable dementia assessment.
          </p>
          <button
            onClick={() => nav('/upload')}
            className="btn-primary inline-flex items-center gap-3 text-lg"
          >
            <Upload className="h-6 w-6" />
            Upload MRI Scan
            <span className="animate-pulse">→</span>
          </button>
        </div>
        {/* Features Grid */}
        <div className="grid md:grid-cols-3 gap-8 mb-16">
          {[
            { icon: Brain, title: 'AI-Powered Analysis', desc: 'ResNet50 with 23.5M parameters, pretrained on ImageNet' },
            { icon: Zap, title: 'Rapid Results', desc: 'Get comprehensive analysis results in under 60 seconds' },
            { icon: Shield, title: 'High Accuracy', desc: '93% accuracy across all dementia severity levels' },
          ].map((feature, i) => (
            <div key={i} className="card-medical group animate-slide-up">
              <div className="p-4 bg-gradient-to-br from-blue-500 to-blue-600 rounded-xl w-fit mb-4 group-hover:scale-110 group-hover:rotate-6">
                <feature.icon className="h-8 w-8 text-white" />
              </div>
              <h3 className="text-xl font-bold mb-2 text-gray-900">{feature.title}</h3>
              <p className="text-gray-600 leading-relaxed">{feature.desc}</p>
            </div>
          ))}
        </div>
      </main>
    </div>
  )
}
