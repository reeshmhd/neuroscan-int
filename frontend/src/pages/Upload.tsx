import React, { useState, useEffect } from 'react'
import { Upload as UIcon, FileImage, CheckCircle2, WifiOff, Wifi, ArrowLeft, Brain, X } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { api } from '../services/api'

export default function Upload() {
  const nav = useNavigate()
  const [file, setFile] = useState<File | null>(null)
  const [status, setStatus] = useState<'checking' | 'online' | 'offline'>('checking')
  const [processing, setProcessing] = useState(false)
  const [progress, setProgress] = useState(0)
  const [isDragging, setIsDragging] = useState(false)

  useEffect(() => {
    api.healthCheck()
      .then(() => setStatus('online'))
      .catch(() => setStatus('offline'))
  }, [])

  const analyze = async () => {
    if (!file || status !== 'online') return
    setProcessing(true)
    setProgress(0)
    const interval = setInterval(() => setProgress((p) => (p >= 90 ? 90 : p + 15)), 200)
    try {
      const result = await api.predictImage(file)
      clearInterval(interval)
      setProgress(100)
      setTimeout(() => nav('/results', { state: { result, fileName: file.name } }), 700)
    } catch {
      clearInterval(interval)
      setProcessing(false)
      setProgress(0)
      alert('Analysis failed. Please try again.')
    }
  }
  const Icon = status === 'online' ? CheckCircle2 : status === 'offline' ? WifiOff : Wifi
  return (
    <div className="min-h-screen">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-md shadow-lg border-b border-gray-200/50">
        <div className="max-w-7xl mx-auto px-6 py-6 flex justify-between items-center">
          <button
            onClick={() => nav('/')}
            className="text-blue-700 flex items-center gap-1 hover:text-blue-900"
          >
            <ArrowLeft className="w-5 h-5" />
            Home
          </button>
          <div className="flex items-center space-x-4 group cursor-pointer" onClick={() => nav('/')}>
            <div className="p-3 gradient-medical rounded-xl shadow-lg group-hover:scale-110 group-hover:rotate-6 transition-all">
              <Brain className="h-8 w-8 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-gray-900 to-gray-700 bg-clip-text text-transparent">
                NeuroScan AI
              </h1>
              <p className="text-sm text-gray-600 font-medium">Advanced MRI-Only Analysis</p>
            </div>
          </div>
        </div>
      </header>
      {/* Status & Upload UI */}
      <main className="max-w-4xl mx-auto px-6 py-12">
        <div className="mb-8 text-center">
          <div className={`inline-flex items-center gap-2 px-4 py-2 rounded-full ${
            status === 'online' ? 'bg-green-50 border border-green-200' : 'bg-red-50 border border-red-200'
          }`}>
            <Icon className={`h-5 w-5 ${status === 'online' ? 'text-green-600' : 'text-red-600'}`} />
            <span className="font-medium">
              {status === 'online' ? 'AI Model Online' : status === 'offline' ? 'Model Offline' : 'Connecting...'}
            </span>
          </div>
        </div>
        <div
          onDrop={e => { e.preventDefault(); setIsDragging(false); setFile(e.dataTransfer.files[0]) }}
          onDragOver={e => { e.preventDefault(); setIsDragging(true) }}
          onDragLeave={() => setIsDragging(false)}
          className={`border-2 border-dashed rounded-2xl p-12 text-center transition-all ${
            isDragging ? 'border-blue-500 bg-blue-50 scale-105' : 'border-gray-300 hover:border-blue-400 hover:bg-blue-50/30'
          }`}
        >
          <input
            type="file"
            accept=".nii,.nii.gz,image/*"
            onChange={e => setFile(e.target.files![0])}
            hidden
            id="upload"
          />
          <label htmlFor="upload" className="cursor-pointer block">
            {file ? (
              <FileImage className="mx-auto h-16 w-16 text-blue-600 mb-4 animate-bounce" />
            ) : (
              <UIcon className="mx-auto h-16 w-16 text-gray-400 mb-4" />
            )}
            <p className="text-xl font-semibold text-gray-700 mb-2">
              {file ? file.name : 'Drop your MRI scan here'}
            </p>
            <p className="text-sm text-gray-500">or click to browse</p>
            <p className="text-xs text-gray-400 mt-2">Supports: .nii, .nii.gz, .jpg, .png (max 50MB)</p>
          </label>
        </div>
        {file && !processing && (
          <div className="mt-6 p-4 bg-green-50 border border-green-200 rounded-xl flex items-center justify-between">
            <div className="flex items-center gap-3">
              <CheckCircle2 className="h-5 w-5 text-green-600" />
              <div>
                <p className="font-medium text-green-900">File Ready</p>
                <p className="text-sm text-green-700">{file.name} ({(file.size / 1024 / 1024).toFixed(2)} MB)</p>
              </div>
            </div>
            <button onClick={() => setFile(null)} className="p-2 hover:bg-red-100 rounded-full transition-all">
              <X className="h-5 w-5 text-red-600" />
            </button>
          </div>
        )}
        {processing && (
          <div className="mt-6">
            <div className="flex justify-between text-sm mb-2">
              <span className="text-gray-600">🧠 Analyzing MRI scan...</span>
              <span className="font-bold text-blue-600">{progress}%</span>
            </div>
            <div className="h-3 bg-gray-200 rounded-full overflow-hidden">
              <div
                className="h-full gradient-medical transition-all duration-300"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>
        )}
        <div className="text-center mt-10">
          <button
            onClick={analyze}
            disabled={!file || status !== 'online' || processing}
            className="btn-primary inline-flex items-center gap-3"
          >
            {processing ? '🧠 Analyzing...' : '🔍 Analyze with AI'}
          </button>
        </div>
      </main>
    </div>
  )
}
