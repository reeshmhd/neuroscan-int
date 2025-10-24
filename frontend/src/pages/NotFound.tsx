import React from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { AlertTriangle } from 'lucide-react'

export default function NotFound() {
  const nav = useNavigate()
  const { pathname } = useLocation()

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gradient-subtle p-6">
      <AlertTriangle className="h-16 w-16 text-red-600 mb-4 animate-pulse-slow" />
      <h1 className="text-5xl font-bold text-gray-900 mb-2">404</h1>
      <p className="text-gray-700 mb-6">
        Oops! The page <code className="bg-gray-100 px-1 rounded">{pathname}</code> was not found.
      </p>
      <button
        onClick={() => nav('/')}
        className="btn-primary inline-flex items-center gap-2"
      >
        Go Home
      </button>
    </div>
  )
}
