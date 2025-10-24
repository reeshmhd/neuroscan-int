import React from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { PredictionResult } from '../services/api'
import { Brain, CheckCircle2, AlertTriangle, Info, ArrowLeft } from 'lucide-react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts'

const detailedGuidance: Record<string, string> = {
  'Normal': (
    "Your MRI scan does not show evidence of dementia." +
    " To maintain cognitive health, continue a balanced diet, regular exercise, mental stimulation, " +
    "adequate sleep, and routine health check-ups. If you notice any symptoms, seek medical advice promptly."
  ),
  'Mild': (
    "Mild dementia detected. You may experience minor memory lapses or changes in thinking ability." +
    " Consider scheduling regular check-ups, share your results with a healthcare provider, and discuss " +
    "symptom management as early intervention can be most effective."
  ),
  'Moderate': (
    "Moderate dementia features apparent memory loss and daily task difficulties. Please consult with a neurologist or specialist soon." +
    " Family support, safety planning, and professional guidance will help improve quality of life."
  ),
  'Severe Dementia': (
    "Severe dementia is present; the person may require full-time support. Immediate medical consultation is advised for care planning." +
    " Caregiver resources and specialist care can help ease management of advanced symptoms."
  ),
  'VeryMild': (
    "Very mild cognitive impairment identified—early monitoring is recommended." +
    " Consider a detailed assessment by a neurologist, maintain healthy habits, " +
    "and involve your loved ones in regular wellness planning. Early action can delay progression."
  ),
}

export default function Results() {
  const nav = useNavigate()
  const { result } = useLocation().state as { result: PredictionResult }

  const RiskIcon = result.confidence_percentage > 80
    ? CheckCircle2
    : result.confidence_percentage > 50
      ? AlertTriangle
      : Info

  // Prepare data for vertical bar chart with class names on x-axis
  const data = Object.entries(result.probabilities).map(([name, d]) => ({
    name: name.replace(' Dementia', '').replace('Very Mild', 'V.Mild'),
    value: parseFloat(d.percentage.toFixed(2)),
    color: name === result.predicted_class ? result.color : '#e2e8f0',
    active: name === result.predicted_class,
  }))

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100 py-10 max-w-6xl mx-auto px-6">
      <header className="flex justify-between mb-5 items-center">
        <button
          onClick={() => nav('/upload')}
          className="text-blue-700 flex items-center hover:text-blue-900"
        >
          <ArrowLeft className="w-6 h-6 mr-1" />
          Analyze Another
        </button>
        <div className="flex items-center space-x-2">
          <Brain className="h-7 w-7 text-blue-600 animate-pulse" />
          <span className="font-bold text-blue-700 text-lg">NeuroScan AI</span>
        </div>
      </header>

      <div className="grid md:grid-cols-2 gap-8">
        {/* LEFT: Result/Guidance */}
        <div className="card-hero p-10 flex flex-col justify-center">
          <div className="flex items-center gap-6 mb-6">
            <div className="gradient-medical rounded-full p-6 shadow-lg">
              <Brain className="h-16 w-16 text-white animate-pulse-slow" />
            </div>
            <div>
              <div className="inline-flex items-center gap-2 mb-2">
                <RiskIcon className="h-7 w-7 text-green-600" />
                <h2 className="text-3xl font-bold">{result.predicted_class}</h2>
              </div>
              <div className="text-xl text-gray-900 font-medium mb-1">
                Confidence: <span className="font-bold text-blue-700">{result.confidence_percentage.toFixed(2)}%</span>
              </div>
            </div>
          </div>
          <div className="mb-4 text-base text-gray-700 leading-relaxed">
            {detailedGuidance[result.predicted_class] ??
              "Detailed analysis complete. Please consult a healthcare professional for next steps."
            }
          </div>
          <div className="text-sm text-gray-500 mt-auto">
            <strong>Note:</strong> This tool is for research purposes only. Please consult a medical professional before making clinical decisions.
          </div>
        </div>
        {/* RIGHT: Bar Chart */}
        <div className="card-medical p-6">
          <h3 className="text-xl font-bold mb-4 text-gray-900">Probability Distribution</h3>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart
              data={data}
              margin={{ left: 0, right: 10, top: 15, bottom: 15 }}
            >
              <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
              <XAxis
                dataKey="name"
                tick={{ fill: '#64748b', fontWeight: 600 }}
                fontSize={16}
                axisLine
                label={{ value: "Classes", position: "insideBottom", offset: -10 }}
              />
              <YAxis
                domain={[0, 100]}
                tick={{ fill: '#64748b', fontWeight: 600 }}
                tickFormatter={v => `${v}%`}
                fontSize={16}
                axisLine
                label={{ value: "Confidence (%)", angle: -90, position: "insideLeft" }}
              />
              <Tooltip
                cursor={{ fill: '#f0f9ff' }}
                contentStyle={{ backgroundColor: 'white', borderRadius: '0.75rem', fontWeight: 600 }}
                formatter={val => [`${val}%`, "Confidence"]}
              />
              <Bar dataKey="value" radius={0}>
                {data.map((entry, idx) => (
                  <Cell key={idx} fill={entry.active ? result.color : '#d1d5db'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  )
}
