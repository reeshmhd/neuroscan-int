const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000/api'

export interface PredictionResult {
  success: boolean
  predicted_class: string
  predicted_class_index: number
  confidence: number
  confidence_percentage: number
  description: string
  probabilities: Record<string,{probability:number;percentage:number}>
  severity_level: 'low'|'medium'|'high'|'critical'
  color: string
  model_info?: {architecture:string;parameters:string;device:string}
  error?: string
}

class AlzheimerAPI {
  constructor(private baseURL = API_BASE_URL){}
  async healthCheck(){
    const r = await fetch(`${this.baseURL}/health`)
    if(!r.ok) throw new Error()
    return r.json()
  }
  async predictImage(file:File){
    const fd = new FormData()
    fd.append('image',file,file.name)
    const r = await fetch(`${this.baseURL}/predict`,{method:'POST',body:fd})
    const data = await r.json()
    if(!r.ok) throw new Error(data.error)
    return data
  }
}

export const api = new AlzheimerAPI()
