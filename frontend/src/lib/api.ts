import axios from 'axios'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || '',
  timeout: 60000, // 60s for model inference
})

export interface PredictionResponse {
  diagnosis: string
  confidence: number
  confidenceLevel: 'high' | 'moderate' | 'low'
  allPredictions: Record<string, number>
}

export interface VisualizationResponse {
  image: string // base64 data URL
}

export interface ApiError {
  error: string
}

export async function predictImage(file: File): Promise<PredictionResponse> {
  const formData = new FormData()
  formData.append('image', file)
  const response = await api.post<PredictionResponse>('/api/predict', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  })
  return response.data
}

export async function getGradCAM(file: File): Promise<VisualizationResponse> {
  const formData = new FormData()
  formData.append('image', file)
  const response = await api.post<VisualizationResponse>('/api/gradcam', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  })
  return response.data
}

export async function getSaliency(file: File): Promise<VisualizationResponse> {
  const formData = new FormData()
  formData.append('image', file)
  const response = await api.post<VisualizationResponse>('/api/saliency', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  })
  return response.data
}

export async function healthCheck(): Promise<{ status: string; model_loaded: boolean }> {
  const response = await api.get<{ status: string; model_loaded: boolean }>('/api/health')
  return response.data
}
