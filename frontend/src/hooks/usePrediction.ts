import { useState, useCallback } from 'react'
import { predictImage, getGradCAM, getSaliency, type PredictionResponse } from '@/lib/api'
import axios from 'axios'

interface UsePredictionState {
  prediction: PredictionResponse | null
  gradcamImage: string | null
  saliencyImage: string | null
  isAnalyzing: boolean
  isLoadingGradcam: boolean
  isLoadingSaliency: boolean
  error: string | null
}

interface UsePredictionReturn extends UsePredictionState {
  analyze: (file: File) => Promise<void>
  loadGradCAM: (file: File) => Promise<void>
  loadSaliency: (file: File) => Promise<void>
  reset: () => void
}

const initialState: UsePredictionState = {
  prediction: null,
  gradcamImage: null,
  saliencyImage: null,
  isAnalyzing: false,
  isLoadingGradcam: false,
  isLoadingSaliency: false,
  error: null,
}

export function usePrediction(): UsePredictionReturn {
  const [state, setState] = useState<UsePredictionState>(initialState)

  const analyze = useCallback(async (file: File) => {
    setState((prev) => ({
      ...prev,
      isAnalyzing: true,
      error: null,
      prediction: null,
      gradcamImage: null,
      saliencyImage: null,
    }))

    try {
      const result = await predictImage(file)
      setState((prev) => ({
        ...prev,
        isAnalyzing: false,
        prediction: result,
      }))
    } catch (err) {
      let errorMessage = 'Analysis failed. Please try again.'
      if (axios.isAxiosError(err) && err.response?.data?.error) {
        errorMessage = err.response.data.error
      }
      setState((prev) => ({
        ...prev,
        isAnalyzing: false,
        error: errorMessage,
      }))
    }
  }, [])

  const loadGradCAM = useCallback(async (file: File) => {
    setState((prev) => ({
      ...prev,
      isLoadingGradcam: true,
      error: null,
    }))

    try {
      const result = await getGradCAM(file)
      setState((prev) => ({
        ...prev,
        isLoadingGradcam: false,
        gradcamImage: result.image,
      }))
    } catch (err) {
      let errorMessage = 'Failed to generate Grad-CAM visualization.'
      if (axios.isAxiosError(err) && err.response?.data?.error) {
        errorMessage = err.response.data.error
      }
      setState((prev) => ({
        ...prev,
        isLoadingGradcam: false,
        error: errorMessage,
      }))
    }
  }, [])

  const loadSaliency = useCallback(async (file: File) => {
    setState((prev) => ({
      ...prev,
      isLoadingSaliency: true,
      error: null,
    }))

    try {
      const result = await getSaliency(file)
      setState((prev) => ({
        ...prev,
        isLoadingSaliency: false,
        saliencyImage: result.image,
      }))
    } catch (err) {
      let errorMessage = 'Failed to generate saliency map.'
      if (axios.isAxiosError(err) && err.response?.data?.error) {
        errorMessage = err.response.data.error
      }
      setState((prev) => ({
        ...prev,
        isLoadingSaliency: false,
        error: errorMessage,
      }))
    }
  }, [])

  const reset = useCallback(() => {
    setState(initialState)
  }, [])

  return {
    ...state,
    analyze,
    loadGradCAM,
    loadSaliency,
    reset,
  }
}
