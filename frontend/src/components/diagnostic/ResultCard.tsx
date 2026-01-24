import { motion } from 'framer-motion'
import { Activity, Eye, Layers } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { ConfidenceTooltip } from './ConfidenceTooltip'
import { cn } from '@/lib/utils'
import type { PredictionResponse } from '@/lib/api'

interface ResultCardProps {
  prediction: PredictionResponse
  onViewGradCAM: () => void
  onViewSaliency: () => void
  isLoadingGradcam: boolean
  isLoadingSaliency: boolean
}

function getConfidenceColor(level: PredictionResponse['confidenceLevel']) {
  switch (level) {
    case 'high':
      return 'text-success'
    case 'moderate':
      return 'text-warning'
    case 'low':
      return 'text-destructive'
    default:
      return 'text-foreground'
  }
}

function getConfidenceBadgeColor(level: PredictionResponse['confidenceLevel']) {
  switch (level) {
    case 'high':
      return 'bg-success/10 text-success border-success/20'
    case 'moderate':
      return 'bg-warning/10 text-warning border-warning/20'
    case 'low':
      return 'bg-destructive/10 text-destructive border-destructive/20'
    default:
      return 'bg-muted text-foreground'
  }
}

export function ResultCard({
  prediction,
  onViewGradCAM,
  onViewSaliency,
  isLoadingGradcam,
  isLoadingSaliency,
}: ResultCardProps) {
  const confidencePercent = (prediction.confidence * 100).toFixed(1)

  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.4, ease: 'easeOut' }}
    >
      <Card>
        <CardHeader className="pb-4">
          <div className="flex items-center gap-2">
            <Activity className="h-5 w-5 text-primary" />
            <CardTitle className="text-lg">Analysis Results</CardTitle>
          </div>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Diagnosis */}
          <div className="space-y-2">
            <p className="text-sm text-muted-foreground">Diagnosis</p>
            <p className="text-2xl font-semibold">{prediction.diagnosis}</p>
          </div>

          {/* Confidence */}
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <p className="text-sm text-muted-foreground">Confidence</p>
              <ConfidenceTooltip />
            </div>
            <div className="flex items-center gap-3">
              <span
                className={cn(
                  'text-3xl font-bold tabular-nums',
                  getConfidenceColor(prediction.confidenceLevel)
                )}
              >
                {confidencePercent}%
              </span>
              <span
                className={cn(
                  'rounded-full border px-2.5 py-0.5 text-xs font-medium capitalize',
                  getConfidenceBadgeColor(prediction.confidenceLevel)
                )}
              >
                {prediction.confidenceLevel}
              </span>
            </div>
          </div>

          {/* Visualization buttons */}
          <div className="space-y-3 pt-2">
            <p className="text-sm text-muted-foreground">Model Explainability</p>
            <div className="flex flex-wrap gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={onViewGradCAM}
                disabled={isLoadingGradcam}
                className="gap-2"
              >
                <Eye className="h-4 w-4" />
                {isLoadingGradcam ? 'Loading...' : 'View Grad-CAM'}
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={onViewSaliency}
                disabled={isLoadingSaliency}
                className="gap-2"
              >
                <Layers className="h-4 w-4" />
                {isLoadingSaliency ? 'Loading...' : 'View Saliency Map'}
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  )
}
