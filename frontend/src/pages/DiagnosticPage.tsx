import { useState, useCallback } from 'react'
import { motion } from 'framer-motion'
import { Loader2, Activity, AlertTriangle } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { ImageUpload } from '@/components/diagnostic/ImageUpload'
import { ImagePreview } from '@/components/diagnostic/ImagePreview'
import { ResultCard } from '@/components/diagnostic/ResultCard'
import { VisualizationModal } from '@/components/diagnostic/VisualizationModal'
import { usePrediction } from '@/hooks/usePrediction'

export function DiagnosticPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [imagePreviewUrl, setImagePreviewUrl] = useState<string | null>(null)
  const [isDragActive, setIsDragActive] = useState(false)
  const [gradcamModalOpen, setGradcamModalOpen] = useState(false)
  const [saliencyModalOpen, setSaliencyModalOpen] = useState(false)

  const {
    prediction,
    gradcamImage,
    saliencyImage,
    isAnalyzing,
    isLoadingGradcam,
    isLoadingSaliency,
    error,
    analyze,
    loadGradCAM,
    loadSaliency,
    reset,
  } = usePrediction()

  const handleImageSelect = useCallback((file: File) => {
    setSelectedFile(file)
    const url = URL.createObjectURL(file)
    setImagePreviewUrl(url)
    reset()
  }, [reset])

  const handleRemoveImage = useCallback(() => {
    if (imagePreviewUrl) {
      URL.revokeObjectURL(imagePreviewUrl)
    }
    setSelectedFile(null)
    setImagePreviewUrl(null)
    reset()
  }, [imagePreviewUrl, reset])

  const handleAnalyze = useCallback(async () => {
    if (selectedFile) {
      await analyze(selectedFile)
    }
  }, [selectedFile, analyze])

  const handleViewGradCAM = useCallback(async () => {
    setGradcamModalOpen(true)
    if (selectedFile && !gradcamImage) {
      await loadGradCAM(selectedFile)
    }
  }, [selectedFile, gradcamImage, loadGradCAM])

  const handleViewSaliency = useCallback(async () => {
    setSaliencyModalOpen(true)
    if (selectedFile && !saliencyImage) {
      await loadSaliency(selectedFile)
    }
  }, [selectedFile, saliencyImage, loadSaliency])

  return (
    <div className="py-12">
      <div className="container mx-auto max-w-4xl px-4">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
          className="mb-8 text-center"
        >
          <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-primary/10">
            <Activity className="h-6 w-6 text-primary" />
          </div>
          <h1 className="text-3xl font-bold">Tissue Analysis</h1>
          <p className="mt-2 text-muted-foreground">
            Upload a lung histopathology image for AI-powered classification
          </p>
        </motion.div>

        <div className="grid gap-6 lg:grid-cols-2">
          {/* Left column: Upload / Preview */}
          <div className="space-y-4">
            {imagePreviewUrl ? (
              <ImagePreview
                imageUrl={imagePreviewUrl}
                fileName={selectedFile?.name ?? 'image'}
                onRemove={handleRemoveImage}
              />
            ) : (
              <ImageUpload
                onImageSelect={handleImageSelect}
                isDragActive={isDragActive}
                onDragEnter={() => setIsDragActive(true)}
                onDragLeave={() => setIsDragActive(false)}
              />
            )}

            {/* Analyze button */}
            <Button
              onClick={handleAnalyze}
              disabled={!selectedFile || isAnalyzing}
              className="w-full gap-2"
              size="lg"
            >
              {isAnalyzing ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Analyzing...
                </>
              ) : (
                'Analyze Image'
              )}
            </Button>

            {/* Error message */}
            {error && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="rounded-lg border border-destructive/50 bg-destructive/10 p-4"
              >
                <p className="text-sm text-destructive">{error}</p>
              </motion.div>
            )}
          </div>

          {/* Right column: Results */}
          <div>
            {prediction ? (
              <ResultCard
                prediction={prediction}
                onViewGradCAM={handleViewGradCAM}
                onViewSaliency={handleViewSaliency}
                isLoadingGradcam={isLoadingGradcam}
                isLoadingSaliency={isLoadingSaliency}
              />
            ) : (
              <Card className="h-full min-h-[300px]">
                <CardContent className="flex h-full items-center justify-center p-6">
                  <div className="text-center">
                    <p className="text-muted-foreground">
                      {selectedFile
                        ? 'Click "Analyze Image" to run classification'
                        : 'Upload an image to begin'}
                    </p>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        </div>

        {/* Disclaimer */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.4, delay: 0.3 }}
          className="mt-8"
        >
          <div className="flex gap-3 rounded-lg border border-border bg-card p-4">
            <AlertTriangle className="h-5 w-5 shrink-0 text-warning" />
            <div className="space-y-1">
              <p className="text-sm font-medium">Educational Tool Disclaimer</p>
              <p className="text-xs text-muted-foreground">
                This tool is for educational and research purposes only. It is not intended
                for clinical diagnosis or medical decision-making. Always consult qualified
                healthcare professionals for medical advice and treatment.
              </p>
            </div>
          </div>
        </motion.div>
      </div>

      {/* Visualization Modals */}
      <VisualizationModal
        open={gradcamModalOpen}
        onOpenChange={setGradcamModalOpen}
        title="Grad-CAM Visualization"
        description="Highlights regions that most influenced the model's prediction"
        imageUrl={gradcamImage}
        isLoading={isLoadingGradcam}
      />

      <VisualizationModal
        open={saliencyModalOpen}
        onOpenChange={setSaliencyModalOpen}
        title="Saliency Map"
        description="Shows pixel-level importance for the classification"
        imageUrl={saliencyImage}
        isLoading={isLoadingSaliency}
      />
    </div>
  )
}
