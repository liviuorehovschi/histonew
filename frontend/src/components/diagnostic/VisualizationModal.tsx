import { AlertTriangle } from 'lucide-react'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Skeleton } from '@/components/ui/skeleton'

interface VisualizationModalProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  title: string
  description: string
  imageUrl: string | null
  isLoading: boolean
}

export function VisualizationModal({
  open,
  onOpenChange,
  title,
  description,
  imageUrl,
  isLoading,
}: VisualizationModalProps) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
          <DialogDescription>{description}</DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          {/* Image container */}
          <div className="relative aspect-square w-full overflow-hidden rounded-lg border border-border bg-muted">
            {isLoading ? (
              <div className="flex h-full items-center justify-center">
                <div className="space-y-4 p-8 text-center">
                  <Skeleton className="mx-auto h-48 w-48 rounded-lg" />
                  <p className="text-sm text-muted-foreground">
                    Generating visualization...
                  </p>
                </div>
              </div>
            ) : imageUrl ? (
              <img
                src={imageUrl}
                alt={title}
                className="h-full w-full object-contain p-2"
              />
            ) : (
              <div className="flex h-full items-center justify-center">
                <p className="text-muted-foreground">No image available</p>
              </div>
            )}
          </div>

          {/* Disclaimer */}
          <div className="flex gap-3 rounded-lg border border-border bg-muted/30 p-3">
            <AlertTriangle className="h-5 w-5 shrink-0 text-warning" />
            <p className="text-xs text-muted-foreground">
              This visualization is a model explanation tool and should not be used as
              clinical evidence. It highlights regions the model found relevant for its
              prediction. Always consult qualified medical professionals for diagnosis
              and treatment decisions.
            </p>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}
