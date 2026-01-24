import { X } from 'lucide-react'
import { motion } from 'framer-motion'
import { Button } from '@/components/ui/button'

interface ImagePreviewProps {
  imageUrl: string
  fileName: string
  onRemove: () => void
}

export function ImagePreview({ imageUrl, fileName, onRemove }: ImagePreviewProps) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.3 }}
      className="relative overflow-hidden rounded-lg border border-border bg-card"
    >
      <div className="relative aspect-square w-full max-w-[400px] mx-auto">
        <img
          src={imageUrl}
          alt={fileName}
          className="h-full w-full object-contain p-4"
        />
      </div>

      <div className="flex items-center justify-between border-t border-border bg-muted/30 px-4 py-3">
        <span className="truncate text-sm text-muted-foreground">{fileName}</span>
        <Button
          variant="ghost"
          size="sm"
          onClick={onRemove}
          className="h-8 gap-1.5 text-muted-foreground hover:text-destructive"
        >
          <X className="h-4 w-4" />
          <span className="hidden sm:inline">Remove</span>
        </Button>
      </div>
    </motion.div>
  )
}
