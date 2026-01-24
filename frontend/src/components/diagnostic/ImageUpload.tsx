import { useCallback } from 'react'
import { Upload, Image as ImageIcon } from 'lucide-react'
import { motion } from 'framer-motion'
import { cn } from '@/lib/utils'

interface ImageUploadProps {
  onImageSelect: (file: File) => void
  isDragActive?: boolean
  onDragEnter?: () => void
  onDragLeave?: () => void
}

export function ImageUpload({
  onImageSelect,
  isDragActive = false,
  onDragEnter,
  onDragLeave,
}: ImageUploadProps) {
  const handleDrop = useCallback(
    (e: React.DragEvent<HTMLLabelElement>) => {
      e.preventDefault()
      onDragLeave?.()
      const file = e.dataTransfer.files[0]
      if (file && file.type.startsWith('image/')) {
        onImageSelect(file)
      }
    },
    [onImageSelect, onDragLeave]
  )

  const handleDragOver = useCallback((e: React.DragEvent<HTMLLabelElement>) => {
    e.preventDefault()
  }, [])

  const handleDragEnter = useCallback(
    (e: React.DragEvent<HTMLLabelElement>) => {
      e.preventDefault()
      onDragEnter?.()
    },
    [onDragEnter]
  )

  const handleDragLeave = useCallback(
    (e: React.DragEvent<HTMLLabelElement>) => {
      e.preventDefault()
      onDragLeave?.()
    },
    [onDragLeave]
  )

  const handleFileSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0]
      if (file) {
        onImageSelect(file)
      }
    },
    [onImageSelect]
  )

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <label
        className={cn(
          'group relative flex min-h-[300px] cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed transition-all duration-200',
          isDragActive
            ? 'border-primary bg-primary/5'
            : 'border-border hover:border-primary/50 hover:bg-muted/50'
        )}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
      >
        <input
          type="file"
          accept="image/*"
          onChange={handleFileSelect}
          className="sr-only"
        />

        <div className="flex flex-col items-center gap-4 p-8 text-center">
          <div
            className={cn(
              'rounded-full p-4 transition-colors',
              isDragActive ? 'bg-primary/10' : 'bg-muted group-hover:bg-primary/10'
            )}
          >
            {isDragActive ? (
              <ImageIcon className="h-8 w-8 text-primary" />
            ) : (
              <Upload className="h-8 w-8 text-muted-foreground group-hover:text-primary" />
            )}
          </div>

          <div className="space-y-2">
            <p className="text-lg font-medium">
              {isDragActive ? 'Drop your image here' : 'Upload histopathology image'}
            </p>
            <p className="text-sm text-muted-foreground">
              Drag and drop or click to browse
            </p>
            <p className="text-xs text-muted-foreground">
              Supports PNG, JPG, JPEG formats
            </p>
          </div>
        </div>
      </label>
    </motion.div>
  )
}
