import { Info } from 'lucide-react'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip'

export function ConfidenceTooltip() {
  return (
    <TooltipProvider>
      <Tooltip delayDuration={200}>
        <TooltipTrigger asChild>
          <button
            className="inline-flex items-center justify-center rounded-full p-1 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
            aria-label="Confidence level explanation"
          >
            <Info className="h-4 w-4" />
          </button>
        </TooltipTrigger>
        <TooltipContent
          side="top"
          className="max-w-[280px] p-4"
        >
          <div className="space-y-3">
            <p className="font-medium">Understanding Confidence Levels</p>
            <ul className="space-y-2 text-sm">
              <li className="flex items-start gap-2">
                <span className="mt-1 h-2 w-2 shrink-0 rounded-full bg-success" />
                <span>
                  <strong>High (90%+):</strong> Strong prediction confidence
                </span>
              </li>
              <li className="flex items-start gap-2">
                <span className="mt-1 h-2 w-2 shrink-0 rounded-full bg-warning" />
                <span>
                  <strong>Moderate (60-89%):</strong> Consider verification
                </span>
              </li>
              <li className="flex items-start gap-2">
                <span className="mt-1 h-2 w-2 shrink-0 rounded-full bg-destructive" />
                <span>
                  <strong>Low (&lt;60%):</strong> Manual review recommended
                </span>
              </li>
            </ul>
            <p className="text-xs text-muted-foreground">
              This is an AI-assisted tool. Always consult qualified medical professionals
              for clinical decisions.
            </p>
          </div>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  )
}
