import { motion } from 'framer-motion'
import { Info } from 'lucide-react'
import { Section } from '@/components/about/Section'
import { AuthorCard } from '@/components/about/AuthorCard'

export function AboutPage() {
  return (
    <div className="py-12">
      <div className="container mx-auto max-w-3xl px-4">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
          className="mb-12 text-center"
        >
          <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-primary/10">
            <Info className="h-6 w-6 text-primary" />
          </div>
          <h1 className="text-3xl font-bold">About Histomancer</h1>
          <p className="mt-2 text-muted-foreground">
            AI-powered lung cancer detection from histopathology slides
          </p>
        </motion.div>

        {/* Content sections */}
        <div className="space-y-10">
          <Section title="The Problem" delay={0.1}>
            <p>
              Lung cancer remains one of the leading causes of cancer-related deaths worldwide.
              Early and accurate detection is crucial for improving patient outcomes. However,
              analyzing histopathology slides is time-consuming and requires significant expertise,
              making it challenging to scale diagnostic capabilities in resource-limited settings.
            </p>
          </Section>

          <Section title="Our Solution" delay={0.15}>
            <p>
              Histomancer uses deep learning to classify lung tissue samples into three categories:
            </p>
            <ul className="mt-4 list-inside list-disc space-y-2">
              <li>
                <strong>Lung Adenocarcinoma</strong> - The most common type of lung cancer
              </li>
              <li>
                <strong>Lung Squamous Cell Carcinoma</strong> - Another major subtype of lung cancer
              </li>
              <li>
                <strong>Normal Lung Tissue</strong> - Healthy tissue with no cancerous cells
              </li>
            </ul>
          </Section>

          <Section title="Technical Details" delay={0.2}>
            <p className="mb-4">
              The system is built on EfficientNetB0, a state-of-the-art convolutional neural network
              architecture known for its excellent accuracy-to-computation ratio.
            </p>
            <div className="rounded-lg border border-border bg-card p-4">
              <dl className="grid gap-3 text-sm sm:grid-cols-2">
                <div>
                  <dt className="font-medium text-foreground">Model Architecture</dt>
                  <dd>EfficientNetB0 (Transfer Learning)</dd>
                </div>
                <div>
                  <dt className="font-medium text-foreground">Input Resolution</dt>
                  <dd>224 x 224 pixels</dd>
                </div>
                <div>
                  <dt className="font-medium text-foreground">Training Dataset</dt>
                  <dd>25,000+ histopathology images</dd>
                </div>
                <div>
                  <dt className="font-medium text-foreground">Test Accuracy</dt>
                  <dd>98.25%</dd>
                </div>
              </dl>
            </div>
          </Section>

          <Section title="Explainability" delay={0.25}>
            <p>
              Understanding why an AI model makes certain predictions is crucial in medical applications.
              Histomancer provides two visualization techniques:
            </p>
            <ul className="mt-4 list-inside list-disc space-y-2">
              <li>
                <strong>Grad-CAM</strong> - Highlights regions of the image that most influenced
                the model's prediction by analyzing gradients at the final convolutional layer
              </li>
              <li>
                <strong>Saliency Maps</strong> - Shows pixel-level importance by computing
                gradients of the prediction with respect to input pixels
              </li>
            </ul>
          </Section>

          <Section title="Limitations & Disclaimer" delay={0.3}>
            <div className="rounded-lg border border-warning/30 bg-warning/5 p-4">
              <p className="text-sm">
                <strong className="text-warning">Important:</strong> Histomancer is an educational
                tool and research prototype. It is <strong>not</strong> intended for clinical diagnosis
                or to replace professional medical judgment. The predictions should be verified by
                qualified pathologists and healthcare professionals. Never make medical decisions
                based solely on this tool's output.
              </p>
            </div>
          </Section>

          {/* Author section */}
          <div className="pt-6">
            <h2 className="mb-4 text-xl font-semibold">Created By</h2>
            <AuthorCard />
          </div>
        </div>
      </div>
    </div>
  )
}
