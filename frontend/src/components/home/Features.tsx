import { motion } from 'framer-motion'
import { Brain, Eye, Zap } from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'

const features = [
  {
    icon: Brain,
    title: 'Deep Learning Classification',
    description:
      'Powered by EfficientNetB0, our model accurately classifies lung tissue into adenocarcinoma, squamous cell carcinoma, or normal tissue.',
  },
  {
    icon: Eye,
    title: 'Explainable AI',
    description:
      'Grad-CAM and saliency map visualizations help you understand which regions influenced the model prediction.',
  },
  {
    icon: Zap,
    title: 'Instant Results',
    description:
      'Get classification results in seconds. Our optimized pipeline ensures fast, reliable analysis of histopathology images.',
  },
]

export function Features() {
  return (
    <section className="py-20">
      <div className="container mx-auto px-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.4 }}
          className="mx-auto max-w-2xl text-center"
        >
          <h2 className="text-3xl font-bold">How It Works</h2>
          <p className="mt-4 text-muted-foreground">
            Our AI-powered diagnostic tool combines state-of-the-art deep learning
            with explainable AI techniques.
          </p>
        </motion.div>

        <div className="mt-16 grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {features.map((feature, index) => {
            const Icon = feature.icon
            return (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.4, delay: index * 0.1 }}
              >
                <Card className="h-full transition-colors hover:border-primary/50">
                  <CardHeader>
                    <div className="mb-2 w-fit rounded-lg bg-primary/10 p-2.5">
                      <Icon className="h-5 w-5 text-primary" />
                    </div>
                    <CardTitle className="text-lg">{feature.title}</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <CardDescription className="text-sm leading-relaxed">
                      {feature.description}
                    </CardDescription>
                  </CardContent>
                </Card>
              </motion.div>
            )
          })}
        </div>
      </div>
    </section>
  )
}
