import { motion } from 'framer-motion'

const stats = [
  { value: '98.25%', label: 'Accuracy', description: 'Test set performance' },
  { value: '25K+', label: 'Images', description: 'Training dataset size' },
  { value: '3', label: 'Classes', description: 'Diagnostic categories' },
]

export function Stats() {
  return (
    <section className="border-y border-border bg-card/50 py-16">
      <div className="container mx-auto px-4">
        <div className="grid grid-cols-1 gap-8 sm:grid-cols-3">
          {stats.map((stat, index) => (
            <motion.div
              key={stat.label}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.4, delay: index * 0.1 }}
              className="text-center"
            >
              <p className="text-4xl font-bold text-primary">{stat.value}</p>
              <p className="mt-2 font-medium">{stat.label}</p>
              <p className="text-sm text-muted-foreground">{stat.description}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  )
}
