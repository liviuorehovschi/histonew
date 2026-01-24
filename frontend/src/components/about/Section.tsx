import { motion } from 'framer-motion'
import type { ReactNode } from 'react'

interface SectionProps {
  title: string
  children: ReactNode
  delay?: number
}

export function Section({ title, children, delay = 0 }: SectionProps) {
  return (
    <motion.section
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.4, delay }}
      className="space-y-4"
    >
      <h2 className="text-xl font-semibold">{title}</h2>
      <div className="text-muted-foreground leading-relaxed">{children}</div>
    </motion.section>
  )
}
