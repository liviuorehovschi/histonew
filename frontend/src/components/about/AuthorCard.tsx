import { motion } from 'framer-motion'
import { Github, Linkedin, Globe } from 'lucide-react'
import { Card, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'

export function AuthorCard() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.4, delay: 0.2 }}
    >
      <Card>
        <CardContent className="p-6">
          <div className="flex flex-col items-center gap-4 text-center sm:flex-row sm:text-left">
            {/* Avatar placeholder */}
            <div className="flex h-16 w-16 shrink-0 items-center justify-center rounded-full bg-primary/10 text-2xl font-bold text-primary">
              LO
            </div>

            <div className="flex-1">
              <h3 className="font-semibold">Liviu Orehovschi</h3>
              <p className="text-sm text-muted-foreground">
                Machine Learning Engineer & Full-Stack Developer
              </p>
            </div>

            <div className="flex gap-2">
              <Button variant="outline" size="icon" asChild>
                <a
                  href="https://orehovschi.com"
                  target="_blank"
                  rel="noopener noreferrer"
                  aria-label="Portfolio"
                >
                  <Globe className="h-4 w-4" />
                </a>
              </Button>
              <Button variant="outline" size="icon" asChild>
                <a
                  href="https://github.com/liviu-orehovschi"
                  target="_blank"
                  rel="noopener noreferrer"
                  aria-label="GitHub"
                >
                  <Github className="h-4 w-4" />
                </a>
              </Button>
              <Button variant="outline" size="icon" asChild>
                <a
                  href="https://linkedin.com/in/liviu-orehovschi"
                  target="_blank"
                  rel="noopener noreferrer"
                  aria-label="LinkedIn"
                >
                  <Linkedin className="h-4 w-4" />
                </a>
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  )
}
