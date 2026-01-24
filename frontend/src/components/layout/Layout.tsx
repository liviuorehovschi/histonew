import type { ReactNode } from 'react'
import { Navbar } from './Navbar'
import { Watermark } from './Watermark'

interface LayoutProps {
  children: ReactNode
}

export function Layout({ children }: LayoutProps) {
  return (
    <div className="relative min-h-screen bg-background">
      <Navbar />
      <main className="flex-1">{children}</main>
      <Watermark />
    </div>
  )
}
