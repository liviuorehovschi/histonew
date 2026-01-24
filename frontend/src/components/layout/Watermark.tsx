export function Watermark() {
  return (
    <a
      href="https://orehovschi.com"
      target="_blank"
      rel="noopener noreferrer"
      className="fixed bottom-4 right-4 z-50 opacity-[0.08] transition-opacity duration-300 hover:opacity-25"
      aria-label="Created by Orehovschi"
    >
      <svg
        width="80"
        height="24"
        viewBox="0 0 80 24"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className="text-foreground"
      >
        <text
          x="0"
          y="18"
          fill="currentColor"
          fontFamily="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"
          fontSize="12"
          fontWeight="500"
          letterSpacing="0.05em"
        >
          Orehovschi
        </text>
      </svg>
    </a>
  )
}
