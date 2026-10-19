import { useId } from 'react'

export function BrandMark() {
  const gradientId = `brand-pick-${useId().replaceAll(':', '')}`

  return (
    <svg
      viewBox="0 0 32 32"
      role="img"
      aria-label="六弦拨片"
    >
      <defs>
        <linearGradient
          id={gradientId}
          x1="16"
          y1="5.5"
          x2="16"
          y2="28"
          gradientUnits="userSpaceOnUse"
        >
          <stop stopColor="var(--brand-start)" />
          <stop offset="1" stopColor="var(--brand-end)" />
        </linearGradient>
      </defs>
      <rect
        x="2.9"
        y="2.9"
        width="26.2"
        height="26.2"
        rx="6.1"
        fill="var(--brand-tile)"
        stroke="var(--brand-tile-border)"
        strokeWidth="0.35"
      />
      <path
        d="M16 5.76C9.92 5.12 5.76 8.64 6.72 13.76c.96 5.76 5.76 11.52 9.28 14.08 3.52-2.56 8.32-8.32 9.28-14.08C26.24 8.64 22.08 5.12 16 5.76Z"
        fill={`url(#${gradientId})`}
      />
      <g
        fill="none"
        stroke="var(--brand-symbol-ink)"
        strokeLinecap="round"
      >
        <circle cx="16" cy="13.76" r="3.68" strokeWidth="0.8" />
        {[10.56, 12.74, 14.91, 17.09, 19.26, 21.44].map((x, index) => (
          <line
            key={x}
            x1={x}
            y1="7.84"
            x2={x}
            y2="22.24"
            strokeWidth={0.38 - index * 0.025}
          />
        ))}
        <line x1="11.36" y1="19.84" x2="20.64" y2="19.84" strokeWidth="1.02" />
      </g>
    </svg>
  )
}
