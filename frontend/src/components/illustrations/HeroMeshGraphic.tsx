import React from 'react';

interface HeroMeshGraphicProps {
  className?: string;
  width?: number | string;
  height?: number | string;
}

export const HeroMeshGraphic: React.FC<HeroMeshGraphicProps> = ({
  className = "w-full max-w-lg mx-auto",
  width = 500,
  height = 360
}) => {
  return (
    <svg
      viewBox="0 0 500 360"
      width={width}
      height={height}
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      role="img"
      aria-label="Facial Component Guided Neural Grid Analysis"
    >
      <defs>
        <linearGradient id="gridLineGrad" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#0ea5e9" stopOpacity="0.15" />
          <stop offset="50%" stopColor="#3b82f6" stopOpacity="0.3" />
          <stop offset="100%" stopColor="#0284c7" stopOpacity="0.08" />
        </linearGradient>

        <linearGradient id="scanWave" x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stopColor="#38bdf8" stopOpacity="0" />
          <stop offset="50%" stopColor="#38bdf8" stopOpacity="0.65" />
          <stop offset="100%" stopColor="#0284c7" stopOpacity="0" />
        </linearGradient>

        <linearGradient id="spatialGlow" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="#06b6d4" />
          <stop offset="100%" stopColor="#3b82f6" />
        </linearGradient>

        <linearGradient id="temporalGlow" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="#a855f7" />
          <stop offset="100%" stopColor="#ec4899" />
        </linearGradient>

        <filter id="meshGlow" x="-20%" y="-20%" width="140%" height="140%">
          <feGaussianBlur stdDeviation="3" result="blur" />
          <feComposite in="SourceGraphic" in2="blur" operator="over" />
        </filter>
      </defs>

      {/* Background Frame with HUD Crosshairs */}
      <rect x="20" y="20" width="460" height="320" rx="16" fill="#030712" stroke="#1e293b" strokeWidth="1.2" />
      
      {/* Corner Brackets */}
      <path d="M 28 44 L 28 28 L 44 28" stroke="#38bdf8" strokeWidth="2" strokeLinecap="round" />
      <path d="M 472 44 L 472 28 L 456 28" stroke="#38bdf8" strokeWidth="2" strokeLinecap="round" />
      <path d="M 28 316 L 28 332 L 44 332" stroke="#38bdf8" strokeWidth="2" strokeLinecap="round" />
      <path d="M 472 316 L 472 332 L 456 332" stroke="#38bdf8" strokeWidth="2" strokeLinecap="round" />

      {/* HUD Telemetry Top Bar */}
      <text x="36" y="46" fill="#64748b" fontSize="10" fontFamily="monospace" letterSpacing="1">
        SYS.DFD-FCG // CVPR&apos;25 ViT-L/14 // SHAPLEY ADDITIVE
      </text>
      <circle cx="456" cy="42" r="3.5" fill="#10b981" />
      <text x="412" y="46" fill="#10b981" fontSize="9" fontFamily="monospace">ONLINE</text>

      {/* 16x16 Neural Patch Sub-grid (Center Face Focus) */}
      <g stroke="url(#gridLineGrad)" strokeWidth="0.8">
        {/* Vertical lines */}
        {[170, 186, 202, 218, 234, 250, 266, 282, 298, 314, 330].map((x) => (
          <line key={`v-${x}`} x1={x} y1="70" x2={x} y2="270" strokeDasharray={x % 32 === 0 ? "none" : "2 2"} />
        ))}
        {/* Horizontal lines */}
        {[80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256].map((y) => (
          <line key={`h-${y}`} x1="160" y1={y} x2="340" y2={y} strokeDasharray={y % 32 === 0 ? "none" : "2 2"} />
        ))}
      </g>

      {/* Facial Contour Geometry */}
      <path
        d="M 250 85 C 295 85 320 120 320 170 C 320 220 288 252 250 268 C 212 252 180 220 180 170 C 180 120 205 85 250 85 Z"
        stroke="#38bdf8"
        strokeWidth="1.8"
        strokeOpacity="0.75"
        fill="#0ea5e9"
        fillOpacity="0.03"
      />

      {/* 68-Point Landmark Wireframe Mesh */}
      {/* Eyebrows */}
      <path d="M 198 126 Q 214 120 230 124" stroke="#38bdf8" strokeWidth="1.5" strokeLinecap="round" />
      <path d="M 270 124 Q 286 120 302 126" stroke="#38bdf8" strokeWidth="1.5" strokeLinecap="round" />
      
      {/* Eyes & Diagnostic Scan Circles */}
      <path d="M 204 142 Q 216 136 228 142 Q 216 148 204 142 Z" stroke="#38bdf8" strokeWidth="1.4" fill="#0284c7" fillOpacity="0.2" />
      <circle cx="216" cy="142" r="2.5" fill="#38bdf8" filter="url(#meshGlow)" />
      <circle cx="216" cy="142" r="6" stroke="#38bdf8" strokeWidth="0.8" strokeOpacity="0.4" />

      <path d="M 272 142 Q 284 136 296 142 Q 284 148 272 142 Z" stroke="#38bdf8" strokeWidth="1.4" fill="#0284c7" fillOpacity="0.2" />
      <circle cx="284" cy="142" r="2.5" fill="#38bdf8" filter="url(#meshGlow)" />
      <circle cx="284" cy="142" r="6" stroke="#38bdf8" strokeWidth="0.8" strokeOpacity="0.4" />

      {/* Nose Bridge and Base */}
      <path d="M 250 134 L 250 178 L 242 184 L 258 184" stroke="#38bdf8" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
      <circle cx="250" cy="178" r="2" fill="#38bdf8" />

      {/* Lips Wireframe */}
      <path d="M 226 210 Q 250 205 274 210 Q 250 220 226 210 Z" stroke="#38bdf8" strokeWidth="1.5" fill="#0284c7" fillOpacity="0.2" />
      <line x1="234" y1="210" x2="266" y2="210" stroke="#38bdf8" strokeWidth="1" strokeOpacity="0.8" />

      {/* Chin Indicator */}
      <circle cx="250" cy="256" r="2.5" fill="#38bdf8" />

      {/* Dynamic Vertical Scan Beam */}
      <rect x="160" y="150" width="180" height="24" fill="url(#scanWave)" />
      <line x1="160" y1="162" x2="340" y2="162" stroke="#38bdf8" strokeWidth="1.5" strokeOpacity="0.9" />

      {/* Left Feature Callout: Spatial Branch (FCG) */}
      <g transform="translate(40, 100)">
        <rect x="0" y="0" width="105" height="58" rx="8" fill="#0b1329" stroke="#1d4ed8" strokeWidth="1" />
        <circle cx="14" cy="16" r="4" fill="#06b6d4" />
        <text x="24" y="20" fill="#f8fafc" fontSize="10" fontWeight="600" fontFamily="sans-serif">Spatial Head</text>
        <text x="12" y="36" fill="#94a3b8" fontSize="8" fontFamily="monospace">φ_S (FCG Queries)</text>
        <text x="12" y="48" fill="#38bdf8" fontSize="9" fontWeight="bold" fontFamily="monospace">EYES · LIPS · NOSE</text>
        {/* Pointer line */}
        <path d="M 105 28 L 175 142" stroke="#06b6d4" strokeWidth="1" strokeDasharray="3 3" strokeOpacity="0.6" />
      </g>

      {/* Right Feature Callout: Temporal Branch (Motion dynamics) */}
      <g transform="translate(355, 100)">
        <rect x="0" y="0" width="105" height="58" rx="8" fill="#180b29" stroke="#7e22ce" strokeWidth="1" />
        <circle cx="14" cy="16" r="4" fill="#a855f7" />
        <text x="24" y="20" fill="#f8fafc" fontSize="10" fontWeight="600" fontFamily="sans-serif">Temporal Head</text>
        <text x="12" y="36" fill="#94a3b8" fontSize="8" fontFamily="monospace">φ_T (16×16 Dynamics)</text>
        <text x="12" y="48" fill="#c084fc" fontSize="9" fontWeight="bold" fontFamily="monospace">MICRO-JITTER</text>
        {/* Pointer line */}
        <path d="M 0 28 L -65 180" stroke="#a855f7" strokeWidth="1" strokeDasharray="3 3" strokeOpacity="0.6" />
      </g>

      {/* Bottom Math Proof Banner */}
      <g transform="translate(100, 290)">
        <rect x="0" y="0" width="300" height="34" rx="8" fill="#0f172a" stroke="#334155" strokeWidth="0.8" />
        <text x="150" y="16" fill="#94a3b8" fontSize="9" textAnchor="middle" fontFamily="monospace">
          EXACT SHAPLEY BRANCH DECOMPOSITION
        </text>
        <text x="150" y="27" fill="#38bdf8" fontSize="10" textAnchor="middle" fontWeight="bold" fontFamily="monospace">
          φ_S + φ_T = P(FAKE) - 0.50 [Zero Residual]
        </text>
      </g>
    </svg>
  );
};