import React from 'react';

interface PipelineDiagramGraphicProps {
  className?: string;
}

export const PipelineDiagramGraphic: React.FC<PipelineDiagramGraphicProps> = ({
  className = "w-full max-w-4xl mx-auto my-8"
}) => {
  return (
    <svg
      viewBox="0 0 760 140"
      width="100%"
      height="140"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      role="img"
      aria-label="Forensic Deepfake Detection Pipeline Workflow"
    >
      <defs>
        <linearGradient id="pipeGrad" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="#0ea5e9" />
          <stop offset="50%" stopColor="#3b82f6" />
          <stop offset="100%" stopColor="#8b5cf6" />
        </linearGradient>
      </defs>

      {/* Stage 1: Ingest & 68-pt Alignment */}
      <g transform="translate(10, 20)">
        <rect x="0" y="0" width="160" height="96" rx="12" fill="#0b1329" stroke="#1e3a8a" strokeWidth="1.2" />
        <rect x="12" y="12" width="28" height="28" rx="6" fill="#0284c7" fillOpacity="0.2" />
        <path d="M 21 21 L 31 26 L 21 31 Z" fill="#38bdf8" />
        <text x="48" y="24" fill="#94a3b8" fontSize="9" fontFamily="monospace">STAGE 01</text>
        <text x="48" y="37" fill="#f8fafc" fontSize="11" fontWeight="bold">Video Ingest</text>
        <text x="12" y="62" fill="#64748b" fontSize="9" fontFamily="sans-serif">68-Point Landmark Track</text>
        <text x="12" y="76" fill="#0284c7" fontSize="9" fontFamily="monospace">150×150 Aligned Crops</text>
      </g>

      {/* Connector Arrow 1 */}
      <path d="M 175 68 L 195 68" stroke="url(#pipeGrad)" strokeWidth="2" strokeDasharray="3 3" />
      <polygon points="195,64 203,68 195,72" fill="#38bdf8" />

      {/* Stage 2: Foundation Model ViT-L/14 */}
      <g transform="translate(205, 20)">
        <rect x="0" y="0" width="160" height="96" rx="12" fill="#0b172b" stroke="#1e40af" strokeWidth="1.2" />
        <rect x="12" y="12" width="28" height="28" rx="6" fill="#2563eb" fillOpacity="0.2" />
        <circle cx="26" cy="26" r="6" stroke="#60a5fa" strokeWidth="1.5" />
        <text x="48" y="24" fill="#94a3b8" fontSize="9" fontFamily="monospace">STAGE 02</text>
        <text x="48" y="37" fill="#f8fafc" fontSize="11" fontWeight="bold">CLIP ViT-L/14</text>
        <text x="12" y="62" fill="#64748b" fontSize="9" fontFamily="sans-serif">Frozen 24-Layer Tokens</text>
        <text x="12" y="76" fill="#3b82f6" fontSize="9" fontFamily="monospace">16×16 Patch Dynamics</text>
      </g>

      {/* Connector Arrow 2 */}
      <path d="M 370 68 L 390 68" stroke="url(#pipeGrad)" strokeWidth="2" strokeDasharray="3 3" />
      <polygon points="390,64 398,68 390,72" fill="#60a5fa" />

      {/* Stage 3: Spatial (FCG) & Temporal Branches */}
      <g transform="translate(400, 20)">
        <rect x="0" y="0" width="160" height="96" rx="12" fill="#130e2e" stroke="#4c1d95" strokeWidth="1.2" />
        <rect x="12" y="12" width="28" height="28" rx="6" fill="#7c3aed" fillOpacity="0.2" />
        <rect x="20" y="20" width="12" height="12" rx="2" stroke="#a78bfa" strokeWidth="1.5" />
        <text x="48" y="24" fill="#94a3b8" fontSize="9" fontFamily="monospace">STAGE 03</text>
        <text x="48" y="37" fill="#f8fafc" fontSize="11" fontWeight="bold">DFD-FCG Heads</text>
        <text x="12" y="62" fill="#64748b" fontSize="9" fontFamily="sans-serif">Spatial (4 Anatomy Queries)</text>
        <text x="12" y="76" fill="#8b5cf6" fontSize="9" fontFamily="monospace">Temporal Frame Jitter</text>
      </g>

      {/* Connector Arrow 3 */}
      <path d="M 565 68 L 585 68" stroke="url(#pipeGrad)" strokeWidth="2" strokeDasharray="3 3" />
      <polygon points="585,64 593,68 585,72" fill="#8b5cf6" />

      {/* Stage 4: Shapley Explainability & PDF Report */}
      <g transform="translate(595, 20)">
        <rect x="0" y="0" width="155" height="96" rx="12" fill="#150a26" stroke="#581c87" strokeWidth="1.2" />
        <rect x="12" y="12" width="28" height="28" rx="6" fill="#9333ea" fillOpacity="0.2" />
        <path d="M 21 21 H 31 V 31 H 21 Z" stroke="#c084fc" strokeWidth="1.4" fill="none" />
        <text x="48" y="24" fill="#94a3b8" fontSize="9" fontFamily="monospace">STAGE 04</text>
        <text x="48" y="37" fill="#f8fafc" fontSize="11" fontWeight="bold">Explainability</text>
        <text x="12" y="62" fill="#64748b" fontSize="9" fontFamily="sans-serif">Zero-Residual Shapley</text>
        <text x="12" y="76" fill="#c084fc" fontSize="9" fontFamily="monospace">Heatmaps &amp; PDF Audit</text>
      </g>
    </svg>
  );
};