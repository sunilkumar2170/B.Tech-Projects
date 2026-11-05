import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import {
  ShieldCheck,
  AlertTriangle,
  Download,
  FileText,
  Clock,
  Layers,
  Cpu,
  Eye,
  Sliders,
  ChevronLeft,
  ChevronRight,
  Info,
  Maximize2
} from 'lucide-react';
import {
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid
} from 'recharts';
import { api } from '../services/api';
import type { AnalysisResult } from '../types';

interface ResultsProps {
  analysisId?: string;
  initialResult?: AnalysisResult | null;
  onReset?: () => void;
}

export const Results: React.FC<ResultsProps> = ({
  analysisId: propAnalysisId,
  initialResult,
  onReset
}) => {
  const { analysisId: routeAnalysisId } = useParams<{ analysisId: string }>();
  const targetId = propAnalysisId || routeAnalysisId;

  const [result, setResult] = useState<AnalysisResult | null>(initialResult || null);
  const [isLoading, setIsLoading] = useState<boolean>(!initialResult && !!targetId);
  const [error, setError] = useState<string | null>(null);

  // Interactive state
  const [selectedClipIndex, setSelectedClipIndex] = useState<number>(0);
  const [currentFrameIdx, setCurrentFrameIdx] = useState<number>(0);
  const [activeHeatmapTab, setActiveHeatmapTab] = useState<'spatial' | 'temporal'>('spatial');

  useEffect(() => {
    if (initialResult) {
      setResult(initialResult);
      setIsLoading(false);
      return;
    }
    if (!targetId) {
      setIsLoading(false);
      setError('No analysis record provided.');
      return;
    }

    setIsLoading(true);
    api.getAnalysisResult(targetId)
      .then((data) => {
        setResult(data);
        setIsLoading(false);
      })
      .catch((err) => {
        console.error('Failed to load analysis result:', err);
        setError(err.response?.data?.detail || 'Could not load analysis results.');
        setIsLoading(false);
      });
  }, [targetId, initialResult]);

  if (isLoading) {
    return (
      <div className="min-h-[70vh] flex flex-col items-center justify-center p-8">
        <div className="w-12 h-12 rounded-2xl bg-blue-500/10 border border-blue-500/30 text-blue-400 flex items-center justify-center animate-spin mb-4">
          <Cpu className="w-6 h-6" />
        </div>
        <p className="text-sm font-semibold text-slate-200">Loading Forensic Analysis...</p>
        <p className="text-xs text-slate-500 font-mono mt-1">Retrieving verified attributions from storage</p>
      </div>
    );
  }

  if (error || !result) {
    return (
      <div className="max-w-xl mx-auto my-16 p-6 rounded-2xl bg-slate-900 border border-slate-800 text-center">
        <AlertTriangle className="w-10 h-10 text-rose-400 mx-auto mb-3" />
        <h2 className="text-lg font-bold text-slate-100 mb-2">Analysis Unavailable</h2>
        <p className="text-xs text-slate-400 mb-6">{error || 'Requested analysis record could not be found.'}</p>
        <Link
          to="/"
          className="px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold"
        >
          Analyze a Video
        </Link>
      </div>
    );
  }

  const verdict = result.verdict;
  const isManipulated = verdict.verdict === 'LIKELY MANIPULATED';
  const isAuthentic = verdict.verdict === 'LIKELY AUTHENTIC';

  const verdictColor = isManipulated
    ? 'text-rose-400'
    : isAuthentic
    ? 'text-emerald-400'
    : 'text-amber-400';

  const verdictBorder = isManipulated
    ? 'border-rose-500/30 bg-rose-950/10'
    : isAuthentic
    ? 'border-emerald-500/30 bg-emerald-950/10'
    : 'border-amber-500/30 bg-amber-950/10';

  // Branch share pie data
  const branchData = [
    { name: 'Spatial Branch (S)', value: result.branch_analysis.pct_spatial, color: '#2563eb' },
    { name: 'Temporal Branch (T)', value: result.branch_analysis.pct_temporal, color: '#f59e0b' },
  ];

  // Clip chart data
  const clipChartData = result.clips.map((c) => ({
    name: `Clip ${c.clip_index + 1}`,
    time: `${c.start_timestamp_s}s`,
    p_fake: Math.round(c.p_final * 100),
    p_real: Math.round(c.p_real * 100),
    spatial: Math.round(c.pct_spatial),
    temporal: Math.round(c.pct_temporal),
  }));

  const activeClip = (result.clips && result.clips.length > 0) ? (result.clips[selectedClipIndex] || result.clips[0]) : null;
  const activeFrame = (result.frames && result.frames.length > 0) ? (result.frames[currentFrameIdx] || result.frames[0]) : null;

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-10">
      {/* Top Banner / Actions */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between pb-6 border-b border-slate-800 gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-xs font-mono text-slate-500 uppercase tracking-wider">Forensic Audit</span>
            <span className="text-slate-600">·</span>
            <span className="text-xs font-mono text-slate-400">{result.filename}</span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-slate-100 tracking-tight">
            Deepfake Evidence Dashboard
          </h1>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          {onReset && (
            <button
              onClick={onReset}
              className="px-3.5 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 font-medium text-xs border border-slate-700 transition-all flex items-center gap-2"
            >
              <ChevronLeft className="w-4 h-4" />
              <span>Analyze Another Video</span>
            </button>
          )}
          <a
            href={api.getDownloadReportUrl(result.analysis_id)}
            download
            className="px-4 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-medium text-xs shadow-sm transition-all flex items-center gap-2"
          >
            <Download className="w-4 h-4" />
            <span>Download PDF Report</span>
          </a>
          <button
            onClick={() => {
              const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(result, null, 2));
              const downloadAnchor = document.createElement('a');
              downloadAnchor.setAttribute('href', dataStr);
              downloadAnchor.setAttribute('download', `forensic_${result.analysis_id}.json`);
              document.body.appendChild(downloadAnchor);
              downloadAnchor.click();
              downloadAnchor.remove();
            }}
            className="px-3.5 py-2.5 rounded-xl bg-slate-900 hover:bg-slate-800 text-slate-300 font-medium text-xs border border-slate-800 transition-colors flex items-center gap-1.5"
          >
            <FileText className="w-4 h-4 text-slate-400" />
            <span>Export JSON</span>
          </button>
        </div>
      </div>

      {/* SECTION A: OVERALL VERDICT CARD */}
      <div className={`p-6 sm:p-8 rounded-2xl border ${verdictBorder} shadow-xl`}>
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-center">
          <div className="lg:col-span-5 space-y-3">
            <div className="flex flex-wrap items-center gap-2">
              <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded-full bg-slate-900/80 border border-slate-800 text-xs font-mono text-slate-400">
                <ShieldCheck className="w-3.5 h-3.5 text-blue-400" />
                <span>DFD-FCG Foundation Architecture</span>
              </div>
              {verdict.inference_mode === 'benchmark' || (!verdict.inference_mode && result.analysis_id.startsWith('sample_')) ? (
                <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/20 text-cyan-300 text-[11px] font-mono">
                  <span className="w-1.5 h-1.5 rounded-full bg-cyan-400"></span>
                  Verified Research Benchmark
                </span>
              ) : verdict.inference_mode === 'live' ? (
                <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-300 text-[11px] font-mono">
                  <span className="w-1.5 h-1.5 rounded-full bg-indigo-400"></span>
                  Live DFD-FCG Inference
                </span>
              ) : verdict.inference_mode === 'remote' ? (
                <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-blue-500/10 border border-blue-500/20 text-blue-300 text-[11px] font-mono">
                  <span className="w-1.5 h-1.5 rounded-full bg-blue-400"></span>
                  Remote GPU Worker
                </span>
              ) : (
                <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-amber-500/10 border border-amber-500/20 text-amber-300 text-[11px] font-mono">
                  <span className="w-1.5 h-1.5 rounded-full bg-amber-400"></span>
                  Forensic Quality Audit
                </span>
              )}
            </div>
            <h2 className={`text-3xl sm:text-4xl font-extrabold tracking-tight ${verdictColor}`}>
              {verdict.verdict}
            </h2>
            <p className="text-xs text-slate-300 leading-relaxed max-w-md">
              {result.final_explanation}
            </p>
            <div className="text-[11px] font-mono text-slate-500 pt-1">
              Engine status: <span className="text-slate-400">{verdict.checkpoint_status}</span>
            </div>
          </div>

          <div className="lg:col-span-7 grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="p-4 rounded-xl bg-slate-950/80 border border-slate-800/80">
              <span className="text-[11px] font-mono text-slate-500 block mb-1">P(MANIPULATED)</span>
              <span className="text-2xl font-extrabold text-slate-100 font-mono">
                {(verdict.p_fake * 100).toFixed(1)}%
              </span>
              <span className="text-[10px] text-slate-400 block mt-1">Mean over clips</span>
            </div>

            <div className="p-4 rounded-xl bg-slate-950/80 border border-slate-800/80">
              <span className="text-[11px] font-mono text-slate-500 block mb-1">P(AUTHENTIC)</span>
              <span className="text-2xl font-extrabold text-slate-100 font-mono">
                {(verdict.p_real * 100).toFixed(1)}%
              </span>
              <span className="text-[10px] text-slate-400 block mt-1">1 − P(Fake)</span>
            </div>

            <div className="p-4 rounded-xl bg-slate-950/80 border border-slate-800/80">
              <span className="text-[11px] font-mono text-slate-500 block mb-1">THRESHOLDS</span>
              <div className="text-xs font-mono text-slate-300 space-y-0.5 mt-1">
                <div>Real: &lt;{verdict.threshold_authentic.toFixed(2)}</div>
                <div>Fake: &ge;{verdict.threshold_manipulated.toFixed(2)}</div>
              </div>
              <span className="text-[10px] text-slate-500 block mt-1">Middle: Inconclusive</span>
            </div>

            <div className="p-4 rounded-xl bg-slate-950/80 border border-slate-800/80">
              <span className="text-[11px] font-mono text-slate-500 block mb-1">SAMPLING AUDIT</span>
              <span className="text-base font-bold text-slate-200 block mt-1">
                {verdict.n_clips_analyzed} Clips
              </span>
              <span className="text-[10px] text-slate-400 block mt-1">
                {verdict.video_duration_s}s video · {verdict.processing_time_s}s time
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* SECTION B: SPATIAL VS TEMPORAL ANALYSIS */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        <div className="lg:col-span-6 p-6 rounded-2xl bg-slate-900 border border-slate-800">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-base font-bold text-slate-100 flex items-center gap-2">
                <Layers className="w-4 h-4 text-blue-400" />
                <span>Spatial vs Temporal Attribution</span>
              </h3>
              <p className="text-xs text-slate-400">
                Exact Shapley efficiency decomposition (φ_S + φ_T = P_final − 0.5)
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 items-center gap-4 py-4">
            <div className="h-48 relative flex items-center justify-center">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={branchData}
                    cx="50%"
                    cy="50%"
                    innerRadius={55}
                    outerRadius={80}
                    paddingAngle={4}
                    dataKey="value"
                  >
                    {branchData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                </PieChart>
              </ResponsiveContainer>
              <div className="absolute text-center pointer-events-none">
                <span className="text-xs font-mono text-slate-500 block">DOMINANT</span>
                <span className="text-sm font-bold text-slate-200">
                  {result.branch_analysis.dominant_branch}
                </span>
              </div>
            </div>

            <div className="space-y-4">
              <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800">
                <div className="flex items-center justify-between text-xs font-mono mb-1">
                  <span className="text-blue-400 font-bold">Spatial Branch (S)</span>
                  <span className="text-slate-200 font-extrabold">{result.branch_analysis.pct_spatial}%</span>
                </div>
                <p className="text-[11px] text-slate-400 leading-tight">
                  Evaluates facial appearance, synthetic boundary blending, and spatial artifacts.
                </p>
                <div className="mt-2 text-[10px] font-mono text-slate-500">
                  Shapley value (φ_S): {result.branch_analysis.phi_spatial > 0 ? '+' : ''}{result.branch_analysis.phi_spatial.toFixed(4)}
                </div>
              </div>

              <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800">
                <div className="flex items-center justify-between text-xs font-mono mb-1">
                  <span className="text-amber-400 font-bold">Temporal Branch (T)</span>
                  <span className="text-slate-200 font-extrabold">{result.branch_analysis.pct_temporal}%</span>
                </div>
                <p className="text-[11px] text-slate-400 leading-tight">
                  Evaluates inter-frame motion consistency, jitter, and optical temporal coherence.
                </p>
                <div className="mt-2 text-[10px] font-mono text-slate-500">
                  Shapley value (φ_T): {result.branch_analysis.phi_temporal > 0 ? '+' : ''}{result.branch_analysis.phi_temporal.toFixed(4)}
                </div>
              </div>
            </div>
          </div>

          <div className="p-3 rounded-xl bg-blue-950/20 border border-blue-900/30 text-[11px] text-blue-300 flex items-start gap-2">
            <Info className="w-4 h-4 text-blue-400 shrink-0 mt-0.5" />
            <span>
              <b>Note:</b> Branch attribution percentages reflect the relative diagnostic reliance between appearance and motion cues,
              which is distinct from the final prediction probability.
            </span>
          </div>
        </div>

        {/* SECTION C: FACIAL COMPONENT ANALYSIS */}
        <div className="lg:col-span-6 p-6 rounded-2xl bg-slate-900 border border-slate-800 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className="text-base font-bold text-slate-100 flex items-center gap-2">
                  <Eye className="w-4 h-4 text-indigo-400" />
                  <span>Facial Component Analysis</span>
                </h3>
                <p className="text-xs text-slate-400">
                  FCG Query Space decomposition (Eyes, Nose, Lips, Skin)
                </p>
              </div>
              <span className="text-xs font-mono px-2 py-0.5 rounded bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                Dominant: {result.facial_components.dominant_component.toUpperCase()}
              </span>
            </div>

            <div className="grid grid-cols-2 gap-3 mb-4">
              {Object.entries(result.facial_components.components).map(([name, item]) => {
                const isFakeEvidence = item.evidence_direction === 'FAKE';
                return (
                  <div
                    key={name}
                    className="p-3.5 rounded-xl bg-slate-950/70 border border-slate-800 hover:border-slate-700 transition-colors"
                  >
                    <div className="flex items-center justify-between mb-1.5">
                      <span className="text-xs font-bold text-slate-200 capitalize">{item.name}</span>
                      <span
                        className={`text-[10px] px-1.5 py-0.2 rounded font-mono font-bold ${
                          isFakeEvidence
                            ? 'bg-rose-500/10 text-rose-400 border border-rose-500/20'
                            : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                        }`}
                      >
                        {item.evidence_direction}
                      </span>
                    </div>

                    <div className="flex items-baseline justify-between text-xs font-mono mb-1">
                      <span className="text-slate-500">Share:</span>
                      <span className="text-slate-200 font-bold">{item.normalized_share_pct}%</span>
                    </div>

                    <div className="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden mb-2">
                      <div
                        className={`h-1.5 rounded-full ${isFakeEvidence ? 'bg-rose-500' : 'bg-emerald-500'}`}
                        style={{ width: `${item.normalized_share_pct}%` }}
                      />
                    </div>

                    <div className="text-[10px] text-slate-400 leading-tight">
                      Signed logit: <b className="font-mono text-slate-300">{item.signed_contribution > 0 ? '+' : ''}{item.signed_contribution.toFixed(3)}</b>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          <p className="text-xs text-slate-400 leading-relaxed p-3 rounded-xl bg-slate-950/50 border border-slate-800">
            {result.facial_components.interpretation}
          </p>
        </div>
      </div>

      {/* SECTION D: FACIAL REGION HEATMAPS VIEWER */}
      <div className="p-6 sm:p-8 rounded-2xl bg-slate-900 border border-slate-800">
        <div className="flex flex-col md:flex-row md:items-center justify-between pb-6 mb-6 border-b border-slate-800 gap-4">
          <div>
            <h3 className="text-base font-bold text-slate-100 flex items-center gap-2">
              <Sliders className="w-4 h-4 text-cyan-400" />
              <span>Facial Region Evidence & 16×16 Heatmaps</span>
            </h3>
            <p className="text-xs text-slate-400">
              Correlates facial landmarks with actual 16×16 patch-level spatial and temporal attribution values.
            </p>
          </div>

          {/* Heatmap Mode Toggle */}
          <div className="flex items-center gap-2 p-1 rounded-xl bg-slate-950 border border-slate-800">
            <button
              onClick={() => setActiveHeatmapTab('spatial')}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                activeHeatmapTab === 'spatial'
                  ? 'bg-blue-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Spatial Heatmap
            </button>
            <button
              onClick={() => setActiveHeatmapTab('temporal')}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                activeHeatmapTab === 'temporal'
                  ? 'bg-amber-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Temporal Heatmap
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-center">
          {/* Images preview */}
          <div className="lg:col-span-7 grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 text-center">
              <span className="text-[11px] font-mono text-slate-400 block mb-2 font-semibold">
                Extracted Face Crop (150×150 Aligned)
              </span>
              <div className="aspect-square rounded-lg overflow-hidden bg-black max-w-[240px] mx-auto border border-slate-800">
                {result.facial_regions.face_crop_url ? (
                  <img
                    src={result.facial_regions.face_crop_url}
                    alt="Aligned Face"
                    className="w-full h-full object-cover"
                  />
                ) : (
                  <div className="w-full h-full flex items-center justify-center text-slate-600 text-xs font-mono">
                    Face Crop
                  </div>
                )}
              </div>
              <span className="text-[10px] text-slate-500 mt-2 block font-mono">
                Canonical 68-landmark geometry
              </span>
            </div>

            <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 text-center">
              <span className="text-[11px] font-mono text-slate-400 block mb-2 font-semibold">
                {activeHeatmapTab === 'spatial' ? 'Spatial 16×16 Heatmap Overlay' : 'Temporal 16×16 Heatmap Overlay'}
              </span>
              <div className="aspect-square rounded-lg overflow-hidden bg-black max-w-[240px] mx-auto border border-slate-800">
                {activeHeatmapTab === 'spatial' ? (
                  result.facial_regions.heatmap_spatial_url ? (
                    <img
                      src={result.facial_regions.heatmap_spatial_url}
                      alt="Spatial Heatmap"
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center text-slate-600 text-xs font-mono">
                      Spatial Heatmap
                    </div>
                  )
                ) : (
                  result.facial_regions.heatmap_temporal_url ? (
                    <img
                      src={result.facial_regions.heatmap_temporal_url}
                      alt="Temporal Heatmap"
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center text-slate-600 text-xs font-mono">
                      Temporal Heatmap
                    </div>
                  )
                )}
              </div>
              <span className="text-[10px] text-slate-500 mt-2 block font-mono">
                Red: Manipulated Evidence · Blue: Authentic Evidence
              </span>
            </div>
          </div>

          {/* Regional share breakdown table */}
          <div className="lg:col-span-5 space-y-3">
            <h4 className="text-xs font-mono font-semibold text-slate-400 uppercase tracking-wider mb-2">
              Anatomical Region Decomposition
            </h4>

            <div className="space-y-2">
              {Object.entries(result.facial_regions.regions).map(([rName, reg]) => (
                <div
                  key={rName}
                  className="p-3 rounded-xl bg-slate-950/80 border border-slate-800 flex items-center justify-between text-xs"
                >
                  <div>
                    <span className="font-bold text-slate-200 capitalize block">{rName}</span>
                    <span className="text-[10px] font-mono text-slate-500">
                      {reg.patch_count} patches · Density: {reg.density > 0 ? '+' : ''}{reg.density.toFixed(4)}
                    </span>
                  </div>

                  <div className="text-right font-mono">
                    <span className="font-bold text-slate-200 block">
                      {activeHeatmapTab === 'spatial' 
                        ? `${(reg.spatial_share_pct ?? reg.share_pct ?? 0).toFixed(1)}%` 
                        : `${(reg.temporal_share_pct ?? reg.share_pct ?? 0).toFixed(1)}%`}
                    </span>
                    <span className="text-[10px] text-slate-500">
                      Signed: {(() => {
                        const sVal = activeHeatmapTab === 'spatial' 
                          ? (reg.spatial_signed ?? reg.signed ?? 0)
                          : (reg.temporal_signed ?? reg.signed ?? 0);
                        return sVal > 0 ? `+${sVal.toFixed(2)}` : sVal.toFixed(2);
                      })()}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* SECTION E: TEMPORAL TIMELINE ANALYSIS */}
      <div className="p-6 sm:p-8 rounded-2xl bg-slate-900 border border-slate-800">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between mb-6 gap-2">
          <div>
            <h3 className="text-base font-bold text-slate-100 flex items-center gap-2">
              <Clock className="w-4 h-4 text-amber-400" />
              <span>Temporal Clip Timeline (Overlapping 3-Second Windows)</span>
            </h3>
            <p className="text-xs text-slate-400">
              Evaluates frame sequences with 0.333s stride to assess motion continuity across the video.
            </p>
          </div>
          <span className="text-xs font-mono text-slate-500">
            Click any clip to inspect individual evidence
          </span>
        </div>

        {/* Clip Chart */}
        <div className="h-56 w-full mb-6">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={clipChartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="time" stroke="#64748b" fontSize={11} />
              <YAxis domain={[0, 100]} stroke="#64748b" fontSize={11} unit="%" />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#0f172a',
                  borderColor: '#334155',
                  borderRadius: '0.75rem',
                  fontSize: '11px',
                  fontFamily: 'monospace',
                }}
              />
              <Line
                type="monotone"
                dataKey="p_fake"
                name="P(Fake) %"
                stroke="#f43f5e"
                strokeWidth={2.5}
                dot={{ r: 3, fill: '#f43f5e' }}
                activeDot={{ r: 5 }}
              />
              <Line
                type="monotone"
                dataKey="spatial"
                name="Spatial Share %"
                stroke="#3b82f6"
                strokeWidth={1.5}
                strokeDasharray="4 4"
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="temporal"
                name="Temporal Share %"
                stroke="#f59e0b"
                strokeWidth={1.5}
                strokeDasharray="4 4"
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Selected Clip Inspector */}
        {activeClip ? (
          <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 grid grid-cols-2 sm:grid-cols-5 gap-3 text-xs font-mono">
            <div>
              <span className="text-slate-500 block text-[10px]">SELECTED CLIP</span>
              <span className="text-slate-200 font-bold">Clip {activeClip.clip_index + 1} of {result.clips.length}</span>
            </div>
            <div>
              <span className="text-slate-500 block text-[10px]">TIMESTAMP RANGE</span>
              <span className="text-slate-200 font-bold">{activeClip.start_timestamp_s}s → {activeClip.end_timestamp_s}s</span>
            </div>
            <div>
              <span className="text-slate-500 block text-[10px]">CLIP VERDICT</span>
              <span className={`font-bold ${activeClip.verdict === 'FAKE' ? 'text-rose-400' : 'text-emerald-400'}`}>
                {activeClip.verdict} ({(activeClip.p_final * 100).toFixed(1)}%)
              </span>
            </div>
            <div>
              <span className="text-slate-500 block text-[10px]">SPATIAL SHARE</span>
              <span className="text-blue-400 font-bold">{activeClip.pct_spatial}%</span>
            </div>
            <div>
              <span className="text-slate-500 block text-[10px]">TEMPORAL SHARE</span>
              <span className="text-amber-400 font-bold">{activeClip.pct_temporal}%</span>
            </div>
          </div>
        ) : null}

        {/* Clip Slider Buttons */}
        <div className="flex items-center justify-between mt-4">
          <button
            onClick={() => setSelectedClipIndex((prev) => Math.max(0, prev - 1))}
            disabled={selectedClipIndex === 0}
            className="p-1.5 rounded-lg bg-slate-800 text-slate-300 disabled:opacity-40 hover:bg-slate-700 text-xs flex items-center gap-1"
          >
            <ChevronLeft className="w-4 h-4" />
            <span>Previous Clip</span>
          </button>
          <span className="text-xs font-mono text-slate-400">
            Clip {selectedClipIndex + 1} / {result.clips.length}
          </span>
          <button
            onClick={() => setSelectedClipIndex((prev) => Math.min(result.clips.length - 1, prev + 1))}
            disabled={selectedClipIndex === result.clips.length - 1}
            className="p-1.5 rounded-lg bg-slate-800 text-slate-300 disabled:opacity-40 hover:bg-slate-700 text-xs flex items-center gap-1"
          >
            <span>Next Clip</span>
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* SECTION F & G: FRAME INSPECTION & VIDEO QUALITY AUDIT */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Frame Inspection */}
        <div className="lg:col-span-6 p-6 rounded-2xl bg-slate-900 border border-slate-800 flex flex-col justify-between">
          {activeFrame ? (
            <div>
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-base font-bold text-slate-100 flex items-center gap-2">
                  <Maximize2 className="w-4 h-4 text-blue-400" />
                  <span>Frame Inspection Panel</span>
                </h3>
                <span className="text-xs font-mono text-slate-400">
                  Frame {activeFrame.frame_index} ({activeFrame.timestamp_s}s)
                </span>
              </div>

              <div className="flex items-center gap-4 mb-4">
                <div className="w-24 h-24 rounded-xl overflow-hidden bg-black border border-slate-800 shrink-0">
                  {activeFrame.face_crop_url ? (
                    <img
                      src={activeFrame.face_crop_url}
                      alt="Frame face"
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center text-slate-600 text-xs font-mono">
                      Frame Crop
                    </div>
                  )}
                </div>

                <div className="space-y-1 text-xs font-mono">
                  <div>
                    <span className="text-slate-500">Tracked Face: </span>
                    <span className="text-emerald-400 font-bold">Yes (68 Landmarks)</span>
                  </div>
                  <div>
                    <span className="text-slate-500">Local P(Fake): </span>
                    <span className="text-slate-200 font-bold">{(activeFrame.p_fake * 100).toFixed(1)}%</span>
                  </div>
                  <div>
                    <span className="text-slate-500">Dominant Component: </span>
                    <span className="text-indigo-400 font-bold capitalize">{activeFrame.dominant_part}</span>
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-4 gap-2 text-[11px] font-mono text-center">
                {Object.entries(activeFrame.components || {}).map(([k, val]) => (
                  <div key={k} className="p-2 rounded-lg bg-slate-950 border border-slate-800">
                    <span className="text-slate-500 block capitalize">{k}</span>
                    <span className="font-bold text-slate-200">{val > 0 ? `+${val.toFixed(2)}` : val.toFixed(2)}</span>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="text-slate-500 text-xs font-mono py-8 text-center">No frame tracking samples available for this video.</div>
          )}

          <div className="flex items-center justify-between pt-4 mt-4 border-t border-slate-800">
            <button
              onClick={() => setCurrentFrameIdx((prev) => Math.max(0, prev - 1))}
              disabled={currentFrameIdx === 0}
              className="p-1 rounded-lg bg-slate-800 text-slate-300 disabled:opacity-40 text-xs flex items-center gap-1"
            >
              <ChevronLeft className="w-3.5 h-3.5" />
              <span>Prev Frame</span>
            </button>
            <span className="text-xs font-mono text-slate-500">
              {currentFrameIdx + 1} / {result.frames.length} sampled frames
            </span>
            <button
              onClick={() => setCurrentFrameIdx((prev) => Math.min(result.frames.length - 1, prev + 1))}
              disabled={currentFrameIdx === result.frames.length - 1}
              className="p-1 rounded-lg bg-slate-800 text-slate-300 disabled:opacity-40 text-xs flex items-center gap-1"
            >
              <span>Next Frame</span>
              <ChevronRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>

        {/* Video Quality Report */}
        <div className="lg:col-span-6 p-6 rounded-2xl bg-slate-900 border border-slate-800 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-base font-bold text-slate-100 flex items-center gap-2">
                <ShieldCheck className="w-4 h-4 text-emerald-400" />
                <span>Video Quality & Forensic Caveats</span>
              </h3>
              <span className="text-xs font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-400">
                Independent Quality Audit
              </span>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 text-xs font-mono mb-4">
              <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-800">
                <span className="text-slate-500 block text-[10px]">RESOLUTION</span>
                <span className="text-slate-200 font-bold">{result.quality_report.resolution}</span>
              </div>
              <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-800">
                <span className="text-slate-500 block text-[10px]">FPS / DURATION</span>
                <span className="text-slate-200 font-bold">{result.quality_report.fps} fps · {result.quality_report.duration_s}s</span>
              </div>
              <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-800">
                <span className="text-slate-500 block text-[10px]">SHARPNESS</span>
                <span className="text-slate-200 font-bold">{result.quality_report.blur_assessment}</span>
              </div>
              <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-800">
                <span className="text-slate-500 block text-[10px]">COMPRESSION</span>
                <span className="text-slate-200 font-bold">{result.quality_report.compression_level}</span>
              </div>
            </div>

            <div className="space-y-1.5 text-xs text-slate-300">
              {result.quality_report.quality_notes.map((note, idx) => (
                <div key={idx} className="flex items-start gap-2 p-2 rounded-lg bg-slate-950/60 border border-slate-800/80">
                  <span className="text-blue-400 text-sm leading-none">•</span>
                  <span className="text-[11px] leading-relaxed">{note}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="p-3 rounded-xl bg-amber-950/20 border border-amber-900/30 text-[11px] text-amber-300 flex items-start gap-2 mt-4">
            <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
            <span>
              Low video sharpness and compression artifacts can degrade motion features. The forensic pipeline keeps quality metrics
              strictly isolated from manipulation probability.
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};
