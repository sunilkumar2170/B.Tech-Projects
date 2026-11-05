import React, { useState, useRef, useEffect } from 'react';
import {
  UploadCloud,
  FileVideo,
  X,
  AlertCircle,
  Play,
  RotateCcw,
  Sparkles,
  ArrowRight,
  Activity,
  CheckCircle2,
  Clock,
  Film
} from 'lucide-react';
import { ForensicLogo } from '../components/ui/ForensicLogo';
import { HeroMeshGraphic } from '../components/illustrations/HeroMeshGraphic';
import { PipelineDiagramGraphic } from '../components/illustrations/PipelineDiagramGraphic';
import { Results } from './Results';
import { api } from '../services/api';
import type { AnalysisStatus, AnalysisResult, ProcessingStage, HealthResponse } from '../types';

type FlowState = 'upload' | 'preview' | 'processing' | 'results';

export const Home: React.FC = () => {
  const [state, setState] = useState<FlowState>('upload');
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isSample, setIsSample] = useState<boolean>(false);
  const [sampleKey, setSampleKey] = useState<string | null>(null);
  const [videoPreviewUrl, setVideoPreviewUrl] = useState<string | null>(null);
  const [videoMeta, setVideoMeta] = useState<{
    name: string;
    sizeFormatted: string;
    format: string;
    duration: number;
    width: number;
    height: number;
  } | null>(null);

  // Analysis & Processing state
  const [analysisId, setAnalysisId] = useState<string | null>(null);
  const [statusData, setStatusData] = useState<AnalysisStatus | null>(null);
  const [elapsedSeconds, setElapsedSeconds] = useState<number>(0);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [healthInfo, setHealthInfo] = useState<HealthResponse | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    api.getHealth()
      .then((h) => setHealthInfo(h))
      .catch((e) => console.log('Notice: health check connection:', e));
  }, []);

  // Clean up object URL on unmount or file change
  useEffect(() => {
    return () => {
      if (videoPreviewUrl && !isSample) {
        URL.revokeObjectURL(videoPreviewUrl);
      }
    };
  }, [videoPreviewUrl, isSample]);

  // Handle file validation and preview loading
  const processSelectedFile = (file: File) => {
    setErrorMessage(null);

    // Validate extension
    const ext = '.' + file.name.split('.').pop()?.toLowerCase();
    const validExts = ['.mp4', '.mov', '.avi', '.mkv'];
    if (!validExts.includes(ext)) {
      setErrorMessage(`Unsupported format '${ext}'. Supported formats: MP4, MOV, AVI, MKV.`);
      return;
    }

    // Validate size (max 200 MB)
    const maxSize = 200 * 1024 * 1024;
    if (file.size > maxSize) {
      setErrorMessage('File size exceeds the 200 MB limit.');
      return;
    }

    if (videoPreviewUrl && !isSample) {
      URL.revokeObjectURL(videoPreviewUrl);
    }

    const url = URL.createObjectURL(file);
    setSelectedFile(file);
    setIsSample(false);
    setSampleKey(null);
    setVideoPreviewUrl(url);

    // Read metadata via temporary HTML5 video
    const tempVideo = document.createElement('video');
    tempVideo.preload = 'metadata';
    tempVideo.src = url;
    tempVideo.onloadedmetadata = () => {
      setVideoMeta({
        name: file.name,
        sizeFormatted: `${(file.size / (1024 * 1024)).toFixed(1)} MB`,
        format: file.name.split('.').pop()?.toUpperCase() || 'MP4',
        duration: Math.round(tempVideo.duration) || 0,
        width: tempVideo.videoWidth || 0,
        height: tempVideo.videoHeight || 0
      });
      setState('preview');
    };
    tempVideo.onerror = () => {
      setVideoMeta({
        name: file.name,
        sizeFormatted: `${(file.size / (1024 * 1024)).toFixed(1)} MB`,
        format: file.name.split('.').pop()?.toUpperCase() || 'MP4',
        duration: 0,
        width: 0,
        height: 0
      });
      setState('preview');
    };
  };

  // Drag and drop handlers
  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processSelectedFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      processSelectedFile(e.target.files[0]);
    }
  };

  // Quick benchmark sample loader
  const handleLoadSample = (sample: '000_003' | '000') => {
    setErrorMessage(null);
    if (videoPreviewUrl && !isSample) {
      URL.revokeObjectURL(videoPreviewUrl);
    }
    const sampleUrl = `/samples/${sample}.mp4`;
    setSelectedFile(null);
    setIsSample(true);
    setSampleKey(sample);
    setVideoPreviewUrl(sampleUrl);

    setVideoMeta({
      name: `faceforensics_${sample}.mp4 (Benchmark Sample)`,
      sizeFormatted: '12.4 MB',
      format: 'MP4',
      duration: 10,
      width: 720,
      height: 720
    });
    setState('preview');
  };

  const handleRemoveVideo = () => {
    if (videoPreviewUrl && !isSample) {
      URL.revokeObjectURL(videoPreviewUrl);
    }
    setSelectedFile(null);
    setIsSample(false);
    setSampleKey(null);
    setVideoPreviewUrl(null);
    setVideoMeta(null);
    setErrorMessage(null);
    setState('upload');
  };

  const handleChangeVideo = () => {
    fileInputRef.current?.click();
  };

  // Start analysis job on backend
  const handleStartAnalysis = async () => {
    setErrorMessage(null);
    setState('processing');
    setElapsedSeconds(0);
    setStatusData(null);

    try {
      let createdAnalysisId: string;

      if (isSample && sampleKey) {
        const sampleResp = await api.selectSample(sampleKey);
        const startResp = await api.startAnalysis(sampleResp.video.video_id, sampleKey);
        createdAnalysisId = startResp.analysis_id;
      } else if (selectedFile) {
        // Upload video first
        const uploadResp = await api.uploadVideo(selectedFile);
        const startResp = await api.startAnalysis(uploadResp.video.video_id);
        createdAnalysisId = startResp.analysis_id;
      } else {
        throw new Error('No video selected.');
      }

      setAnalysisId(createdAnalysisId);
      pollJobStatus(createdAnalysisId);
    } catch (err: any) {
      console.error('Failed to initiate analysis:', err);
      setErrorMessage(err.response?.data?.detail || err.message || 'Failed to start video analysis.');
      setState('preview');
    }
  };

  // Poll job status until complete or failed
  const pollJobStatus = (id: string) => {
    const startTime = Date.now();
    const timerInterval = setInterval(() => {
      setElapsedSeconds(Math.floor((Date.now() - startTime) / 1000));
    }, 1000);

    const statusInterval = setInterval(async () => {
      try {
        const status = await api.getAnalysisStatus(id);
        setStatusData(status);

        if (status.status === 'completed') {
          clearInterval(statusInterval);
          clearInterval(timerInterval);
          // Fetch final result
          const result = await api.getAnalysisResult(id);
          setAnalysisResult(result);
          setState('results');
        } else if (status.status === 'failed') {
          clearInterval(statusInterval);
          clearInterval(timerInterval);
          setErrorMessage(status.error_message || 'Video analysis encountered a processing error.');
        }
      } catch (err: any) {
        console.error('Status poll error:', err);
      }
    }, 1200);
  };

  const handleResetToUpload = () => {
    handleRemoveVideo();
    setAnalysisId(null);
    setStatusData(null);
    setAnalysisResult(null);
    setErrorMessage(null);
    setState('upload');
  };

  const formatSeconds = (sec: number) => {
    const m = Math.floor(sec / 60);
    const s = sec % 60;
    return `${m}:${s < 10 ? '0' : ''}${s}`;
  };

  // Stages definition with backend fallbacks
  const pipelineStages: ProcessingStage[] = statusData?.stages || [
    { id: 'video_validation', name: 'Video validation & integrity check', status: 'completed', progress: 100 },
    { id: 'frame_extraction', name: 'Frame extraction & temporal indexing', status: 'in_progress', progress: 30 },
    { id: 'face_detection', name: 'Face detection & 68 landmark tracking', status: 'pending', progress: 0 },
    { id: 'quality_assessment', name: 'Video quality & blur variance assessment', status: 'pending', progress: 0 },
    { id: 'dfd_fcg_inference', name: 'DFD-FCG spatial-temporal neural inference', status: 'pending', progress: 0 },
    { id: 'region_attribution', name: '16×16 patch & region attribution', status: 'pending', progress: 0 },
    { id: 'result_aggregation', name: 'Forensic evidence synthesis & PDF report', status: 'pending', progress: 0 },
  ];

  // ----------------------------------------------------
  // STATE 4: Forensic Results Dashboard
  // ----------------------------------------------------
  if (state === 'results' && analysisResult) {
    return (
      <Results
        analysisId={analysisResult.analysis_id}
        initialResult={analysisResult}
        onReset={handleResetToUpload}
      />
    );
  }

  return (
    <div className="min-h-[calc(100vh-4rem)] flex flex-col justify-center items-center py-12 px-4 sm:px-6 lg:px-8">
      {/* Hidden file input */}
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileInputChange}
        accept=".mp4,.mov,.avi,.mkv"
        className="hidden"
      />

      <div className={`w-full ${state === 'upload' ? 'max-w-4xl' : 'max-w-2xl'}`}>
        {/* Error notification */}
        {errorMessage && (
          <div className="mb-6 p-4 rounded-xl bg-rose-950/40 border border-rose-800/60 text-rose-300 text-xs flex items-start gap-3 shadow-lg">
            <AlertCircle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
            <div className="flex-1">
              <p className="font-semibold text-rose-200">Error</p>
              <p className="mt-0.5">{errorMessage}</p>
            </div>
            <button
              onClick={() => setErrorMessage(null)}
              className="text-rose-400 hover:text-rose-200"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        )}

        {/* ----------------------------------------------------
            STATE 1: Minimal Upload Dropzone
        ---------------------------------------------------- */}
        {state === 'upload' && (
          <div className="flex flex-col items-center text-center">
            {/* System Status Pill */}
            {healthInfo && (
              <div className="mb-4">
                {healthInfo.live_inference_available ? (
                  <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-xs font-mono font-medium shadow-sm">
                    <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                    DFD-FCG Active Checkpoint ({healthInfo.device?.toUpperCase() || 'CPU'})
                  </span>
                ) : (
                  <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-300 text-xs font-mono font-medium shadow-sm">
                    <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
                    Research Benchmark Mode · FaceForensics++ Audited
                  </span>
                )}
              </div>
            )}

            {/* Custom SVG Hero Illustration */}
            <div className="mb-6 w-full max-w-lg">
              <HeroMeshGraphic />
            </div>

            {/* Title */}
            <h1 className="text-3xl sm:text-4xl font-extrabold text-slate-100 tracking-tight mb-2">
              Video Deepfake Detection
            </h1>
            <p className="text-xs sm:text-sm text-slate-400 font-mono mb-8 max-w-md">
              DFD-FCG Explainable AI · Facial Component &amp; Temporal Attribution
            </p>

            {/* Central Dropzone */}
            <div
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className={`w-full max-w-2xl p-8 sm:p-12 rounded-2xl border-2 border-dashed transition-all cursor-pointer flex flex-col items-center justify-center text-center group ${
                isDragging
                  ? 'border-cyan-400 bg-cyan-950/20 scale-[1.01]'
                  : 'border-slate-800 hover:border-slate-700 bg-slate-900/60 hover:bg-slate-900'
              }`}
            >
              <div className="w-14 h-14 rounded-2xl bg-cyan-500/10 border border-cyan-500/20 text-cyan-400 flex items-center justify-center mb-4 group-hover:scale-105 transition-transform">
                <UploadCloud className="w-7 h-7" />
              </div>

              <p className="text-base font-semibold text-slate-200 mb-1">
                Drop your video here
              </p>
              <p className="text-xs text-slate-400 mb-3">
                or browse files
              </p>
              <p className="text-[11px] font-mono text-slate-500 mb-6">
                MP4, MOV, AVI, MKV · Maximum 200 MB
              </p>

              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  fileInputRef.current?.click();
                }}
                className="px-6 py-2.5 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-white font-medium text-xs shadow-md transition-colors flex items-center gap-2"
              >
                <FileVideo className="w-4 h-4" />
                <span>Select Video</span>
              </button>
            </div>

            {/* Subtle Benchmark Sample Quick-Loaders */}
            <div className="mt-6 flex flex-wrap items-center justify-center gap-3 text-xs text-slate-500">
              <span>Or test verified benchmark:</span>
              <button
                type="button"
                onClick={() => handleLoadSample('000_003')}
                className="px-2.5 py-1 rounded-lg bg-slate-900 border border-slate-800 text-slate-400 hover:text-cyan-400 hover:border-cyan-500/30 transition-colors font-mono text-[11px]"
              >
                Manipulated Sample (000_003)
              </button>
              <button
                type="button"
                onClick={() => handleLoadSample('000')}
                className="px-2.5 py-1 rounded-lg bg-slate-900 border border-slate-800 text-slate-400 hover:text-cyan-400 hover:border-cyan-500/30 transition-colors font-mono text-[11px]"
              >
                Authentic Sample (000)
              </button>
            </div>

            {/* Pipeline Architecture Graphic */}
            <div className="w-full mt-12 pt-8 border-t border-slate-800/80">
              <h3 className="text-xs font-mono uppercase tracking-wider text-slate-400 text-center mb-2">
                Neural Explainability Pipeline
              </h3>
              <p className="text-xs text-slate-500 text-center max-w-lg mx-auto mb-4">
                68-point facial landmark tracking to exact additive LayerNorm-Linear Shapley attribution
              </p>
              <PipelineDiagramGraphic />
            </div>
          </div>
        )}

        {/* ----------------------------------------------------
            STATE 2: Same-Page Video Preview & Verification
        ---------------------------------------------------- */}
        {state === 'preview' && videoMeta && videoPreviewUrl && (
          <div className="w-full bg-slate-900 border border-slate-800 rounded-2xl p-6 sm:p-8 shadow-2xl">
            {/* Header */}
            <div className="flex items-center justify-between pb-4 border-b border-slate-800 mb-6">
              <div className="flex items-center gap-2.5">
                <div className="w-7 h-7 rounded-lg bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center text-cyan-400">
                  <Film className="w-4 h-4" />
                </div>
                <div>
                  <h2 className="text-sm font-semibold text-slate-200">Video Preview & Metadata</h2>
                  <p className="text-[11px] text-slate-500 font-mono">Verify media integrity before neural execution</p>
                </div>
              </div>
              <button
                onClick={handleRemoveVideo}
                className="p-1.5 rounded-lg text-slate-500 hover:text-rose-400 hover:bg-slate-800 transition-colors"
                title="Remove video"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            {/* Actual Video Player */}
            <div className="w-full rounded-xl overflow-hidden bg-black border border-slate-800 mb-6 flex justify-center items-center max-h-[360px]">
              <video
                src={videoPreviewUrl}
                controls
                playsInline
                className="w-full max-h-[360px] object-contain"
              />
            </div>

            {/* Video Metadata Grid */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-6 text-xs">
              <div className="p-3 rounded-xl bg-slate-950 border border-slate-800/80">
                <span className="text-[10px] text-slate-500 uppercase tracking-wider block mb-1">File Name</span>
                <span className="font-mono text-slate-200 font-medium truncate block" title={videoMeta.name}>
                  {videoMeta.name}
                </span>
              </div>
              <div className="p-3 rounded-xl bg-slate-950 border border-slate-800/80">
                <span className="text-[10px] text-slate-500 uppercase tracking-wider block mb-1">File Size</span>
                <span className="font-mono text-slate-200 font-medium">
                  {videoMeta.sizeFormatted}
                </span>
              </div>
              <div className="p-3 rounded-xl bg-slate-950 border border-slate-800/80">
                <span className="text-[10px] text-slate-500 uppercase tracking-wider block mb-1">Duration</span>
                <span className="font-mono text-slate-200 font-medium">
                  {videoMeta.duration > 0 ? formatSeconds(videoMeta.duration) : 'N/A'}
                </span>
              </div>
              <div className="p-3 rounded-xl bg-slate-950 border border-slate-800/80">
                <span className="text-[10px] text-slate-500 uppercase tracking-wider block mb-1">Resolution</span>
                <span className="font-mono text-slate-200 font-medium">
                  {videoMeta.width > 0 ? `${videoMeta.width} × ${videoMeta.height}` : 'Variable'}
                </span>
              </div>
            </div>

            {/* Action Buttons */}
            <div className="flex flex-col sm:flex-row items-center justify-between gap-3 pt-2">
              <div className="flex items-center gap-2 w-full sm:w-auto">
                <button
                  type="button"
                  onClick={handleChangeVideo}
                  className="px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-medium transition-colors w-full sm:w-auto"
                >
                  Change Video
                </button>
                <button
                  type="button"
                  onClick={handleRemoveVideo}
                  className="px-4 py-2 rounded-xl bg-transparent hover:bg-slate-800 text-slate-400 hover:text-rose-400 text-xs font-medium transition-colors w-full sm:w-auto"
                >
                  Remove Video
                </button>
              </div>

              <button
                type="button"
                onClick={handleStartAnalysis}
                className="px-6 py-2.5 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-white font-semibold text-xs shadow-lg transition-all flex items-center justify-center gap-2 w-full sm:w-auto"
              >
                <span>Analyze Video</span>
                <ArrowRight className="w-4 h-4" />
              </button>
            </div>
          </div>
        )}

        {/* ----------------------------------------------------
            STATE 3: Compact Processing Progress Panel
        ---------------------------------------------------- */}
        {state === 'processing' && (
          <div className="w-full bg-slate-900 border border-slate-800 rounded-2xl p-6 sm:p-8 shadow-2xl">
            {/* Header */}
            <div className="flex items-center justify-between pb-4 border-b border-slate-800 mb-6">
              <div className="flex items-center gap-3">
                <div className="w-9 h-9 rounded-xl bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center text-cyan-400">
                  <Activity className="w-5 h-5 animate-pulse" />
                </div>
                <div>
                  <h2 className="text-base font-bold text-slate-100">
                    Executing Forensic Analysis
                  </h2>
                  <p className="text-xs text-slate-400 font-mono">
                    Elapsed: {elapsedSeconds}s · Job: {analysisId ? analysisId.slice(0, 8) : '...'}
                  </p>
                </div>
              </div>

              <div className="text-right font-mono">
                <span className="text-lg font-bold text-cyan-400">
                  {statusData?.progress || 12}%
                </span>
              </div>
            </div>

            {/* Overall Progress Bar */}
            <div className="w-full h-2 rounded-full bg-slate-950 overflow-hidden mb-6 border border-slate-800">
              <div
                className="h-full bg-gradient-to-r from-cyan-500 to-blue-600 transition-all duration-300"
                style={{ width: `${Math.max(statusData?.progress || 12, 5)}%` }}
              />
            </div>

            {/* 7 Forensic Stages Checklist */}
            <div className="space-y-3 mb-6">
              {pipelineStages.map((st, idx) => {
                const isDone = st.status === 'completed';
                const isInProg = st.status === 'in_progress';

                return (
                  <div
                    key={st.id}
                    className={`flex items-center justify-between p-3 rounded-xl border text-xs transition-colors ${
                      isDone
                        ? 'bg-slate-950/60 border-emerald-950/40 text-slate-300'
                        : isInProg
                        ? 'bg-cyan-950/20 border-cyan-800/50 text-cyan-200'
                        : 'bg-slate-950/30 border-slate-800/40 text-slate-500'
                    }`}
                  >
                    <div className="flex items-center gap-3">
                      <span className="font-mono text-[10px] text-slate-500 w-4">
                        0{idx + 1}
                      </span>
                      {isDone ? (
                        <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
                      ) : isInProg ? (
                        <div className="w-4 h-4 rounded-full border-2 border-cyan-400 border-t-transparent animate-spin shrink-0" />
                      ) : (
                        <div className="w-4 h-4 rounded-full border border-slate-700 shrink-0" />
                      )}
                      <span className="font-medium">{st.name}</span>
                    </div>

                    <div className="font-mono text-[11px]">
                      {isDone ? (
                        <span className="text-emerald-400">Complete</span>
                      ) : isInProg ? (
                        <span className="text-cyan-400 animate-pulse">Running</span>
                      ) : (
                        <span className="text-slate-600">Pending</span>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Abort / Retry Options */}
            {statusData?.status === 'failed' ? (
              <div className="pt-4 border-t border-slate-800 flex items-center justify-between">
                <span className="text-xs text-rose-400 font-medium">Pipeline halted due to error.</span>
                <button
                  onClick={handleResetToUpload}
                  className="px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold flex items-center gap-2"
                >
                  <RotateCcw className="w-3.5 h-3.5" />
                  <span>Retry Another Video</span>
                </button>
              </div>
            ) : (
              <p className="text-[11px] text-slate-500 text-center font-mono">
                Processing video clips through DFD-FCG spatial-temporal decomposition...
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
