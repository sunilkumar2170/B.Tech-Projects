export interface VideoMetadata {
  video_id: string;
  original_filename: string;
  stored_filename: string;
  file_size_bytes: number;
  duration_seconds: number;
  fps: number;
  total_frames: number;
  width: number;
  height: number;
  mime_type: string;
  is_sample: boolean;
  sample_key?: string;
  created_at: string;
}

export interface ProcessingStage {
  id: string;
  name: string;
  status: 'pending' | 'in_progress' | 'completed' | 'failed';
  progress: number;
  message?: string;
}

export interface AnalysisStatus {
  analysis_id: string;
  video_id: string;
  status: 'queued' | 'processing' | 'completed' | 'failed';
  current_stage: string;
  progress: number;
  stages: ProcessingStage[];
  elapsed_seconds: number;
  error_message?: string;
  created_at: string;
  completed_at?: string;
}

export interface OverallVerdict {
  verdict: 'LIKELY AUTHENTIC' | 'LIKELY MANIPULATED' | 'INCONCLUSIVE';
  p_fake: number;
  p_real: number;
  threshold_authentic: number;
  threshold_manipulated: number;
  n_clips_analyzed: number;
  video_duration_s: number;
  processing_time_s: number;
  model_architecture: string;
  checkpoint_status: string;
  inference_mode?: 'benchmark' | 'live' | 'remote' | 'fallback';
}

export interface HealthResponse {
  status: string;
  service: string;
  tagline: string;
  version: string;
  live_inference_available: boolean;
  python_version: string;
  torch_version: string;
  opencv_version: string;
  cuda_available: boolean;
  device: string;
  remote_gpu_worker: {
    configured: boolean;
    endpoint: string | null;
  };
  checkpoint_status: {
    checkpoint_present: boolean;
    weights_path: string;
    model_loaded: boolean;
    message: string;
  };
  thresholds: {
    authentic: number;
    manipulated: number;
  };
  database?: {
    connected: boolean;
    provider?: string;
    database?: string;
    latency_ms?: number;
    version?: string;
    total_saved_analyses?: number;
    pooling?: string;
    error?: string;
  };
}

export interface BranchShapley {
  pct_spatial: number;
  pct_temporal: number;
  phi_spatial: number;
  phi_temporal: number;
  dominant_branch: string;
  explanation: string;
}

export interface FacialComponentDetail {
  name: string;
  signed_contribution: number;
  normalized_share_pct: number;
  evidence_direction: 'FAKE' | 'REAL';
  relative_importance: 'HIGH' | 'MEDIUM' | 'LOW';
  description: string;
}

export interface FacialComponentsReport {
  components: Record<string, FacialComponentDetail>;
  dominant_component: string;
  interpretation: string;
}

export interface RegionDetail {
  name: string;
  patch_count: number;
  signed?: number;
  share_pct?: number;
  spatial_signed?: number;
  spatial_share_pct?: number;
  temporal_signed?: number;
  temporal_share_pct?: number;
  density: number;
}

export interface FacialRegionsReport {
  regions: Record<string, RegionDetail>;
  dominant_spatial_region: string;
  dominant_temporal_region: string;
  grid_size: number;
  patch_spatial_attributions?: number[][];
  patch_temporal_attributions?: number[][];
  face_crop_url?: string;
  heatmap_spatial_url?: string;
  heatmap_temporal_url?: string;
}

export interface ClipDetail {
  clip_index: number;
  start_frame: number;
  end_frame: number;
  start_timestamp_s: number;
  end_timestamp_s: number;
  p_final: number;
  p_final_raw?: number;
  p_real: number;
  verdict: string;
  pct_spatial: number;
  pct_temporal: number;
  phi_spatial: number;
  phi_temporal: number;
  phi_spatial_raw?: number;
  phi_temporal_raw?: number;
  contrib_parts?: Record<string, number>;
  p_spatial?: number;
  p_temporal?: number;
  p_fusion?: number;
  p_spatial_raw?: number;
  p_temporal_raw?: number;
  p_fusion_raw?: number;
}

export interface FrameInspectionDetail {
  frame_index: number;
  timestamp_s: number;
  face_detected: boolean;
  face_crop_url?: string;
  landmark_count: number;
  p_fake: number;
  dominant_part: string;
  components: Record<string, number>;
}

export interface VideoQualityReport {
  resolution: string;
  fps: number;
  duration_s: number;
  total_frames: number;
  blur_score: number;
  blur_assessment: string;
  compression_warning: boolean;
  compression_level: string;
  face_detection_consistency: number;
  lighting_warning: boolean;
  lighting_assessment: string;
  quality_notes: string[];
}

export interface AnalysisResult {
  analysis_id: string;
  video_id: string;
  filename: string;
  created_at: string;
  verdict: OverallVerdict;
  branch_analysis: BranchShapley;
  facial_components: FacialComponentsReport;
  facial_regions: FacialRegionsReport;
  clips: ClipDetail[];
  frames: FrameInspectionDetail[];
  quality_report: VideoQualityReport;
  final_explanation: string;
  research_citations: string[];
  report_pdf_url: string;
}

export interface SampleVideo {
  sample_key: string;
  name: string;
  ground_truth: 'REAL' | 'FAKE';
  description: string;
  duration: string;
  url: string;
}
