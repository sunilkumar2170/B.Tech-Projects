import axios from 'axios';
import type { VideoMetadata, AnalysisStatus, AnalysisResult, SampleVideo, HealthResponse } from '../types';

const API_BASE = '/api';

export const api = {
  async getHealth(): Promise<HealthResponse> {
    const res = await axios.get(`${API_BASE}/health`);
    return res.data;
  },

  async getSampleVideos(): Promise<{ samples: SampleVideo[] }> {
    const res = await axios.get(`${API_BASE}/videos/samples`);
    return res.data;
  },

  async selectSample(sampleKey: string): Promise<{ success: boolean; video: VideoMetadata }> {
    const res = await axios.post(`${API_BASE}/videos/sample/${sampleKey}`);
    return res.data;
  },

  async uploadVideo(
    file: File,
    onProgress?: (percent: number) => void
  ): Promise<{ success: boolean; video: VideoMetadata }> {
    const formData = new FormData();
    formData.append('file', file);

    const res = await axios.post(`${API_BASE}/videos/upload`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        if (progressEvent.total && onProgress) {
          const percent = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          onProgress(percent);
        }
      },
    });
    return res.data;
  },

  async startAnalysis(videoId: string, sampleKey?: string): Promise<{ analysis_id: string }> {
    const params = sampleKey ? { sample_key: sampleKey } : {};
    const res = await axios.post(`${API_BASE}/analysis/${videoId}`, null, { params });
    return res.data;
  },

  async getAnalysisStatus(analysisId: string): Promise<AnalysisStatus> {
    const res = await axios.get(`${API_BASE}/analysis/${analysisId}/status`);
    return res.data;
  },

  async getAnalysisResult(analysisId: string): Promise<AnalysisResult> {
    const res = await axios.get(`${API_BASE}/analysis/${analysisId}/result`);
    return res.data;
  },

  getDownloadReportUrl(analysisId: string): string {
    return `${API_BASE}/reports/${analysisId}/download`;
  },
};
